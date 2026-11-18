"""`paperflow brief`: build a briefing (EPUB, markdown, audio scripts, audio)."""
from __future__ import annotations

import re
import textwrap
from datetime import date
from pathlib import Path

from tqdm import tqdm

from .config import Config


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:n] or "section"


def _select(store, args) -> list[int]:
    from .triage import final_labels
    from .triage_cli import _find_paper
    if args.papers:
        pids = []
        for ref in re.split(r"[,\s]+", args.papers.strip()):
            pid = _find_paper(store, ref) if ref else None
            if pid is None and ref:
                raise SystemExit(f"No paper with id or DOI {ref!r}.")
            if pid:
                pids.append(pid)
        return pids
    return [pid for pid, f in final_labels(store).items() if f["label"] == "mandatory"]


def _meta_line(p) -> str:
    a = p.authors
    who = (a[0] + (" … " + a[-1] if len(a) > 2 else (", " + a[1] if len(a) == 2 else ""))) if a else ""
    bits = [who, " ".join(str(x) for x in (p.venue, p.year) if x)]
    if p.is_preprint:
        bits.append("preprint")
    if p.article_type and p.article_type not in ("research", "unknown"):
        bits.append(p.article_type)
    if p.notes_source != "full text":
        bits.append(f"from the {p.notes_source or 'abstract'}")
    if p.state not in ("new", "unmarked"):
        bits.append(f"you: {p.state}")
    bits.append(f"id {p.id}")
    if p.doi:
        bits.append(p.doi)
    return " · ".join(b for b in bits if b)


def _limit(sections, limit):
    keep, n = [], 0
    for project, ps in sections:
        ps = ps[: max(0, limit - n)]
        n += len(ps)
        if ps:
            keep.append((project, ps))
    return keep


def _renumber(groups, start: int = 0) -> int:
    n = start
    for _, ps in groups:
        for p in ps:
            n += 1
            p.number = n
    return n


def cmd_brief(cfg: Config, args) -> None:
    from .briefing import (WINDOW_KEY, Writer, chunk_chars_for, chunks_for, gather, new_papers,
                           overview_batches, overview_chars_for, register)
    from .store import Store, utcnow
    run_started = utcnow()
    store = Store(cfg.database)
    profile = cfg.profile.read_text(encoding="utf-8") if cfg.profile.exists() else ""
    weekly = bool(getattr(args, "new", False))
    with_overview = weekly and not getattr(args, "no_overview", False)

    if weekly:
        sel = new_papers(store)
        if sel["untriaged"] and not getattr(args, "force", False):
            store.close()
            raise SystemExit(f"{len(sel['untriaged'])} paper(s) added since the last weekly briefing haven't been "
                             "triaged yet, so they can't be sorted into this one. Run `paperflow triage` first "
                             "(or add --force to brief without them; they'd be left out for good).")
        mand_pids, int_pids = sel["mandatory"], (sel["interest"] if with_overview else [])
        if not mand_pids and not int_pids:
            store.close()
            print(f"Nothing new to brief since {(sel['since'] or 'the start')[:16].replace('T', ' ')}.")
            return
    else:
        mand_pids, int_pids = _select(store, args), []
        if not mand_pids:
            store.close()
            raise SystemExit("No papers to brief (no mandatory papers yet? Run `paperflow triage` first).")

    sections = gather(store, mand_pids, profile) if mand_pids else []
    if args.limit:
        sections = _limit(sections, args.limit)
    papers = [p for _, ps in sections for p in ps]
    topics = gather(store, int_pids, profile, start=len(papers)) if int_pids else []

    from .classify import make_backend, unload
    from types import SimpleNamespace
    backend = None if args.dry_run else make_backend(cfg)
    sizing = backend or SimpleNamespace(num_ctx=cfg.ollama_num_ctx if cfg.llm_provider == "ollama" else None)
    chunk_chars, ov_chars = chunk_chars_for(sizing), overview_chars_for(sizing)

    if args.dry_run:
        _dry_run(cfg, sections, topics, chunk_chars, ov_chars)
        store.close()
        return

    from .classify import ModelUnavailable
    writer = Writer(store, backend, profile, chunk_chars)
    failed: dict[int, str] = {}
    overviews: dict[str, str] = {}
    try:
        for stage, fn in (("notes", lambda p: writer.notes(p, redo=args.redo)),
                          ("writing", lambda p: writer.write(p, "mandatory", redo=args.redo or args.rewrite))):
            for p in (tqdm(papers, desc=stage, unit="paper", dynamic_ncols=True) if papers else []):
                if p.id in failed:
                    continue
                try:
                    fn(p)
                except (KeyboardInterrupt, SystemExit, ModelUnavailable):
                    raise
                except Exception as e:
                    failed[p.id] = f"{e.__class__.__name__}: {e}"
        # Numbers are final once failures are dropped; overviews mention papers by number.
        sections = [(proj, [p for p in ps if p.id not in failed]) for proj, ps in sections]
        sections = [(proj, ps) for proj, ps in sections if ps]
        n_mand = _renumber(sections)
        _renumber(topics, n_mand)
        batches = [(topic, b) for topic, ps in topics for b in overview_batches(ps, ov_chars)]
        for topic, batch in (tqdm(batches, desc="overviews", unit="group", dynamic_ncols=True) if batches else []):
            try:
                text = writer.overview(topic, batch, redo=args.redo or args.rewrite)
                overviews[topic] = (overviews[topic] + "\n\n" + text) if topic in overviews else text
            except (KeyboardInterrupt, SystemExit, ModelUnavailable):
                raise
            except Exception as e:
                for p in batch:
                    failed[p.id] = f"overview: {e.__class__.__name__}: {e}"
    except ModelUnavailable as e:
        store.close()
        done = sum(1 for p in papers if p.notes is not None)
        print(f"\nStopped: {e}\n(Notes were finished for {done} of {len(papers)} paper(s) before it stopped.)")
        return
    except KeyboardInterrupt:
        store.close()
        print("\nStopped. Notes, write-ups and overviews made so far are saved; run the same command to continue.")
        return

    # Interest papers whose overview failed are left out too; keep the numbering contiguous.
    topics = [(t, [p for p in ps if p.id not in failed]) for t, ps in topics]
    topics = [(t, ps) for t, ps in topics if ps and t in overviews]
    papers = [p for _, ps in sections for p in ps]
    interest = [p for _, ps in topics for p in ps]
    if not papers and not interest:
        store.close()
        raise SystemExit("Every paper failed; nothing to assemble. First error: " + next(iter(failed.values())))

    pilot = bool(args.papers or args.limit)
    today = date.today()
    if weekly:
        name = args.name or f"weekly-{today.isoformat()}"
        title = f"Weekly briefing · {today.strftime('%d %B %Y')}"
    else:
        name = args.name or f"mandatory-{today.isoformat()}{'-pilot' if pilot else ''}"
        title = f"Mandatory papers · {today.strftime('%d %B %Y')}" + (" (pilot)" if pilot else "")
    out = cfg.briefings_dir / name
    out.mkdir(parents=True, exist_ok=True)

    n_full = sum(p.notes_source == "full text" for p in papers)
    front = []
    if papers:
        front.append(f"{len(papers)} mandatory paper(s) in {len(sections)} section(s), written by {writer.model} "
                     f"from {n_full} full text(s) and {len(papers) - n_full} abstract(s).")
    if interest:
        front.append(f"Then {len(interest)} paper(s) of interest, in short overviews by topic written from their "
                     "abstracts.")
    front += [
        "Papers new to you get a full briefing, papers you know the gist of get a short refresher, and papers "
        "you've read get a one-line reminder. Each paper's line under its title says where its briefing came from.",
        "Papers are numbered as in the audio version. Afterwards, `paperflow review --briefing latest` walks "
        "through them in the same order so you can mark what to save.",
    ]

    # Markdown + EPUB
    from .epub import Chapter, Item, write_epub
    md = [f"# {title}", *front]
    chapters = []
    for project, ps in sections:
        md.append(f"## {project}")
        ch = Chapter(project)
        for p in ps:
            paras = [x for x in p.text.split("\n\n") if x.strip()]
            md += [f"### {p.number}. {p.title}", f"*{_meta_line(p)}*", *paras]
            ch.items.append(Item(f"p{p.number}", f"{p.number}. {p.title}", _meta_line(p), paras))
        chapters.append(ch)
    for topic, ps in topics:
        paras = [x for x in overviews[topic].split("\n\n") if x.strip()]
        md += [f"## Also of interest: {topic}", *paras, "Papers in this overview:"]
        md += [f"- **{p.number}. {p.title}** · {_meta_line(p)}" for p in ps]
        chapters.append(Chapter(f"Also of interest: {topic}", paragraphs=paras,
                                items=[Item(f"p{p.number}", f"{p.number}. {p.title}", _meta_line(p), []) for p in ps]))
    (out / "briefing.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    write_epub(out / f"{name}.epub", title, front, chapters)

    # Audio script (one for the whole briefing: a chapter per mandatory paper, one per overview topic)
    from .speech import LEXICON_TEMPLATE, load_lexicon, speakable, spoken_authors
    from .tts import encode, render_to_wav, script_to_text
    if not cfg.pronunciations.exists():
        cfg.pronunciations.write_text(LEXICON_TEMPLATE, encoding="utf-8")
    lexicon = load_lexicon(cfg.pronunciations)
    summary = []
    if papers:
        summary.append(f"{len(papers)} mandatory paper{'s' if len(papers) != 1 else ''}")
    if interest:
        summary.append(f"{len(interest)} more of interest")
    script = [("chapter", "Introduction"),
              ("text", speakable(f"{title.replace('·', '.')}. {', then '.join(summary)}.", lexicon)), ("pause", 1.0)]
    for i, (project, ps) in enumerate(sections, 1):
        for k, p in enumerate(ps):
            script.append(("chapter", f"Paper {p.number}: {p.title}"))
            if k == 0:  # the section is announced at the start of its first paper's chapter
                script += [("text", speakable(f"Section {i}: {project}.", lexicon)), ("pause", 1.0)]
            intro = [spoken_authors(p.authors)]
            if p.venue:
                intro.append(f"in {p.venue}")
            if p.year:
                intro.append(str(p.year))
            line = ", ".join(x for x in intro if x)
            extra = (" A preprint." if p.is_preprint else "") + (
                " This briefing is based on the abstract only." if p.notes_source != "full text" else "")
            script += [("text", speakable(f"Paper {p.number}. {p.title}.", lexicon)), ("pause", 0.5),
                       ("text", speakable(f"{line}.{extra}", lexicon) if line else speakable(extra, lexicon)),
                       ("pause", 0.7)]
            for para in p.text.split("\n\n"):
                if para.strip():
                    script += [("text", speakable(para, lexicon)), ("pause", 0.35)]
            script.append(("pause", 1.3))
    for topic, ps in topics:
        script += [("chapter", f"Also of interest: {topic}"),
                   ("text", speakable(f"Also of interest: {topic}.", lexicon)), ("pause", 1.0)]
        for para in overviews[topic].split("\n\n"):
            if para.strip():
                script += [("text", speakable(para, lexicon)), ("pause", 0.4)]
        script.append(("pause", 1.3))
    for old in (out / "scripts", out / "audio"):  # layout of earlier versions: one file per section
        if old.is_dir():
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
    (out / "script.txt").write_text(script_to_text(script), encoding="utf-8")

    audio_note = ""
    if not args.no_audio and cfg.tts == "kokoro":
        unload(backend)  # give the GPU memory back before loading the voice model
        from .tts import Kokoro
        voice = args.voice or cfg.voice
        tts = Kokoro(voice, cfg.speed)
        n_ch = sum(1 for kind, _ in script if kind == "chapter")
        bar = tqdm(total=n_ch, desc="audio", unit="chapter", dynamic_ncols=True)
        chapters_at = render_to_wav(tts, script, out / f"{name}.wav", progress=lambda: bar.update(1))
        bar.close()
        made = encode(out / f"{name}.wav", out / name, cfg.audio_format, title, name, chapters_at)
        audio_note = f"\n  audio:  {made} ({len(chapters_at)} chapters, voice {voice})"
    elif cfg.tts == "none" or args.no_audio:
        audio_note = "\n  audio:  skipped (the script is in script.txt)"

    register(store, name, "weekly" if weekly else "mandatory", sections + topics, str(out))
    if weekly:
        # Move the window on, but never past a paper that was left out: the next run picks it up
        # (papers already in a briefing are excluded anyway).
        left_out = [r[0] for r in store.db.execute(
            f"SELECT added_at FROM papers WHERE id IN ({','.join('?' * len(failed))})", list(failed)) if r[0]] if failed else []
        store.set_meta(WINDOW_KEY, min([run_started] + left_out))
        store.commit()
    store.close()

    print(f"\nBriefing '{name}': {len(papers)} mandatory, {len(interest)} of interest; "
          f"{writer.calls} model call(s) this run.")
    print(f"  ebook:  {out / (name + '.epub')}\n  text:   {out / 'briefing.md'}\n  script: {out / 'script.txt'}" + audio_note)
    if writer.recovered:
        print(f"\n{writer.recovered} notes reply/replies ran past the length limit and were recovered "
              "(redone in smaller pieces, or the complete part kept).")
    if writer.issues:
        print(f"\nNotes on this run ({len(writer.issues)}):")
        for line in writer.issues[:10]:
            print(f"  {line}")
        if writer.debug_files:
            print(f"  The raw replies are in {writer.debug_files[0].parent} if you want to see what went wrong.")
    if failed:
        print(f"\n{len(failed)} paper(s) failed and were left out (the next run tries them again):")
        for pid, err in list(failed.items())[:10]:
            print(f"  [{pid}] {textwrap.shorten(err, 110)}")
    print("\nAfterwards: `paperflow review --briefing latest` to mark what to save, then `paperflow zotero`. "
          f"Add pronunciation fixes to {cfg.pronunciations.name}.")


def _dry_run(cfg, sections, topics, chunk_chars, ov_chars) -> None:
    from .briefing import chunks_for, overview_batches
    papers = [p for _, ps in sections for p in ps]
    calls = 0
    if papers:
        print(f"{len(papers)} mandatory paper(s) in {len(sections)} section(s); full texts split into pieces of "
              f"up to {chunk_chars:,} characters.\n")
    for project, ps in sections:
        print(f"{project}")
        for p in ps:
            n = len(chunks_for(p.fulltext, chunk_chars)) if p.fulltext else 1
            calls += n + 1
            src = f"from full text ({n} piece{'s' if n > 1 else ''})" if p.fulltext else (
                "from abstract" if p.abstract else "from title only")
            length = {"full": "full briefing", "refresher": "refresher", "reminder": "reminder"}[p.length]
            print(f"  {p.number:3d}. id {p.id:<6} {length:<13} {src:<26} {textwrap.shorten(p.title, 70)}")
    if topics:
        print("\nAlso of interest (overviews written from abstracts):")
        for topic, ps in topics:
            nb = len(overview_batches(ps, ov_chars))
            calls += nb
            print(f"  {topic}: {len(ps)} paper(s), {nb} overview request(s)")
    print(f"\nAt most {calls} model calls (notes, write-ups and overviews already made are reused).")
    n_abs = sum(1 for p in papers if not p.fulltext)
    if n_abs:
        print(f"{n_abs} mandatory paper(s) would be briefed without a full text. If you have PDFs for them, put them "
              f"in {cfg.fulltext_dir} and run `paperflow fulltext`; name a file id<number>.pdf to assign it directly.")


def register_cli(sub) -> None:
    p = sub.add_parser("brief", help="write a briefing of your mandatory papers (EPUB, text, audio)")
    p.add_argument("--new", action="store_true",
                   help="the weekly briefing: papers added since the last one, mandatory papers in full plus "
                        "short overviews of the papers of interest")
    p.add_argument("--no-overview", action="store_true", help="with --new: leave out the papers of interest")
    p.add_argument("--force", action="store_true",
                   help="with --new: brief even if some new papers haven't been triaged (they'd be left out)")
    p.add_argument("--papers", help="only these papers, by id or DOI, comma-separated (e.g. for a pilot)")
    p.add_argument("--limit", type=int, help="only the first N papers in briefing order")
    p.add_argument("--name", help="name of the briefing (default: mandatory-<date>)")
    p.add_argument("--dry-run", action="store_true", help="show which papers, in what order and how, then stop")
    p.add_argument("--no-audio", action="store_true", help="make the EPUB, text and audio script only")
    from .tts import VOICES_HINT
    p.add_argument("--voice", help=f"Kokoro voice for this run (default from config). {VOICES_HINT}")
    p.add_argument("--rewrite", action="store_true", help="rewrite the write-ups (keep the notes)")
    p.add_argument("--redo", action="store_true", help="redo the notes and the write-ups")
    p.set_defaults(func=cmd_brief)
