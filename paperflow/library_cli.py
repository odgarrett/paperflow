"""Commands for your own relationship with papers: review, mark, fulltext."""
from __future__ import annotations

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import Config


def _store(cfg):
    from .store import Store
    return Store(cfg.database)


# -- review / mark -------------------------------------------------------------------------

def cmd_review(cfg: Config, args) -> None:
    from .reading import briefing_papers, review_loop, select_papers
    store = _store(cfg)
    numbers = None
    if args.briefing:
        name, pids = briefing_papers(store, args.briefing)
        numbers = {pid: i for i, pid in enumerate(pids, 1)}
        print(f"Reviewing {len(pids)} papers from {name}, in the order they were presented.")
    else:
        labels = args.label or (None if (args.state or args.new) else ["mandatory"])
        pids = select_papers(store, labels, args.state, args.new, include_marked=args.all)
        what = ", ".join(labels or ["any label"])
        note = "" if (args.all or args.state) else " not yet marked (rerun to resume; --all to include marked ones)"
        print(f"Reviewing {len(pids)} papers ({what}){note}.")
    if not pids:
        store.close()
        return
    try:
        changed = review_loop(store, pids, numbers)
    except KeyboardInterrupt:
        store.commit()
        changed = None
    store.close()
    if changed:
        print(f"\nDone: {changed['state']} reading state(s), {changed['label']} label(s), "
              f"{changed['type']} article type(s) changed.")
    else:
        print("\nStopped; everything marked so far is saved.")


def cmd_mark(cfg: Config, args) -> None:
    from .reading import history, set_state
    from .triage_cli import _find_paper, _short
    store = _store(cfg)
    pid = _find_paper(store, args.paper)
    if pid is None:
        raise SystemExit(f"No paper with id or DOI {args.paper!r} in the database.")
    set_state(store, pid, args.state, args.note)
    store.commit()
    title = store.db.execute("SELECT title FROM papers WHERE id=?", (pid,)).fetchone()[0]
    hist = history(store, pid)
    store.close()
    trail = " → ".join(f"{s} ({t[:10]})" for s, t in hist)
    print(f"[{pid}] {_short(title)}\n  {trail}")


# -- zotero --------------------------------------------------------------------------------

def zotero_pending(store, include_sent: bool = False) -> list[dict]:
    from .reading import current_states
    sent = {pid for (pid,) in store.db.execute("SELECT paper_id FROM zotero_sent")}
    out = []
    for pid, state in current_states(store).items():
        if state != "saved" or (pid in sent and not include_sent):
            continue
        r = store.db.execute("SELECT title, doi FROM papers WHERE id=?", (pid,)).fetchone()
        if r:
            out.append({"id": pid, "title": r[0], "doi": r[1], "sent": pid in sent})
    return sorted(out, key=lambda x: x["id"])


def cmd_zotero(cfg: Config, args, ask=input) -> None:
    """Saved papers not yet in Zotero: DOIs to paste, then record them as sent."""
    import sys
    from .store import utcnow
    store = _store(cfg)
    papers = zotero_pending(store, include_sent=args.all)
    if not papers:
        store.close()
        print("Nothing to add: no saved papers waiting for Zotero." if not args.all else "No saved papers.")
        return
    with_doi = [p for p in papers if p["doi"]]
    without = [p for p in papers if not p["doi"]]
    if with_doi:
        print(f"In Zotero, click Add Item by Identifier (the magic wand) and paste these {len(with_doi)} DOI(s):\n")
        print("\n".join(p["doi"] for p in with_doi))
    if without:
        print(f"\n{len(without)} saved paper(s) have no DOI; add them by hand:")
        for p in without:
            print(f"  [{p['id']}] {p['title']}")
    pending = [p for p in papers if not p["sent"]]
    if not pending or args.dry_run:
        store.close()
        return
    record = args.mark_sent
    if not record and sys.stdin.isatty():
        try:
            answer = ask(f"\nOnce they're in Zotero, press Enter to record these {len(pending)} as added "
                         "(or type n to leave them for next time): ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        record = answer.strip().lower() not in ("n", "no")
    if record:
        now = utcnow()
        for p in pending:
            store.db.execute("INSERT OR REPLACE INTO zotero_sent VALUES (?, ?)", (p["id"], now))
        store.commit()
        print(f"Recorded {len(pending)} paper(s) as added to Zotero; they won't be listed again "
              "(`paperflow zotero --all` lists every saved paper).")
    else:
        print("Not recorded; they'll be listed again next time.")
    store.close()


# -- fulltext ------------------------------------------------------------------------------

def cmd_fulltext(cfg: Config, args) -> None:
    from .fulltext import Fetcher, save, scan_drop_folder
    from .http import Http
    from .triage import final_labels
    store = _store(cfg)
    labels = args.label or ["mandatory"]

    added, problems = scan_drop_folder(store, cfg.fulltext_dir)
    if added or problems:
        print(f"Drop folder {cfg.fulltext_dir}: {added} new PDF(s) matched to papers.")
        for line in problems:
            print(f"  not used: {line}")
    store.commit()

    finals = final_labels(store)
    wanted = [pid for pid, f in finals.items() if f["label"] in labels]
    status = {pid: st for pid, st in store.db.execute("SELECT paper_id, status FROM fulltexts")}

    def needs(pid):
        st = status.get(pid)
        return st is None or st == "error" or (st == "unavailable" and args.retry)

    if args.missing:
        _print_missing(store, [p for p in wanted if status.get(p) != "ok"], cfg)
        store.close()
        return

    todo = [pid for pid in wanted if needs(pid)]
    if todo:
        papers = []
        for pid in todo:
            p = dict(store.db.execute("SELECT id, doi, published_doi, title, oa_pdf_url FROM papers WHERE id=?",
                                      (pid,)).fetchone())
            r = store.db.execute("SELECT value FROM identifiers WHERE paper_id=? AND scheme='pmid'", (pid,)).fetchone()
            p["pmid"] = r[0] if r else None
            papers.append(p)
        fetcher = Fetcher(Http(cfg.cache, cfg.contact_email, cfg.cache_days), cfg.contact_email)
        print(f"Looking for open-access full texts of {len(papers)} paper(s)… (Ctrl+C is safe)")
        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers or cfg.workers) as pool:
            futures = {pool.submit(fetcher.fetch, p): p for p in papers}
            bar = tqdm(as_completed(futures), total=len(futures), unit="paper", dynamic_ncols=True)
            try:
                for fut in bar:
                    ft = fut.result()
                    save(store, futures[fut]["id"], ft)
                    key = ft.source if ft.status == "ok" else ft.status
                    counts[key] = counts.get(key, 0) + 1
                    store.commit()
            except KeyboardInterrupt:
                bar.close()
                print("\nStopping: saving progress…")
                for f in futures:
                    f.cancel()
        store.commit()
        print("Results: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
        if counts.get("error"):
            print("Lookups that failed (network or rate limits) are retried the next time you run this.")
        for host, reason in getattr(fetcher.http, "skipped_hosts", {}).items():
            print(f"  Stopped contacting {host} during this run: {reason}.")

    have = store.db.execute(
        f"SELECT count(*) FROM fulltexts WHERE status='ok' AND paper_id IN ({','.join('?' * len(wanted))})",
        wanted).fetchone()[0] if wanted else 0
    print(f"\nFull text for {have} of {len(wanted)} {'/'.join(labels)} paper(s).")
    if have < len(wanted):
        print(f"`paperflow fulltext --missing` lists the rest. Put any PDFs you have for them into "
              f"{cfg.fulltext_dir} (any file names) and run `paperflow fulltext` again.")
    store.close()


def _print_missing(store, pids: list[int], cfg) -> None:
    if not pids:
        print("Every selected paper has a full text.")
        return
    print(f"{len(pids)} paper(s) without full text (briefed from the abstract unless you add a PDF to "
          f"{cfg.fulltext_dir}):\n")
    for pid in pids:
        title, doi = store.db.execute("SELECT title, doi FROM papers WHERE id=?", (pid,)).fetchone()
        why = store.db.execute("SELECT detail FROM fulltexts WHERE paper_id=?", (pid,)).fetchone()
        print(f"  [{pid}] {doi or '(no DOI)'}\n      {textwrap.shorten(title or '', 100)}")
        if why and why[0]:
            print(f"      ({textwrap.shorten(why[0], 110)})")


# -- registration --------------------------------------------------------------------------

def register(sub) -> None:
    states = ["new", "familiar", "read", "saved"]
    p = sub.add_parser("review", help="go through papers one keypress at a time: reading state, label, type")
    p.add_argument("--label", action="append",
                   choices=["mandatory", "interest", "skip", "filtered", "pending_llm", "unscored"],
                   help="papers with this triage label (repeatable; default mandatory)")
    p.add_argument("--state", action="append", choices=states + ["unmarked"],
                   help="papers in this reading state (repeatable)")
    p.add_argument("--new", action="store_true", help="only papers added by the most recent ingest")
    p.add_argument("--briefing", help="the papers of a briefing, in order: 'latest', a number or a name")
    p.add_argument("--all", action="store_true", help="include papers you've already marked")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("mark", help="set your reading state of a paper: new, familiar, read or saved")
    p.add_argument("paper", help="paper id (from exports and briefings) or DOI")
    p.add_argument("state", choices=states)
    p.add_argument("--note")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("zotero", help="DOIs of saved papers not yet in Zotero, to paste; then record them as added")
    p.add_argument("--all", action="store_true", help="list every saved paper, including ones already added")
    p.add_argument("--mark-sent", action="store_true", help="record them as added without asking")
    p.add_argument("--dry-run", action="store_true", help="only list them; record nothing")
    p.set_defaults(func=cmd_zotero)

    p = sub.add_parser("fulltext", help="fetch open-access full texts (default: mandatory papers)")
    p.add_argument("--label", action="append", choices=["mandatory", "interest", "skip"],
                   help="which papers (repeatable; default mandatory)")
    p.add_argument("--missing", action="store_true", help="list papers still without full text; fetch nothing")
    p.add_argument("--retry", action="store_true", help="look again for papers where nothing was found before")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_fulltext)
