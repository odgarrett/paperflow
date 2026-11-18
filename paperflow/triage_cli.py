"""Commands for stage 2 (triage): init-triage, examples, filter, classify, triage, label."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from .config import Config


def _store(cfg):
    from .store import Store
    return Store(cfg.database)


def _resolver(cfg):
    from .http import Http
    from .resolver import Resolver
    return Resolver(Http(cfg.cache, cfg.contact_email, cfg.cache_days), cfg)


# -- init-triage ---------------------------------------------------------------------------

def cmd_init(cfg: Config, args) -> None:
    from .templates import EXAMPLES_README, LABEL_GUIDANCE, PROFILE_TEMPLATE
    made = []
    for label in ("mandatory", "interest", "skip"):
        d = cfg.examples_dir / label
        if not d.exists():
            d.mkdir(parents=True)
            made.append(str(d))
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(LABEL_GUIDANCE[label] + "\n\n" + EXAMPLES_README, encoding="utf-8")
    if not cfg.profile.exists():
        text = PROFILE_TEMPLATE
        if cfg.database.exists():  # remind them what they've been subscribing to
            store = _store(cfg)
            rows = store.db.execute("SELECT alert_name, count(DISTINCT paper_id) n FROM sightings "
                                    "WHERE alert_name IS NOT NULL GROUP BY alert_name ORDER BY n DESC LIMIT 30").fetchall()
            store.close()
            if rows:
                text += "\n<!--\nYour alerts, for reference (papers found):\n"
                text += "\n".join(f"  {n:5d}  {name}" for name, n in rows) + "\n-->\n"
        cfg.profile.write_text(text, encoding="utf-8")
        made.append(str(cfg.profile))
    print("Created:\n  " + "\n  ".join(made) if made else "Everything already exists; nothing changed.")
    print(textwrap.dedent(f"""
        Next:
          1. Add example papers to {cfg.examples_dir.name}/mandatory, /interest and (optionally) /skip.
             Easiest: in Zotero, right-click a collection > Export Collection… > BibTeX, saved there.
          2. Fill in {cfg.profile.name}.
          3. paperflow examples      (check what was found)
          4. paperflow filter        (embedding filter)
          5. paperflow classify      (LLM; try --dry-run first)"""))


# -- examples ------------------------------------------------------------------------------

def cmd_examples(cfg: Config, args) -> None:
    from .triage import sync_examples
    store = _store(cfg)
    summary, problems = sync_examples(store, cfg.examples_dir, None if args.offline else _resolver(cfg))
    print("Examples: " + ", ".join(f"{k}: {summary.get(k, 0)}" for k in ("mandatory", "interest", "skip")))
    by_file = store.db.execute("SELECT label, source, count(*), sum(abstract IS NOT NULL) FROM examples "
                               "GROUP BY label, source ORDER BY label, source").fetchall()
    for label, src, n, n_abs in by_file:
        print(f"  [{label}] {src}: {n} paper(s), {n_abs} with abstract")
    linked = store.db.execute("SELECT count(*) FROM judgments WHERE source LIKE 'example:%'").fetchone()[0]
    if linked:
        print(f"{linked} example(s) are also papers from your alerts; they count as your labels for those papers.")
    for p in problems:
        print("  note:", p)
    n_pos = (summary.get("mandatory", 0) + summary.get("interest", 0))
    if n_pos < 20:
        print(f"\nTip: with {n_pos} positive examples the filter works, but 30+ (mandatory + interest) "
              "make its threshold much more reliable.")
    store.close()


# -- filter --------------------------------------------------------------------------------

def _short(t: str | None, n: int = 95) -> str:
    t = t or "(no title)"
    return t if len(t) <= n else t[: n - 1] + "…"


def release_gpu() -> None:
    """Hand GPU memory held by this process (the embedding model) back to the driver, so Ollama can
    load its model next. PyTorch otherwise keeps it reserved until the process exits."""
    import gc
    import sys
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def cmd_filter(cfg: Config, args) -> None:
    try:
        _filter(cfg, args)
    finally:
        release_gpu()


def _filter(cfg: Config, args) -> None:
    from .embed import load_backend
    from .triage import run_filter, sync_examples
    store = _store(cfg)
    summary, problems = sync_examples(store, cfg.examples_dir, _resolver(cfg))
    for p in problems:
        print("  note:", p)
    backend_name = args.embedding or cfg.embedding
    print(f"Embedding with {backend_name} (first run embeds every paper; later runs only new ones)…")
    backend = load_backend(backend_name)
    r = run_filter(store, backend, keep_recall=args.keep_recall or cfg.keep_recall,
                   skip_margin=cfg.skip_margin, threshold=args.threshold)
    pct = 100 * r.n_pass / max(r.n_scored, 1)
    print(f"\nScored {r.n_scored} papers against {r.n_pos} positive and {r.n_skip} skip examples.")
    print(f"Passed: {r.n_pass} ({pct:.1f}%), of which {r.newly_passed} newly. Threshold {r.threshold:.3f}.")
    print(f"  Calibration: {100 * r.recall_est:.0f}% of your positive examples would pass "
          f"(each scored against the others).")
    if r.skip_pass_rate is not None:
        print(f"  {100 * r.skip_pass_rate:.0f}% of your skip examples would pass "
              "(these then rely on the LLM stage to be caught).")
    if r.n_vetoed:
        print(f"  {r.n_vetoed} paper(s) above the threshold were dropped for being closer to your skip examples.")
    if pct > 30:
        print("  That's a lot to send to the LLM. More (and more specific) examples, skip examples for the "
              "noise you see, or a lower keep_recall in config.yaml will tighten it.")

    rows = store.db.execute(
        "SELECT p.title, a.score, a.detail FROM assessments a JOIN papers p ON p.id=a.paper_id "
        "WHERE a.stage='embed' AND a.label='pass' ORDER BY a.score DESC").fetchall()
    if rows:
        print("\nHighest scoring:")
        for t, sc, _ in rows[:5]:
            print(f"  {sc:.3f}  {_short(t)}")
        print("Just above the threshold (the filter's borderline):")
        for t, sc, _ in rows[-5:]:
            print(f"  {sc:.3f}  {_short(t)}")
    drop = store.db.execute(
        "SELECT p.title, a.score FROM assessments a JOIN papers p ON p.id=a.paper_id "
        "WHERE a.stage='embed' AND a.label='drop' AND a.score < ? ORDER BY a.score DESC LIMIT 5",
        (r.threshold,)).fetchall()
    if drop:
        print("Just below the threshold (dropped):")
        for t, sc in drop:
            print(f"  {sc:.3f}  {_short(t)}")
    vetoed = store.db.execute(
        "SELECT p.title, a.score FROM assessments a JOIN papers p ON p.id=a.paper_id "
        "WHERE a.stage='embed' AND a.label='drop' AND a.score >= ? ORDER BY a.score DESC LIMIT 3",
        (r.threshold,)).fetchall()
    if vetoed:
        print("Dropped as closer to your skip examples (top 3):")
        for t, sc in vetoed:
            print(f"  {sc:.3f}  {_short(t)}")
    store.close()


# -- classify ------------------------------------------------------------------------------

def cmd_classify(cfg: Config, args) -> None:
    from .classify import Classifier, build_system, candidates, example_titles, make_backend, paper_message, run_classification
    if not cfg.profile.exists():
        raise SystemExit(f"{cfg.profile} not found. Run `paperflow init-triage` and fill it in first.")
    store = _store(cfg)
    items = candidates(store, redo=args.redo, limit=args.limit)
    model = f"{cfg.llm_provider}:{args.model or cfg.llm_model}"
    if args.dry_run:
        system = build_system(cfg.profile.read_text(encoding="utf-8"), example_titles(store))
        print(f"{len(items)} paper(s) would be sent to {model}.")
        if items:
            print("\n===== fixed part (cached; same for every paper) =====\n" + system[0]["text"])
            print("\n===== per-paper part (first paper) =====\n" + paper_message(items[0]))
            if items[0].get("article_type"):
                print(f"[not sent: article type decided by {items[0].get('article_type_source') or '-'}; "
                      f"paper id {items[0]['id']}]")
            n_chars = len(system[0]["text"])
            print(f"\nFixed part ≈ {n_chars // 4:,} tokens; per paper ≈ {len(paper_message(items[0])) // 4:,} "
                  "tokens in, ≈100 out.")
            if cfg.llm_provider == "ollama":
                need = (n_chars + len(paper_message(items[0]))) // 3 + 500
                fits = "fits" if need <= cfg.ollama_num_ctx else "does NOT fit; raise ollama_num_ctx"
                print(f"Context needed ≈ {need:,} tokens; ollama_num_ctx is {cfg.ollama_num_ctx:,} ({fits}).")
        store.close()
        return
    if not items:
        print("Nothing to classify (run `paperflow filter` first, or use --redo).")
        store.close()
        return
    print(f"Classifying {len(items)} paper(s) with {model}… (Ctrl+C is safe)")
    clf = Classifier(make_backend(cfg, args.model))
    stats = run_classification(store, clf, cfg.profile, items, workers=cfg.llm_workers)
    store.close()
    print("Results: " + ", ".join(f"{k}: {stats.counts.get(k, 0)}" for k in ("mandatory", "interest", "skip")))
    if cfg.llm_provider == "anthropic":
        print(f"Tokens: {stats.input_tokens:,} input, {stats.cache_read_tokens:,} read from cache, "
              f"{stats.cache_write_tokens:,} written to cache, {stats.output_tokens:,} output.")
    else:
        print(f"Tokens processed locally: {stats.input_tokens:,} in, {stats.output_tokens:,} out.")
    if stats.types_corrected:
        from collections import Counter
        moves = Counter(f"{old or 'unknown'} → {new}" for _, old, new in stats.types_corrected)
        print(f"Article types corrected after reading the abstract: {len(stats.types_corrected)} "
              f"({', '.join(f'{m}: {n}' for m, n in moves.most_common())}). "
              "`paperflow set-type` fixes any that are wrong.")
    if stats.errors:
        print(f"{len(stats.errors)} paper(s) failed (they'll be retried next run). First error: {stats.errors[0][1]}")


def cmd_triage(cfg: Config, args) -> None:
    cmd_filter(cfg, argparse.Namespace(embedding=None, keep_recall=None, threshold=None))
    print()
    cmd_classify(cfg, argparse.Namespace(redo=False, limit=args.limit, model=None, dry_run=False))


# -- label ---------------------------------------------------------------------------------

def _find_paper(store, ref: str) -> int | None:
    if ref.isdigit():
        r = store.db.execute("SELECT id FROM papers WHERE id=?", (int(ref),)).fetchone()
        return r[0] if r else None
    from .links import DOI_RE, clean_doi
    m = DOI_RE.search(ref)
    if m:
        r = store.db.execute("SELECT paper_id FROM identifiers WHERE scheme='doi' AND value=?",
                             (clean_doi(m.group(1)),)).fetchone()
        return r[0] if r else None
    return None


def cmd_label(cfg: Config, args) -> None:
    from .triage import add_judgment, final_labels
    store = _store(cfg)
    pid = _find_paper(store, args.paper)
    if pid is None:
        raise SystemExit(f"No paper with id or DOI {args.paper!r} in the database.")
    before = final_labels(store)[pid]
    add_judgment(store, pid, args.label, args.note)
    title = store.db.execute("SELECT title FROM papers WHERE id=?", (pid,)).fetchone()[0]
    print(f"[{pid}] {_short(title)}\n  was: {before['label']} (by {before['by'] or '-'})  →  now: {args.label} (by you)")
    print("Your labels are used as examples from the next `paperflow filter` / `classify` run.")
    store.close()


# -- registration --------------------------------------------------------------------------

def register(sub) -> None:
    p = sub.add_parser("init-triage", help="create the examples folders and a research profile template")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("examples", help="read the examples folders and report what was found")
    p.add_argument("--offline", action="store_true", help="don't look up missing abstracts")
    p.set_defaults(func=cmd_examples)

    p = sub.add_parser("filter", help="embedding filter: score every paper against your examples")
    p.add_argument("--embedding", help="override triage.embedding (specter2 | tfidf | model name)")
    p.add_argument("--keep-recall", type=float, help="override triage.keep_recall (e.g. 0.9)")
    p.add_argument("--threshold", type=float, help="use a fixed similarity threshold instead of calibrating")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("classify", help="LLM stage: label papers that passed the filter")
    p.add_argument("--dry-run", action="store_true", help="show the prompt and how many papers, send nothing")
    p.add_argument("--limit", type=int, help="classify at most N papers (highest filter scores first)")
    p.add_argument("--redo", action="store_true", help="re-classify papers that already have a verdict")
    p.add_argument("--model", help="override triage.llm_model")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("triage", help="filter, then classify")
    p.add_argument("--limit", type=int, help="classify at most N papers")
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("label", help="record your own label for a paper (overrides the automatic ones)")
    p.add_argument("paper", help="paper id (from exports) or DOI")
    p.add_argument("label", choices=["mandatory", "interest", "skip"])
    p.add_argument("--note", help="optional note, e.g. why the automatic label was wrong")
    p.set_defaults(func=cmd_label)
