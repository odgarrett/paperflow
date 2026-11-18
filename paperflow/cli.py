"""paperflow command line.

  paperflow ingest --imap | --mbox FILE | --eml-dir DIR   parse alert emails into the database
  paperflow resolve                                       fetch metadata + abstracts for new papers
  paperflow run                                           ingest --imap, then resolve
  paperflow stats                                         what's in the database
  paperflow export --out papers.jsonl                     dump papers (with sightings) for the next stage
  paperflow parse FILE.eml ...                            dry-run the parsers on sample emails
  paperflow reparse                                       re-run parsers on all stored emails
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config import Config, load_config
from .emailsrc import iter_eml_dir, iter_imap, iter_mbox, iter_outlook_csv
from .parsers import classify_sender, parse_email

log = logging.getLogger("paperflow")


def _store(cfg: Config):
    from .store import Store
    return Store(cfg.database)


def cmd_ingest(cfg: Config, args) -> int:
    """Returns the number of new alert emails."""
    from .store import utcnow
    store = _store(cfg)
    known = store.known_message_ids()
    accept_all = args.all_senders
    run_started = utcnow()

    if args.imap:
        def accept(sender: str, subject: str) -> bool:
            return accept_all or classify_sender(sender, cfg.senders) is not None
        raws = iter_imap(cfg.imap, accept, known)
    elif args.mbox:
        raws = iter_mbox(args.mbox)
    elif getattr(args, "outlook_csv", None):
        raws = iter_outlook_csv(args.outlook_csv)
    else:
        raws = iter_eml_dir(args.eml_dir)

    n_new = n_dup = n_other = n_items = n_failed = 0
    empty: list[str] = []
    for raw in tqdm(raws, desc="emails", unit="msg", disable=not sys.stderr.isatty()):
        try:
            parsed = parse_email(raw, cfg.senders, accept_all=accept_all)
        except Exception as e:
            n_failed += 1
            log.warning("Could not parse a message: %s", e)
            continue
        if parsed is None:
            n_other += 1
            continue
        if parsed.message_id in known:
            n_dup += 1
            continue
        store.add_email(parsed, raw)
        known.add(parsed.message_id)
        n_new += 1
        n_items += len(parsed.sightings)
        if not parsed.sightings:
            empty.append(f"[{parsed.source}] {parsed.subject}")
        if n_new % 50 == 0:
            store.commit()
        if args.limit and n_new >= args.limit:
            break

    merged = store.dedupe_titles()
    if n_new:
        store.set_meta("last_ingest_started", run_started)  # used by `export --new`
    store.commit()
    papers = store.db.execute("SELECT count(*) FROM papers").fetchone()[0]
    added = store.db.execute("SELECT count(*) FROM papers WHERE added_at >= ?", (run_started,)).fetchone()[0]
    pending = store.db.execute("SELECT count(*) FROM papers WHERE status='pending'").fetchone()[0]
    store.close()
    print(f"Ingested {n_new} new alert emails ({n_items} paper sightings); skipped {n_dup} already "
          f"seen, {n_other} non-alert, {n_failed} unparseable.")
    print(f"New papers this run: {added}. Database: {papers} distinct papers, {pending} awaiting "
          f"metadata. Title merges this run: {merged}.")
    if empty:
        print(f"\n{len(empty)} alert emails yielded no papers (parser may need work for these):")
        for line in empty[:15]:
            print("  ", line)
    return n_new


def cmd_resolve(cfg: Config, args) -> None:
    from .http import Http
    from .resolver import Resolver

    store = _store(cfg)
    statuses = ["pending"]
    if args.retry_errors:
        statuses.append("error")
    if args.retry_unmatched:
        statuses.append("unmatched")
    items = store.pending_items(statuses, args.limit)
    if not items:
        print("Nothing to resolve.")
        store.close()
        return

    http = Http(cfg.cache, cfg.contact_email, cfg.cache_days)
    resolver = Resolver(http, cfg)
    workers = args.workers or cfg.workers
    counts: dict[str, int] = {}
    print(f"Resolving {len(items)} papers with {workers} workers… (Ctrl+C is safe; rerun to continue)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resolver.resolve, it) for it in items]
        bar = tqdm(as_completed(futures), total=len(futures), unit="paper", dynamic_ncols=True)
        try:
            for i, fut in enumerate(bar, 1):
                res = fut.result()
                store.apply_resolution(res)
                counts[res.status] = counts.get(res.status, 0) + 1
                bar.set_postfix({k: counts.get(k, 0) for k in ("resolved", "no_abstract", "unmatched", "error")},
                                refresh=False)
                if i % 25 == 0:
                    store.commit()
        except KeyboardInterrupt:
            bar.close()
            print("\nStopping: saving progress (waiting for lookups already in flight)…")
            for f in futures:
                f.cancel()
            store.commit()
    merged = store.dedupe_titles()
    store.commit()
    store.close()
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"Done. {summary}. Title merges after resolution: {merged}.")
    api_hosts = {h: r for h, r in http.skipped_hosts.items() if h.startswith(("api.", "www.ebi.ac.uk"))}
    sites = len(http.skipped_hosts) - len(api_hosts)
    for host, reason in api_hosts.items():
        print(f"  Skipped {host} for part of this run: {reason}.")
    if sites:
        print(f"  {sites} publisher website(s) didn't respond and were skipped (only used as a last resort).")
    if counts.get("error"):
        print("Papers marked 'error' hit a network or rate-limit problem; "
              "`paperflow resolve --retry-errors` tries them again.")


def cmd_types(cfg: Config, args) -> None:
    """Determine article types for papers already in the database."""
    from collections import Counter
    from .http import Http
    from .resolver import Resolver

    store = _store(cfg)
    q = ("SELECT id, doi, title, venue, abstract, is_preprint FROM papers "
         "WHERE status IN ('resolved', 'no_abstract', 'unmatched') AND title IS NOT NULL "
         # types you set, or the LLM corrected after reading the abstract, are never redone
         "AND coalesce(article_type_source, '') NOT IN ('you', 'llm')")
    if not args.all:
        q += " AND article_type IS NULL"
    papers = [dict(r) for r in store.db.execute(q).fetchall()]
    for p in papers:
        r = store.db.execute("SELECT value FROM identifiers WHERE paper_id=? AND scheme='pmid'", (p["id"],)).fetchone()
        p["pmid"] = r[0] if r else None
    if not papers:
        print("Every paper already has an article type (use --all to redo them).")
        store.close()
        return

    http = Http(cfg.cache, cfg.contact_email, cfg.cache_days)
    resolver = Resolver(http, cfg)
    types, sources = Counter(), Counter()
    retry = 0
    print(f"Determining article types for {len(papers)} papers… (Ctrl+C is safe; rerun to continue)")
    with ThreadPoolExecutor(max_workers=args.workers or cfg.workers) as pool:
        futures = {pool.submit(resolver.lookup_type, p): p for p in papers}
        bar = tqdm(as_completed(futures), total=len(futures), unit="paper", dynamic_ncols=True)
        try:
            for i, fut in enumerate(bar, 1):
                p = futures[fut]
                meta, failures = fut.result()
                strong = (meta.article_type_source in ("europepmc", "openalex", "semanticscholar", "preprint server")
                          or meta.article_type in ("correction", "retraction"))
                if failures and not strong:
                    retry += 1  # a source we'd have consulted was unreachable: leave it for next time
                    continue
                store.set_article_type(p["id"], meta.article_type, meta.article_type_source, meta.related)
                types[meta.article_type] += 1
                sources[meta.article_type_source] += 1
                if i % 50 == 0:
                    store.commit()
        except KeyboardInterrupt:
            bar.close()
            print("\nStopping: saving progress…")
            for f in futures:
                f.cancel()
    store.commit()
    store.close()
    print("Article types: " + ", ".join(f"{t}: {n}" for t, n in types.most_common()))
    print("Decided by: " + ", ".join(f"{s}: {n}" for s, n in sources.most_common()))
    if retry:
        print(f"{retry} paper(s) skipped because a metadata source was unreachable; run `paperflow types` "
              "again later to fill them in.")


def cmd_set_type(cfg: Config, args) -> None:
    from .triage_cli import _find_paper, _short
    store = _store(cfg)
    pid = _find_paper(store, args.paper)
    if pid is None:
        raise SystemExit(f"No paper with id or DOI {args.paper!r} in the database.")
    title, old, src = store.db.execute("SELECT title, article_type, article_type_source FROM papers WHERE id=?",
                                       (pid,)).fetchone()
    store.set_article_type(pid, args.type, "you", None)
    store.commit()
    store.close()
    print(f"[{pid}] {_short(title)}\n  was: {old or 'not determined'} (from {src or '-'})  →  now: {args.type} (set by you)")
    print("Run `paperflow classify --redo` (or label the paper yourself) for the triage verdict to reflect it.")


def cmd_stats(cfg: Config, args) -> None:
    store = _store(cfg)
    s = store.stats()
    from collections import Counter
    from .triage import final_labels
    triage = Counter(v["label"] for v in final_labels(store).values())
    n_examples = dict(store.db.execute("SELECT label, count(*) FROM examples GROUP BY label").fetchall())
    store.close()
    print(f"Emails: {sum(s['emails_by_source'].values())}  {s['emails_by_source']}")
    print(f"  by parser: {s['emails_by_parser']}")
    print(f"  with no papers found: {s['emails_with_no_papers']}")
    if s["reported_but_not_shown"]:
        print(f"  results alerts said they found but didn't include: {s['reported_but_not_shown']}")
    print(f"Sightings: {s['sightings']}   Distinct papers: {s['papers']}")
    print(f"  seen in more than one alert email: {s['papers_seen_in_multiple_alerts']}")
    print(f"  by source combination: {s['papers_by_source_combo']}")
    print(f"Status: {s['papers_by_status']}")
    print(f"Abstract sources: {s['abstract_sources']}")
    print(f"Article types: {s['article_types']}")
    if sum(triage.values()) != triage.get("unscored", 0):
        print("Triage: " + ", ".join(f"{k}: {triage[k]}" for k in
                                       ("mandatory", "interest", "skip", "pending_llm", "filtered", "unscored") if triage[k]))
        print(f"  examples: {n_examples}")
    print("Most productive alerts (distinct papers):")
    for source, name, n in s["top_alerts"]:
        print(f"  {n:5d}  [{source}] {name}")


def cmd_export(cfg: Config, args) -> None:
    store = _store(cfg)
    added_since = args.added_since
    if args.new:
        added_since = store.get_meta("last_ingest_started")
        if not added_since:
            raise SystemExit("No ingest has added emails yet, so there is nothing 'new' to export.")
    rows = list(store.export_rows(args.status, args.since, added_since))
    from .triage import final_labels
    finals = final_labels(store)
    for r in rows:
        r["triage"] = finals.get(r["id"])
    if getattr(args, "label", None):
        rows = [r for r in rows if r["triage"] and r["triage"]["label"] in args.label]
    if getattr(args, "type", None):
        rows = [r for r in rows if (r.get("article_type") or "unknown") in args.type]
    from .reading import current_states
    store2 = _store(cfg)
    states = current_states(store2)
    has_text = {pid for (pid,) in store2.db.execute("SELECT paper_id FROM fulltexts WHERE status='ok'")}
    in_zotero = dict(store2.db.execute("SELECT paper_id, sent_at FROM zotero_sent").fetchall())
    store2.close()
    for r in rows:
        r["reading"] = states.get(r["id"], "unmarked")
        r["full_text"] = r["id"] in has_text
        r["added_to_zotero"] = in_zotero.get(r["id"])
    if getattr(args, "state", None):
        rows = [r for r in rows if r["reading"] in args.state]
    if getattr(args, "dois", False):
        dois = [r["doi"] for r in rows if r.get("doi")]
        text = "\n".join(dois) + ("\n" if dois else "")
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"Wrote {len(dois)} DOIs to {args.out}" +
                  (f" ({len(rows) - len(dois)} paper(s) without a DOI left out)" if len(rows) > len(dois) else ""))
        else:
            sys.stdout.write(text)
        return
    if not args.out:
        raise SystemExit("--out is required (except with --dois, which prints to the terminal)")
    order = {"mandatory": 0, "interest": 1, "pending_llm": 2, "skip": 3, "filtered": 4, "unscored": 5}
    rows.sort(key=lambda r: (order.get((r["triage"] or {}).get("label"), 9),
                             -(((r["triage"] or {}).get("embed") or {}).get("score") or 0)))
    store.close()
    out = Path(args.out)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "triage", "triage_by", "projects", "reason", "filter_score", "reading", "full_text",
                        "status", "article_type", "title",
                        "first_author", "year", "venue", "doi", "is_preprint", "has_abstract", "added_to_zotero", "n_sightings",
                        "sources", "alerts", "last_seen", "added_at"])
            for r in rows:
                first = (r["authors"] or [None])[0]
                venue = r["venue"]
                for sg in r["sightings"]:  # before resolution, fall back to what the emails printed
                    if not first and sg["raw_authors"]:
                        first = re.split(r"[;,]", sg["raw_authors"])[0].strip(" …")
                    if not venue and sg["raw_venue"]:
                        venue = sg["raw_venue"]
                t = r["triage"] or {}
                llm = (t.get("llm") or {}).get("detail") or {}
                w.writerow([
                    r["id"], t.get("label"), t.get("by"), "; ".join(llm.get("projects") or []), llm.get("reason"),
                    round((t.get("embed") or {}).get("score") or 0, 3) if t.get("embed") else None,
                    r["reading"], "yes" if r["full_text"] else "",
                    r["status"], r.get("article_type"), r["title"], first, r["year"], venue,
                    r["doi"], r["is_preprint"], bool(r["abstract"]), (r["added_to_zotero"] or "")[:10],
                    len(r["sightings"]),
                    "; ".join(sorted({s["source"] for s in r["sightings"]})),
                    "; ".join(sorted({s["alert_name"] or "" for s in r["sightings"]})), r["last_seen"],
                    r["added_at"],
                ])
    else:
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} papers to {out}")


def cmd_parse(cfg: Config, args) -> None:
    def messages():
        for path in args.files:
            if Path(path).suffix.lower() == ".csv":
                yield from ((f"{path} row", raw) for raw in iter_outlook_csv(path))
            else:
                yield str(path), Path(path).read_bytes()

    shown = 0
    for label, raw in messages():
        parsed = parse_email(raw, cfg.senders, accept_all=True)
        if args.match and args.match.lower() not in parsed.subject.lower():
            continue
        shown += 1
        if args.limit and shown > args.limit:
            break
        print(f"\n=== {label}")
        print(f"source={parsed.source} parser={parsed.parser} kind={parsed.alert_kind} "
              f"alert={parsed.alert_name!r}\nsubject={parsed.subject!r}  date={parsed.received_at}")
        for i, s in enumerate(parsed.sightings, 1):
            trunc = " [truncated]" if s.title_truncated else ""
            print(f"\n {i:2d}. {s.title}{trunc}")
            if s.authors_raw or s.venue_raw or s.year:
                print(f"     by: {s.authors_raw} | {s.venue_raw} | {s.year}")
            if s.alert:
                print(f"     section: {s.alert}")
            print(f"     ids: {s.ids or '-'}")
            print(f"     url: {s.url}")
            if s.snippet:
                print(f"     snippet: {s.snippet[:160]}")
        if not parsed.sightings:
            print("  (no papers found)")
    if args.match and not shown:
        print(f"No emails with {args.match!r} in the subject.")


def cmd_reparse(cfg: Config, args) -> None:
    store = _store(cfg)
    n = 0
    for mid, raw in list(store.iter_raw_emails()):
        parsed = parse_email(raw, cfg.senders, accept_all=True)
        store.clear_sightings(mid)
        store.add_email(parsed, raw)
        n += 1
    orphans = store.delete_orphans()
    merged = store.dedupe_titles()
    store.close()
    print(f"Re-parsed {n} emails; removed {orphans} papers no longer referenced; {merged} title merges.")
    print("Run `paperflow resolve` to fetch metadata for any new papers.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="paperflow", description="Paper-alert ingestion pipeline")
    ap.add_argument("-c", "--config", default="config.yaml", help="config file (default: config.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="parse alert emails into the database")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--imap", action="store_true", help="fetch from the IMAP folders in the config")
    src.add_argument("--mbox", type=Path, help="an mbox file, e.g. from Google Takeout")
    src.add_argument("--eml-dir", type=Path, help="a folder of .eml files")
    src.add_argument("--outlook-csv", type=Path, help="Outlook 'Export to CSV' file")
    p.add_argument("--all-senders", action="store_true",
                   help="treat every email as an alert (generic parser for unknown senders)")
    p.add_argument("--limit", type=int, help="stop after N new emails")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("resolve", help="fetch metadata and abstracts")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--retry-unmatched", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("run", help="ingest from IMAP, then resolve")
    p.set_defaults(func=lambda cfg, a: (
        cmd_ingest(cfg, argparse.Namespace(imap=True, mbox=None, eml_dir=None, outlook_csv=None, all_senders=False, limit=None)),
        cmd_resolve(cfg, argparse.Namespace(retry_errors=True, retry_unmatched=False, limit=None, workers=None)),
    ))

    p = sub.add_parser("types", help="determine article types (review, commentary…) for papers already resolved")
    p.add_argument("--all", action="store_true",
                   help="redo papers that already have a type (except types set by you or corrected by the LLM)")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_types)

    p = sub.add_parser("set-type", help="correct a paper's article type (never overridden automatically)")
    p.add_argument("paper", help="paper id (from exports) or DOI")
    p.add_argument("type", choices=["research", "review", "commentary", "editorial", "correction", "retraction", "other"])
    p.set_defaults(func=cmd_set_type)

    p = sub.add_parser("stats", help="summarize the database")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="write papers to .jsonl or .csv")
    p.add_argument("--out", help="output file (.jsonl or .csv); optional with --dois")
    p.add_argument("--dois", action="store_true",
                   help="write only DOIs, one per line (for Zotero's Add Item by Identifier); prints if no --out")
    p.add_argument("--state", action="append", choices=["new", "familiar", "read", "saved", "unmarked"],
                   help="only papers in this reading state of yours (repeatable)")
    p.add_argument("--label", action="append",
                   choices=["mandatory", "interest", "skip", "filtered", "pending_llm", "unscored"],
                   help="only papers with this triage label (repeatable)")
    p.add_argument("--type", action="append",
                   choices=["research", "review", "commentary", "editorial", "correction", "retraction", "other", "unknown"],
                   help="only papers of this article type (repeatable)")
    p.add_argument("--status", action="append",
                   help="only these statuses (repeatable): resolved, no_abstract, unmatched, error, pending")
    p.add_argument("--new", action="store_true",
                   help="only papers first added by the most recent ingest that added emails")
    p.add_argument("--added-since", help="only papers first added to the database on/after this date (YYYY-MM-DD)")
    p.add_argument("--since", help="only papers last seen in an email dated on/after this date "
                   "(YYYY-MM-DD; needs email dates, so not useful with Outlook CSV exports)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("parse", help="dry-run the parsers on .eml files or an Outlook CSV (no database)")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--match", help="only emails whose subject contains this text")
    p.add_argument("--limit", type=int, default=None, help="show at most N emails")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("reparse", help="re-run parsers over all stored emails")
    p.set_defaults(func=cmd_reparse)

    from .triage_cli import register as register_triage
    register_triage(sub)
    from .library_cli import register as register_library
    register_library(sub)
    from .brief_cli import register_cli as register_brief
    register_brief(sub)
    from .weekly_cli import register as register_weekly
    register_weekly(sub)

    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # titles contain characters a Windows code page can't print
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cfg = load_config(args.config, required=args.cmd not in ("parse",))
    args.func(cfg, args)


if __name__ == "__main__":
    main()
