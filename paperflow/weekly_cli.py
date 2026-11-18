"""`paperflow weekly`: everything between saving the Outlook export and reading the briefing.

  1. ingest the Outlook CSV (emails seen before are skipped)
  2. resolve metadata and abstracts for the new papers (and retry earlier errors)
  3. article types for papers that don't have one yet
  4. triage: embedding filter, then the LLM's verdict on papers not judged yet
  5. open-access full texts for mandatory papers (plus PDFs in your drop folder)
  6. the weekly briefing: new mandatory papers in full, papers of interest in short overviews

Every step saves its progress, so if one stops (no network, Ollama crashed, Ctrl+C), fix the cause
and run `paperflow weekly` again: finished work is skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config


def _step(i: int, n: int, title: str) -> None:
    print(f"\n{'=' * 8} {i}/{n} {title} {'=' * max(4, 60 - len(title))}", flush=True)


def cmd_weekly(cfg: Config, args) -> None:
    from .brief_cli import cmd_brief
    from .cli import cmd_ingest, cmd_resolve, cmd_types
    from .library_cli import cmd_fulltext
    from .triage_cli import cmd_classify, cmd_filter

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else cfg.outlook_csv
    if not args.skip_ingest and not csv_path.exists():
        raise SystemExit(f"Can't find the Outlook export at {csv_path}. Save it there, pass its path "
                         "(`paperflow weekly path/to/export.CSV`), or set `outlook_csv` in config.yaml.")
    steps = []
    if not args.skip_ingest:
        steps.append(("Reading the Outlook export", lambda: cmd_ingest(cfg, argparse.Namespace(
            imap=False, mbox=None, eml_dir=None, outlook_csv=csv_path, all_senders=False, limit=None))))
    steps += [
        ("Metadata and abstracts", lambda: cmd_resolve(cfg, argparse.Namespace(
            retry_errors=True, retry_unmatched=False, limit=None, workers=None))),
        ("Article types", lambda: cmd_types(cfg, argparse.Namespace(all=False, workers=None))),
        ("Filtering", lambda: cmd_filter(cfg, argparse.Namespace(embedding=None, keep_recall=None, threshold=None))),
        ("Classifying", lambda: cmd_classify(cfg, argparse.Namespace(redo=False, limit=None, model=None,
                                                                      dry_run=False))),
    ]
    if not args.skip_fulltext:
        steps.append(("Full texts", lambda: cmd_fulltext(cfg, argparse.Namespace(
            label=None, missing=False, retry=False, workers=None))))
    steps.append(("Weekly briefing", lambda: cmd_brief(cfg, argparse.Namespace(
        new=True, no_overview=args.no_overview, force=False, papers=None, limit=None, name=args.name,
        dry_run=False, no_audio=args.no_audio, voice=args.voice, rewrite=False, redo=False))))

    for i, (title, run) in enumerate(steps, 1):
        _step(i, len(steps), title)
        try:
            result = run()
        except SystemExit as e:
            msg = str(e) if e.code not in (None, 0) else ""
            print(f"\nStopped at step {i} ({title.lower()}){': ' + msg if msg else '.'}")
            print("Everything finished so far is saved; fix the cause and run `paperflow weekly` again.")
            raise SystemExit(1)
        if title == "Reading the Outlook export" and result == 0:
            print("(No new emails in the export; carrying on in case earlier papers still need work.)")

    print("\nNext:\n"
          "  1. Read or listen to the briefing (the folder is listed above).\n"
          "  2. paperflow review --briefing latest   (s = save, f = got the gist, i/x = relabel)\n"
          "  3. paperflow zotero                     (paste the DOIs into Zotero, then confirm)")


def register(sub) -> None:
    p = sub.add_parser("weekly", help="the weekly routine: ingest the Outlook export through to the briefing")
    p.add_argument("csv", nargs="?", help="the Outlook CSV export (default: outlook_csv in config.yaml)")
    p.add_argument("--no-audio", action="store_true", help="make the ebook and script but no audio")
    p.add_argument("--no-overview", action="store_true", help="leave the papers of interest out of the briefing")
    p.add_argument("--skip-fulltext", action="store_true", help="don't look for full texts this time")
    p.add_argument("--skip-ingest", action="store_true", help="start from step 2 (e.g. you ingested already)")
    p.add_argument("--name", help="name of the briefing (default: weekly-<date>)")
    p.add_argument("--voice", help="Kokoro voice for this briefing")
    p.set_defaults(func=cmd_weekly)
