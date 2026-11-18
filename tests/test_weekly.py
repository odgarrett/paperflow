"""The weekly routine: what counts as new, interest overviews, Zotero tracking, the weekly command."""
import io, json, re, shutil, subprocess, sys, tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow import classify, tts as ptts, reading
from paperflow.briefing import WINDOW_KEY, new_papers
from paperflow.config import Config
from paperflow.store import Store
import test_brief as tb

PROFILE = "# Research profile\n### Project: Protein design\n### Project: NLR engineering\n"

def add(store, pid, title, label, project, added, abstract="An abstract."):
    store.db.execute("INSERT INTO papers (id, title, title_key, authors, venue, year, doi, abstract, status, added_at) "
                     "VALUES (?,?,?,?,?,?,?,?,'resolved',?)",
                     (pid, title, title.lower(), json.dumps(["Ana Garcia", "Bo Chen"]), "Plant Cell", 2026,
                      f"10.5555/w{pid}", abstract, added))
    store.db.execute("INSERT INTO identifiers VALUES ('doi', ?, ?)", (f"10.5555/w{pid}", pid))
    if label in ("mandatory", "interest", "skip"):
        store.db.execute("INSERT INTO assessments VALUES (?, 'embed', 'pass', 0.9, '{}', 'x')", (pid,))
        store.db.execute("INSERT INTO assessments VALUES (?, 'llm', ?, NULL, ?, 'x')",
                         (pid, label, json.dumps({"projects": [project] if project else [], "reason": "r"})))
    elif label == "pending_llm":
        store.db.execute("INSERT INTO assessments VALUES (?, 'embed', 'pass', 0.9, '{}', 'x')", (pid,))

class Model(tb.FakeBackend):
    def complete(self, system, user, schema=None, max_tokens=400, temperature=0.0):
        if user.startswith("Topic:"):
            self.calls.append({"schema": False, "user": user, "system": system[0]["text"], "max_tokens": max_tokens})
            nums = re.findall(r"^Paper (\d+):", user, re.M)
            # mention all but the last paper, and two of them as a list, to exercise the coverage check
            body = f"Papers {nums[0]} and {nums[1]} both test binders." if len(nums) > 1 else f"Paper {nums[0]} tests X."
            body += " " + " ".join(f"Paper {n} finds something." for n in nums[2:-1])
            return body, {}
        return super().complete(system, user, schema, max_tokens, temperature)

def run(cfg, backend, **kw):
    a = dict(new=True, no_overview=False, force=False)
    a.update(kw)
    return tb.run(cfg, backend, {}, **a)

def main():
    tmp = Path(tempfile.mkdtemp())
    store = Store(tmp / "data" / "db.sqlite")
    # Backlog: already briefed in a pilot / backlog briefing
    add(store, 1, "Backlog mandatory paper", "mandatory", "NLR engineering", "2026-10-01T00:00:00")
    add(store, 2, "Backlog interest paper", "interest", "NLR engineering", "2026-10-01T00:00:00")
    store.db.execute("INSERT INTO briefings (id, name, kind, created_at) VALUES (1, 'pilot', 'mandatory', 'x')")
    store.db.execute("INSERT INTO briefing_items VALUES (1, 1, 1)")
    # This week's ingest
    store.set_meta("last_ingest_started", "2026-11-03T00:00:00")
    add(store, 10, "New Pik structure", "mandatory", "NLR engineering", "2026-11-03T01:00:00")
    add(store, 11, "New binder design paper", "mandatory", "Protein design", "2026-11-03T01:00:00")
    for i, t in enumerate(["Binders A", "Binders B", "Binders C"]):
        add(store, 20 + i, f"{t} for effectors", "interest", "Protein design", "2026-11-03T01:00:00")
    add(store, 30, "Helper NLR evolution", "interest", "NLR engineering", "2026-11-03T01:00:00")
    add(store, 31, "Something unrelated", "skip", "", "2026-11-03T01:00:00")
    add(store, 32, "Not classified yet", "pending_llm", "", "2026-11-03T01:00:00")
    store.commit()

    sel = new_papers(store)
    assert sorted(sel["mandatory"]) == [10, 11] and sorted(sel["interest"]) == [20, 21, 22, 30]
    assert sel["untriaged"] == [32] and 2 not in sel["interest"]           # backlog interest isn't "new"
    store.close()
    (tmp / "profile.md").write_text(PROFILE)
    cfg = Config(database=tmp / "data" / "db.sqlite", cache=tmp / "c.sqlite", profile=tmp / "profile.md",
                 briefings_dir=tmp / "b", pronunciations=tmp / "p.txt", ollama_num_ctx=12288)

    # Untriaged papers block the weekly briefing.
    try:
        run(cfg, Model(), no_audio=True)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "haven't been triaged" in str(e)
    store = Store(cfg.database)
    store.db.execute("INSERT OR REPLACE INTO assessments VALUES (32, 'llm', 'skip', NULL, '{}', 'x')")
    store.commit(); store.close()

    out = run(cfg, Model(), name="weekly-test")
    print(out.strip().splitlines()[0])
    assert "Briefing 'weekly-test': 2 mandatory, 4 of interest" in out
    bdir = tmp / "b" / "weekly-test"
    md = (bdir / "briefing.md").read_text()
    # mandatory 1-2 (profile order: Protein design first), then interest 3-6
    assert md.index("### 1. New binder design paper") < md.index("### 2. New Pik structure")
    assert "## Also of interest: Protein design" in md and "## Also of interest: NLR engineering" in md
    assert "Papers 3 and 4 both test binders." in md and "Paper 5: Binders C for effectors." in md  # coverage fix
    assert "**6. Helper NLR evolution**" in md
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(bdir / "weekly-test.mp3")],
                           capture_output=True, text=True).stdout
    chapters = [c["tags"]["title"] for c in json.loads(probe)["chapters"]]
    print("   chapters:", chapters)
    assert chapters == ["Introduction", "Paper 1: New binder design paper", "Paper 2: New Pik structure",
                        "Also of interest: Protein design", "Also of interest: NLR engineering"]
    script = (bdir / "script.txt").read_text()
    assert "2 mandatory papers, then 4 more of interest." in script
    store = Store(cfg.database)
    name, pids = reading.briefing_papers(store, "latest")
    assert pids == [11, 10, 20, 21, 22, 30], pids                             # review --briefing covers both parts
    marker = store.get_meta(WINDOW_KEY)
    assert marker > "2026-11-03T01:00:00"
    sel = new_papers(store)
    assert not sel["mandatory"] and not sel["interest"]
    store.close()
    out = run(cfg, Model(), no_audio=True)
    assert "Nothing new to brief" in out
    print("ok: weekly selection, untriaged guard, numbering, overviews with coverage fix, chapters, window")

    # Next week: one paper fails; the window must not move past it.
    store = Store(cfg.database)
    add(store, 40, "Week two mandatory", "mandatory", "NLR engineering", "2099-01-01T00:00:00", abstract="BREAK me")
    add(store, 41, "Week two interest", "interest", "NLR engineering", "2099-01-01T00:00:00")
    store.commit(); store.close()
    out = run(cfg, Model(), no_audio=True, name="week2")
    assert "1 paper(s) failed" in out and "0 mandatory, 1 of interest" in out, out
    store = Store(cfg.database)
    assert new_papers(store)["mandatory"] == [40]                             # still due next time
    store.db.execute("UPDATE papers SET abstract='fine now' WHERE id=40"); store.commit(); store.close()
    out = run(cfg, Model(), no_audio=True, name="week2b")
    assert "1 mandatory, 0 of interest" in out, out
    print("ok: a paper that failed is picked up by the next weekly briefing")

    # Zotero: only papers not yet sent; recorded after confirmation.
    from paperflow.library_cli import cmd_zotero
    store = Store(cfg.database)
    reading.set_state(store, 10, "saved"); reading.set_state(store, 21, "saved")
    store.db.execute("UPDATE papers SET doi=NULL WHERE id=21"); store.commit(); store.close()
    import paperflow.library_cli as lib
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_zotero(cfg, Namespace(all=False, mark_sent=True, dry_run=False))
    out = buf.getvalue()
    assert "10.5555/w10" in out and "[21] Binders B for effectors" in out and "Recorded 2" in out, out
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_zotero(cfg, Namespace(all=False, mark_sent=False, dry_run=False))
    assert "Nothing to add" in buf.getvalue()
    store = Store(cfg.database); reading.set_state(store, 11, "saved"); store.commit(); store.close()
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_zotero(cfg, Namespace(all=False, mark_sent=False, dry_run=True))
    assert buf.getvalue().count("10.5555/") == 1 and "10.5555/w11" in buf.getvalue()
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_zotero(cfg, Namespace(all=True, mark_sent=False, dry_run=True))
    assert "10.5555/w10" in buf.getvalue() and "10.5555/w11" in buf.getvalue()
    print("ok: zotero lists only papers not yet added, records them, --all lists everything")

    # The weekly command: steps in order; stops cleanly at a failing step.
    import paperflow.weekly_cli as wk, paperflow.cli as cli, paperflow.triage_cli as tcli, paperflow.brief_cli as bcli
    calls = []
    fakes = {"cmd_ingest": (cli, lambda c, a: calls.append(("ingest", str(a.outlook_csv))) or 0),
             "cmd_resolve": (cli, lambda c, a: calls.append(("resolve", a.retry_errors))),
             "cmd_types": (cli, lambda c, a: calls.append(("types",))),
             "cmd_filter": (tcli, lambda c, a: calls.append(("filter",))),
             "cmd_classify": (tcli, lambda c, a: calls.append(("classify",))),
             "cmd_fulltext": (lib, lambda c, a: calls.append(("fulltext",))),
             "cmd_brief": (bcli, lambda c, a: calls.append(("brief", a.new, a.no_audio)))}
    saved = {k: getattr(m, k) for k, (m, _) in fakes.items()}
    try:
        for k, (m, f) in fakes.items():
            setattr(m, k, f)
        csv = tmp / "paper_alerts.CSV"; csv.write_text("x")
        cfg.outlook_csv = csv
        buf = io.StringIO()
        with redirect_stdout(buf):
            wk.cmd_weekly(cfg, Namespace(csv=None, no_audio=True, no_overview=False, skip_fulltext=False,
                                         skip_ingest=False, name=None, voice=None))
        assert [c[0] for c in calls] == ["ingest", "resolve", "types", "filter", "classify", "fulltext", "brief"]
        assert calls[0][1] == str(csv) and calls[1][1] is True and calls[-1] == ("brief", True, True)
        assert "6/7 Full texts" in buf.getvalue() and "No new emails" in buf.getvalue()
        calls.clear()
        tcli.cmd_filter = lambda c, a: (_ for _ in ()).throw(SystemExit("Need at least 2 positive examples"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                wk.cmd_weekly(cfg, Namespace(csv=None, no_audio=True, no_overview=False, skip_fulltext=True,
                                             skip_ingest=True, name=None, voice=None))
            except SystemExit:
                pass
        assert [c[0] for c in calls] == ["resolve", "types"]
        assert "Stopped at step 3 (filtering): Need at least 2 positive examples" in buf.getvalue()
        cfg.outlook_csv = tmp / "missing.CSV"
        try:
            wk.cmd_weekly(cfg, Namespace(csv=None, no_audio=True, no_overview=False, skip_fulltext=True,
                                         skip_ingest=False, name=None, voice=None))
            raise AssertionError
        except SystemExit as e:
            assert "Can't find the Outlook export" in str(e)
    finally:
        for k, (m, _) in fakes.items():
            setattr(m, k, saved[k])
    print("ok: weekly runs the steps in order and stops cleanly when one can't continue")
    shutil.rmtree(tmp)
    print("WEEKLY CHECKS PASSED")

if __name__ == "__main__":
    main()
