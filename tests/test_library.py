"""Reading states, the review pass, briefings, DOI export and full-text retrieval."""
import io, json, shutil, sys, tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.config import Config
from paperflow.store import Store
from paperflow import reading, fulltext as ft

def make_db(tmp):
    store = Store(tmp / "data" / "db.sqlite")
    titles = ["Structure of the Pik resistosome", "Sensor and helper NLRs communicate", "A rice blast effector",
              "Engineering integrated domains", "Protein design for plant immunity"]
    for i, t in enumerate(titles, 1):
        store.db.execute("INSERT INTO papers (id, title, title_key, doi, year, abstract, status) VALUES (?,?,?,?,?,?,'resolved')",
                         (i, t, t.lower(), f"10.5555/p{i}", 2026, f"Abstract of {t}. " * 20))
        store.db.execute("INSERT INTO identifiers VALUES ('doi', ?, ?)", (f"10.5555/p{i}", i))
        store.db.execute("INSERT INTO sightings (paper_id, message_id, source) VALUES (?, ?, 'stork')", (i, f"m{i}"))
        store.db.execute("INSERT INTO assessments VALUES (?, 'embed', 'pass', 0.9, '{}', 'x')", (i,))
        label = "mandatory" if i <= 4 else "interest"
        store.db.execute("INSERT INTO assessments VALUES (?, 'llm', ?, NULL, ?, 'x')",
                         (i, label, json.dumps({"projects": ["NLR engineering"], "reason": "Core."})))
    store.commit()
    return store

def test_review(tmp):
    store = make_db(tmp)
    pids = reading.select_papers(store, ["mandatory"], None, False, include_marked=False)
    assert len(pids) == 4, pids
    # paper 1: new; 2: familiar; 3: Enter (leave); then back to 3 and relabel interest + read;
    # 4: type -> review, then saved
    keys = iter(["n", "f", "\r", "b", "i", "r", "t", "2", "s"])
    screens = []
    changed = reading.review_loop(store, pids, read_key=lambda: next(keys), out=screens.append)
    assert changed == {"state": 4, "label": 1, "type": 1}, changed
    st = reading.current_states(store)
    order = {pid: i for i, pid in enumerate(pids)}
    by_pos = {order[p]: s for p, s in st.items()}
    assert by_pos == {0: "new", 1: "familiar", 2: "read", 3: "saved"}, by_pos
    p4 = pids[3]
    assert store.db.execute("SELECT article_type, article_type_source FROM papers WHERE id=?", (p4,)).fetchone()[:] == ("review", "you")
    from paperflow.triage import final_labels
    assert final_labels(store)[pids[2]]["label"] == "interest" and final_labels(store)[pids[2]]["by"] == "you"
    assert "Why: Core." in screens[0] and "[1/4]" in screens[0]
    # resuming: everything is marked now (paper 3 is now 'interest', so it no longer matches 'mandatory')
    assert reading.select_papers(store, ["mandatory"], None, False, include_marked=False) == []
    # quitting midway keeps what was done
    keys = iter(["q"])
    reading.review_loop(store, pids, read_key=lambda: next(keys), out=lambda *_: None)
    store.close()
    print("ok: review pass (states, relabel, type, back, resume, quit)")

def test_briefing_and_mark(tmp):
    store = Store(tmp / "data" / "db.sqlite")
    store.db.execute("INSERT INTO briefings (id, name, kind, created_at) VALUES (1, 'week 44', 'weekly', 'x')")
    for pos, pid in enumerate([5, 2, 4], 1):
        store.db.execute("INSERT INTO briefing_items VALUES (1, ?, ?)", (pos, pid))
    store.commit()
    name, pids = reading.briefing_papers(store, "latest")
    assert (name, pids) == ("week 44", [5, 2, 4])
    keys = iter(["\r", "s", "q"])
    screens = []
    reading.review_loop(store, pids, numbers={5: 1, 2: 2, 4: 3}, read_key=lambda: next(keys), out=screens.append)
    assert "Paper 1" in screens[0] and "Paper 2" in screens[1]
    assert reading.current_states(store)[2] == "saved"
    store.close()
    from paperflow.library_cli import cmd_mark
    cfg = Config(database=tmp / "data" / "db.sqlite", cache=tmp / "c.sqlite", fulltext_dir=tmp / "fulltext")
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_mark(cfg, Namespace(paper="10.5555/p2", state="read", note=None))
    assert "familiar" in buf.getvalue() and "saved" in buf.getvalue() and "read" in buf.getvalue()
    print("ok: briefing order and numbering; mark by DOI with history:", buf.getvalue().splitlines()[1].strip())

def test_export_dois(tmp):
    from paperflow.cli import cmd_export
    cfg = Config(database=tmp / "data" / "db.sqlite", cache=tmp / "c.sqlite")
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_export(cfg, Namespace(out=None, dois=True, state=["saved"], label=None, type=None,
                                  status=None, since=None, new=False, added_since=None))
    saved = buf.getvalue().split()
    assert saved == ["10.5555/p4"], saved   # p2 was saved, then marked read
    cmd_export(cfg, Namespace(out=str(tmp / "r.csv"), dois=False, state=None, label=None, type=None,
                              status=None, since=None, new=False, added_since=None))
    import csv
    rows = {r["id"]: r for r in csv.DictReader(open(tmp / "r.csv"))}
    assert rows["2"]["reading"] == "read" and rows["5"]["reading"] == "unmarked"
    print("ok: export --state saved --dois ->", saved)

# -- full text --------------------------------------------------------------------------------

JATS = b"""<?xml version="1.0"?><!DOCTYPE article><article><front><article-meta>
<title-group><article-title>Structure of the Pik resistosome</article-title></title-group>
<abstract><p>We solved the structure.</p></abstract></article-meta></front>
<body><sec><title>Introduction</title><p>""" + b"NLRs are immune receptors. " * 150 + b"""<xref>1</xref></p></sec>
<sec><title>Results</title><sec><title>Cryo-EM</title><p>""" + b"The map reached 3 angstrom. " * 100 + b"""</p>
<fig><label>Figure 1</label><caption><p>Overall architecture.</p></caption></fig></sec></sec></body>
<back><ref-list><ref>Smith 2020</ref></ref-list></back></article>"""

def make_pdf(path, title, body_lines=400, refs=True, doi=None):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path)); y = 800
    def line(t):
        nonlocal y
        if y < 60:
            c.showPage(); y = 800
        c.drawString(50, y, t); y -= 14
    line(title)
    if doi: line(f"https://doi.org/{doi}")
    for i in range(body_lines):
        line(f"Body sentence {i} about NLR immune receptors and effectors in plants.")
    if refs:
        line(""); line("References")
        for i in range(30): line(f"{i}. Somebody et al. A cited paper. Journal 2020.")
    c.save()
    return path.read_bytes()

def test_converters(tmp):
    text = ft.jats_to_text(JATS)
    assert text.startswith("# Structure of the Pik resistosome") and "## Introduction" in text
    assert "### Cryo-EM" in text and "Figure 1: Overall architecture." in text and "Smith 2020" not in text
    pdf = make_pdf(tmp / "a.pdf", "Structure of the Pik resistosome")
    t = ft.pdf_to_text(pdf)
    assert "Body sentence 399" in t and "A cited paper" not in t
    print("ok: JATS sections/figure legends; PDF text with reference list dropped")

class FakeHttp:
    def __init__(self, tmp): self.tmp, self.calls = tmp, []
    def get_json(self, url, params=None, headers=None):
        from paperflow.http import HttpError
        self.calls.append(url)
        q = (params or {}).get("query", "")
        if url.endswith("/search"):
            if "p1" in q: return {"resultList": {"result": [{"pmcid": "PMC111", "source": "MED", "id": "1", "inEPMC": "Y"}]}}
            if "p6" in q: raise HttpError("www.ebi.ac.uk: HTTP 503")
            return {"resultList": {"result": []}}
        if "api.biorxiv.org" in url:
            if "2026.01.05.697760" in url:
                return {"collection": [{"version": "2", "jatsxml": "https://www.biorxiv.org/x.source.xml"}]}
            return {"collection": []}
        if "unpaywall" in url:
            doi = url.split("/v2/")[1]
            locs = {"10.5555/p3": ["https://repo.org/login", "https://repo.org/p3.pdf"],
                    "10.5555/p4": ["https://repo.org/wrong.pdf"]}.get(doi, [])
            return {"best_oa_location": None, "oa_locations": [{"url_for_pdf": u} for u in locs]}
        return None
    def get_bytes(self, url, **kw):
        self.calls.append(url)
        if url.endswith("PMC111/fullTextXML"): return JATS, "application/xml", url
        if url.endswith("x.source.xml"): return JATS.replace(b"Pik resistosome", b"bioRxiv preprint"), "text/xml", url
        if url.endswith("/login"): return b"<html>Please sign in</html>", "text/html", url
        if url.endswith("p3.pdf"): return make_pdf(self.tmp / "p3.pdf", "A rice blast effector"), "application/pdf", url
        if url.endswith("wrong.pdf"): return make_pdf(self.tmp / "w.pdf", "An unrelated paper on wheat"), "application/pdf", url
        return None

def test_fetch(tmp):
    f = ft.Fetcher(FakeHttp(tmp), "me@x.org")
    r1 = f.fetch({"doi": "10.5555/p1", "title": "Structure of the Pik resistosome"})
    assert r1.status == "ok" and r1.source == "europepmc"
    rb = f.fetch({"doi": "10.64898/2026.01.05.697760", "title": "bioRxiv preprint"})
    assert rb.status == "ok" and rb.source == "biorxiv", rb
    r3 = f.fetch({"doi": "10.5555/p3", "title": "A rice blast effector"})
    assert r3.status == "ok" and r3.source == "open-access pdf" and r3.url.endswith("p3.pdf"), r3
    r4 = f.fetch({"doi": "10.5555/p4", "title": "Engineering integrated domains"})
    assert r4.status == "unavailable" and "doesn't contain the paper's title" in r4.detail, r4
    r5 = f.fetch({"doi": "10.5555/p5", "title": "Protein design"})
    assert r5.status == "unavailable" and "no open-access copy" in r5.detail
    r6 = f.fetch({"doi": "10.5555/p6", "title": "x"})
    assert r6.status == "error"
    print("ok: Europe PMC XML, bioRxiv XML, OA PDF; login page and wrong PDF rejected; errors retried")

def test_fulltext_command(tmp):
    import paperflow.library_cli as lib, paperflow.http as phttp
    drop = tmp / "fulltext"; drop.mkdir()
    make_pdf(drop / "whatever_name.pdf", "Engineering integrated domains", doi="10.5555/p4")
    make_pdf(drop / "stray.pdf", "Some paper not in the database", doi="10.5555/zzz")
    cfg = Config(database=tmp / "data" / "db.sqlite", cache=tmp / "c.sqlite", fulltext_dir=drop)
    real = phttp.Http
    phttp.Http = lambda *a, **k: FakeHttp(tmp)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            lib.cmd_fulltext(cfg, Namespace(label=None, missing=False, retry=False, workers=2))
        out = buf.getvalue()
    finally:
        phttp.Http = real
    print("   " + "\n   ".join(l for l in out.splitlines() if l.strip()))
    store = Store(tmp / "data" / "db.sqlite")
    rows = {pid: (st, src) for pid, st, src in store.db.execute("SELECT paper_id, status, source FROM fulltexts")}
    assert rows[4] == ("ok", "your pdf") and rows[1] == ("ok", "europepmc") and rows[2][0] == "unavailable", rows
    assert "stray.pdf" in out and "not used" in out and (store.fulltext_dir / "1.txt").exists()
    assert ft.load(store, 1).startswith("# Structure of the Pik resistosome")
    # A merge keeps the full text and renames its file
    store.db.execute("INSERT INTO papers (id, title, status) VALUES (9, 'dup', 'resolved')")
    store.merge(9, 1); store.commit()
    assert store.db.execute("SELECT status FROM fulltexts WHERE paper_id=9").fetchone()[0] == "ok"
    assert (store.fulltext_dir / "9.txt").exists() and not (store.fulltext_dir / "1.txt").exists()
    store.close()
    buf = io.StringIO()
    with redirect_stdout(buf):
        lib.cmd_fulltext(cfg, Namespace(label=None, missing=True, retry=False, workers=1))
    assert "[2]" in buf.getvalue() and "10.5555/p2" in buf.getvalue()
    print("ok: drop folder (any file name, matched by DOI), unmatched files reported, --missing, merge")

if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    try:
        test_review(tmp); test_briefing_and_mark(tmp); test_export_dois(tmp)
        test_converters(tmp); test_fetch(tmp); test_fulltext_command(tmp)
    finally:
        shutil.rmtree(tmp)
    print("LIBRARY CHECKS PASSED")
