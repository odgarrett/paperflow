"""Article types: the rules, the backfill command against mocked APIs, and where types show up."""
import json, shutil, sys, tempfile
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.articletypes import determine
from paperflow.config import Config
from paperflow.store import Store

def test_abstract_cues():
    cases = [
        (["europepmc:Journal Article"], "NLR biology", "In this review, we discuss recent structural insights.", "review", "abstract"),
        (["europepmc:Journal Article"], "Structure of Pik", "Here we summarize our findings, which reveal a new interface.", "research", "europepmc"),
        (["europepmc:Journal Article"], "Outcomes in rice farmers", "We reviewed the medical records of 200 patients.", "research", "europepmc"),
        (["biorxiv:new results"], "Protein design in plants", "Here, we review recent advances in computational design.", "review", "abstract"),
        (["europepmc:Comment"], "A new receptor", "In this review we place the study in context.", "commentary", "europepmc"),
        (["openalex:article"], "Rethinking NLR evolution", "In this perspective, we argue that sensor NLRs evolved twice.", "commentary", "abstract"),
        (["openalex:article"], "Pik engineering", "This review summarizes current knowledge of integrated domains.", "review", "abstract"),
        (["crossref:journal-article"], "Plant immunity", "We review the latest progress on resistosomes.", "review", "abstract"),
    ]
    for pub, title, abstract, exp_t, exp_s in cases:
        got = determine(pub, title, None, abstract)
        assert got == (exp_t, exp_s), (title, abstract, got)
    print(f"ok: {len(cases)} abstract cues")


def test_llm_correction():
    """The model's reading of the type replaces weak metadata, never strong metadata or yours."""
    from paperflow import classify as clf
    tmp = Path(tempfile.mkdtemp())
    store = Store(tmp / "db.sqlite")
    papers = [  # (title, abstract, type, source, what the model says, expected type afterwards)
        ("Review hidden as research", "NLRs are... We discuss.", "research", "europepmc", "review", ("review", "llm")),
        ("Real comment", "Short text.", "commentary", "europepmc", "research", ("commentary", "europepmc")),
        ("No abstract here", None, "research", "openalex", "review", ("research", "openalex")),
        ("You said research", "Text.", "research", "you", "review", ("research", "you")),
        ("Journal guess", "Text.", "review", "journal", "research", ("research", "llm")),
        ("Agrees", "Text.", "research", "europepmc", "research", ("research", "europepmc")),
    ]
    for t, a, typ, src, _, _ in papers:
        cur = store.db.execute("INSERT INTO papers (title, title_key, abstract, article_type, article_type_source, status) "
                               "VALUES (?,?,?,?,?,'resolved')", (t, t.lower(), a, typ, src))
        store.db.execute("INSERT INTO assessments VALUES (?, 'embed', 'pass', 0.9, '{}', 'x')", (cur.lastrowid,))
    store.commit()
    said = {t: m for t, _, _, _, m, _ in papers}
    sent = []

    class Fake:
        def __init__(self): self.messages = self
        def create(self, **kw):
            sent.append(kw)
            msg = kw["messages"][0]["content"]
            title = msg.split("\n")[0].removeprefix("Title: ")
            v = {"label": "interest", "projects": [], "reason": "r", "article_type": said[title]}
            return type("R", (), {"content": [type("B", (), {"text": json.dumps(v)})()], "usage": None})()

    c = clf.Classifier(clf.AnthropicBackend("m", client=Fake()))
    profile = tmp / "p.md"; profile.write_text("# profile")
    stats = clf.run_classification(store, c, profile, clf.candidates(store), workers=1, progress=False)
    got = {t: (a, s) for t, a, s in store.db.execute("SELECT title, article_type, article_type_source FROM papers")}
    for t, _, _, _, _, exp in papers:
        assert got[t] == exp, (t, got[t], exp)
    assert len(stats.types_corrected) == 2
    assert "article_type" in sent[0]["output_config"]["format"]["schema"]["required"]
    msgs = {m["messages"][0]["content"].split("\n")[0]: m["messages"][0]["content"] for m in sent}
    assert "Article type (from metadata; may be wrong): review article" not in msgs["Title: You said research"]
    assert "Article type: primary research article" in msgs["Title: You said research"]
    assert "(from metadata; may be wrong)" in msgs["Title: Real comment"]
    detail = json.loads(store.db.execute("SELECT detail FROM assessments a JOIN papers p ON p.id=a.paper_id "
                                         "WHERE a.stage='llm' AND p.title='Real comment'").fetchone()[0])
    assert detail["article_type"] == "research"   # the model's disagreement is still recorded
    store.close()

    # set-type, and `types --all` leaves your types and LLM corrections alone
    import paperflow.cli as cli
    cfg = Config(database=tmp / "db.sqlite", cache=tmp / "c.sqlite")
    cli.cmd_set_type(cfg, Namespace(paper="2", type="review"))
    store = Store(tmp / "db.sqlite")
    assert tuple(store.db.execute("SELECT article_type, article_type_source FROM papers WHERE id=2").fetchone()) == ("review", "you")
    q = store.db.execute("SELECT count(*) FROM papers WHERE status IN ('resolved','no_abstract','unmatched') "
                         "AND coalesce(article_type_source,'') NOT IN ('you','llm')").fetchone()[0]
    assert q == 2, q   # what `types --all` would revisit: 3 and 6; not 1, 5 (llm) or 2, 4 (you)
    store.close()
    shutil.rmtree(tmp)
    print("ok: LLM corrections only replace weak evidence; set-type sticks")


def test_rules():
    cases = [
        # (pub_types, title, venue, expected type, expected source)
        (["europepmc:Review", "europepmc:Journal Article"], "NLR biology", "Annu Rev Phytopathol", "review", "europepmc"),
        (["europepmc:Journal Article", "europepmc:Comment"], "Sensing effectors from inside", "Cell Host Microbe", "commentary", "europepmc"),
        (["europepmc:News"], "Rice blast in the spotlight", "Nature", "commentary", "europepmc"),
        (["europepmc:Published Erratum"], "Some title", "Plant Cell", "correction", "europepmc"),
        (["europepmc:Journal Article", "europepmc:Research Support, Non-U.S. Gov't"], "Structure of Pik", "Nature", "research", "europepmc"),
        # PubMed says research: overrides OpenAlex's inferred "review"
        (["europepmc:Journal Article", "openalex:review"], "Structure of Pik", "Plant Cell", "research", "europepmc"),
        # PubMed says research even in a "Trends" journal; title rule for corrections still wins
        (["europepmc:Journal Article"], "Correction to: Structure of Pik", "Plant Cell", "correction", "title"),
        (["openalex:review"], "Engineering NLRs", "Plant Biotechnology Journal", "review", "openalex"),
        (["openalex:article", "crossref:journal-article"], "Engineering NLRs", "Plant Biotechnology Journal", "research", "openalex"),
        (["semanticscholar:Review", "openalex:article"], "Engineering NLRs", "X", "review", "semanticscholar"),
        (["crossref:peer-review"], "Review of: a manuscript", "X", "other", "crossref"),
        (["crossref:journal-article"], "NLR engineering: a review", "X", "review", "title"),
        (["crossref:journal-article"], "Plant immune receptors", "Trends in Plant Science", "review", "journal"),
        (["crossref:journal-article"], "Plant immune receptors", "Current Opinion in Plant Biology", "review", "journal"),
        (["crossref:journal-article"], "Reply to: Concerns about Pik", "Nature", "commentary", "title"),
        (["biorxiv:new results"], "A resistosome structure", "bioRxiv", "research", "preprint server"),
        (["biorxiv:new results"], "Plant NLRs: a systematic review", "bioRxiv", "review", "title"),
        ([], "Retraction Note: Structure of Pik", None, "retraction", "title"),
        ([], "State-of-the-art protein design achieves binders", None, "unknown", "none"),
        ([], "Structure of Pik", None, "unknown", "none"),
    ]
    for pub, title, venue, exp_t, exp_s in cases:
        got = determine(pub, title, venue)
        assert got == (exp_t, exp_s), (pub, title, venue, got)
    print(f"ok: {len(cases)} type rules")


class FakeHttp:
    down = set()
    def final_url(self, url): return None
    def get_html(self, url): return None
    def get_json(self, url, params=None, headers=None):
        from paperflow.http import HttpError
        host = url.split("/")[2]
        if host in self.down:
            raise HttpError(f"{host}: skipped (rate limited)")
        q = (params or {}).get("query", "")
        if "europepmc" in url or "ebi.ac.uk" in url:
            recs = {
                'DOI:"10.1111/review.1"': {"source": "MED", "id": "1", "title": "NLR biology", "pmid": "1",
                                            "pubTypeList": {"pubType": ["Review", "Journal Article"]}},
                "EXT_ID:2 AND SRC:MED": {"source": "MED", "id": "2", "title": "Sensing effectors from inside", "pmid": "2",
                                          "pubTypeList": {"pubType": ["Journal Article", "Comment"]},
                                          "commentCorrectionList": {"commentCorrection": [
                                              {"id": "99", "source": "MED", "type": "Comment on", "orderIn": "1"}]}},
                'DOI:"10.1111/research.3"': {"source": "MED", "id": "3", "title": "Structure of Pik", "pmid": "3",
                                              "pubTypeList": {"pubType": "Journal Article"}},
            }
            r = recs.get(q)
            return {"resultList": {"result": [r] if r else []}}
        if "openalex" in url:
            if url.endswith("10.2222/oa.4"):
                return {"display_name": "Engineering NLRs", "type": "review", "doi": "https://doi.org/10.2222/oa.4"}
            return None
        if "biorxiv" in url:
            if url.endswith("2026.01.05.697760"):
                return {"collection": [{"doi": "10.64898/2026.01.05.697760", "version": "1", "date": "2026-01-05",
                                        "title": "A resistosome structure", "type": "new results"}]}
            return {"collection": []}
        if "crossref" in url:
            return {"message": {"type": "journal-article", "DOI": "10.3333/cr.5", "title": ["Plant immune receptors"]}}
        raise AssertionError(url)


def test_backfill():
    tmp = Path(tempfile.mkdtemp())
    store = Store(tmp / "db.sqlite")
    rows = [  # (title, venue, doi, pmid, is_preprint, expected type)
        ("NLR biology", "Annu Rev Phytopathol", "10.1111/review.1", None, 0, "review"),
        ("Sensing effectors from inside", "Cell Host Microbe", "10.1111/comment.2", "2", 0, "commentary"),
        ("Structure of Pik", "Nature", "10.1111/research.3", None, 0, "research"),
        ("Engineering NLRs", "Plant Biotechnol J", "10.2222/oa.4", None, 0, "review"),
        ("Plant immune receptors", "Trends in Plant Science", "10.3333/cr.5", None, 0, "review"),
        ("A resistosome structure", "bioRxiv", "10.64898/2026.01.05.697760", None, 1, "research"),
    ]
    for title, venue, doi, pmid, pre, _ in rows:
        cur = store.db.execute("INSERT INTO papers (title, title_key, venue, doi, is_preprint, status) VALUES (?,?,?,?,?,'resolved')",
                               (title, title.lower(), venue, doi, pre))
        store.db.execute("INSERT INTO identifiers VALUES ('doi', ?, ?)", (doi, cur.lastrowid))
        if pmid:
            store.db.execute("INSERT INTO identifiers VALUES ('pmid', ?, ?)", (pmid, cur.lastrowid))
    store.commit(); store.close()

    import paperflow.cli as cli, paperflow.http as phttp
    real_http = phttp.Http
    phttp.Http = lambda *a, **k: FakeHttp()
    try:
        cfg = Config(database=tmp / "db.sqlite", cache=tmp / "c.sqlite")
        FakeHttp.down = {"api.openalex.org"}           # first pass: OpenAlex unreachable
        cli.cmd_types(cfg, Namespace(all=False, workers=2))
        store = Store(tmp / "db.sqlite")
        got = dict(store.db.execute("SELECT title, article_type FROM papers").fetchall())
        assert got["Engineering NLRs"] is None and got["Plant immune receptors"] is None, got  # left for retry
        assert got["NLR biology"] == "review" and got["Sensing effectors from inside"] == "commentary"
        store.close()
        FakeHttp.down = set()                           # second pass: everything reachable
        cli.cmd_types(cfg, Namespace(all=False, workers=2))
    finally:
        phttp.Http = real_http
    store = Store(tmp / "db.sqlite")
    got = {t: (a, s) for t, a, s in store.db.execute("SELECT title, article_type, article_type_source FROM papers")}
    for title, _, _, _, _, exp in rows:
        assert got[title][0] == exp, (title, got[title])
    rel = json.loads(store.db.execute("SELECT related FROM papers WHERE title LIKE 'Sensing%'").fetchone()[0])
    assert rel == [{"type": "Comment on", "id": "99", "source": "MED"}], rel
    print("ok: backfill", {t: v for t, v in got.items()})

    # The type reaches the model's prompt, stats and exports.
    from paperflow.classify import paper_message
    msg = paper_message({"title": "Sensing effectors from inside", "article_type": "commentary"})
    assert "Article type (from metadata; may be wrong): commentary on other work" in msg
    assert "Article type" not in paper_message({"title": "x", "article_type": "unknown"})
    s = store.stats()
    assert s["article_types"]["review"] == 3, s["article_types"]
    store.close()
    cli.cmd_export(cfg, Namespace(out=str(tmp / "r.csv"), status=None, since=None, new=False, added_since=None,
                                  label=None, type=["review"]))
    import csv
    out = list(csv.DictReader(open(tmp / "r.csv")))
    assert len(out) == 3 and {r["article_type"] for r in out} == {"review"}
    shutil.rmtree(tmp)
    print("ok: prompt, stats, export --type")


if __name__ == "__main__":
    test_rules()
    test_abstract_cues()
    test_backfill()
    test_llm_correction()
    print("ARTICLE TYPE CHECKS PASSED")
