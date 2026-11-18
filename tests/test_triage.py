"""Stage 2 end to end: examples in every format -> embedding filter (TF-IDF) -> LLM (mocked)
-> final labels, your own labels overriding, exports. Runs offline."""
import json, random, shutil, sys, tempfile
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.config import Config
from paperflow.store import Store
from paperflow import triage, classify as clf
from paperflow.embed import Tfidf

random.seed(1)
TOPICS = {
    "nlr": (["NLR", "immune receptor", "effector recognition", "resistosome", "integrated domain", "Pik", "Sr35",
             "hypersensitive response", "rice blast", "Magnaporthe oryzae", "TIR domain", "CC domain", "NRC network"],
            "plant immunity: {a} and {b} in {c} signalling during pathogen infection"),
    "design": (["protein language model", "de novo protein design", "directed evolution", "ProteinMPNN",
                "binder design", "fitness landscape", "AlphaFold", "stability engineering", "enzyme design"],
               "protein engineering with {a}: {b} guided by {c} for improved function"),
    "meta": (["metagenomics", "CRISPR arrays", "phage genomes", "groundwater microbiome", "candidate phyla radiation",
              "genome-resolved metagenomics", "soil bacteria"],
             "environmental microbiology: {a} reveals {b} and {c} in uncultivated lineages"),
    "alz": (["amyloid beta", "tau", "Alzheimer's disease", "plasma biomarkers", "neuroinflammation", "microglia",
             "cognitive decline"],
            "clinical neuroscience: {a} and {b} predict {c} in older adults"),
}
def make(topic, i):
    words, tmpl = TOPICS[topic]
    a, b, c = random.sample(words, 3)
    title = f"{a.capitalize()} and {b} shape {c} ({topic} study {i})"
    abstract = tmpl.format(a=a, b=b, c=c) + ". " + " ".join(random.sample(words, 4)) + " were examined in detail."
    return title, abstract

def build_db(path):
    store = Store(path)
    papers = {}
    for topic, n in (("nlr", 40), ("design", 40), ("meta", 60), ("alz", 160)):
        for i in range(n):
            t, a = make(topic, i)
            cur = store.db.execute("INSERT INTO papers (title, title_key, abstract, status, doi, added_at) VALUES (?,?,?,?,?,?)",
                                   (t, t.lower(), a, "resolved", f"10.9999/{topic}.{i}", "2026-01-01"))
            pid = cur.lastrowid
            store.db.execute("INSERT INTO identifiers VALUES ('doi', ?, ?)", (f"10.9999/{topic}.{i}", pid))
            store.db.execute("INSERT INTO sightings (paper_id, message_id, source, alert_name) VALUES (?,?,?,?)",
                             (pid, f"m{pid}", "biorxiv", f"alert-{topic}"))
            papers.setdefault(topic, []).append(pid)
    store.commit()
    return store, papers

def write_examples(d: Path):
    (d / "mandatory" / "zotero-subcollection").mkdir(parents=True)
    (d / "interest").mkdir(parents=True); (d / "Skip").mkdir(parents=True); (d / "random").mkdir()
    bib = []
    for i in range(12):
        t, a = make("nlr", 100 + i)
        bib.append(f"@article{{k{i},\n  title = {{{{{t}}}}},\n  abstract = {{{a}}},\n  journal = {{Plant Cell}},\n}}")
    bib.append("@comment{jabref-meta: x}")
    (d / "mandatory" / "zotero-subcollection" / "nlr.bib").write_text("\n\n".join(bib))
    (d / "mandatory" / "from_alerts.txt").write_text("# papers that came through my alerts\n10.9999/nlr.0\nhttps://doi.org/10.9999/nlr.1\n")
    ris = []
    for i in range(10):
        t, a = make("design", 100 + i)
        ris.append(f"TY  - JOUR\nTI  - {t}\nAB  - {a}\nDO  - 10.8888/design.{i}\nER  - ")
    (d / "interest" / "design.ris").write_text("\n".join(ris))
    items = [dict(zip(("title", "abstract"), make("design", 200 + i))) for i in range(5)]
    (d / "interest" / "more.json").write_text(json.dumps(items))
    # a real PDF with its DOI on page 1
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(d / "interest" / "paper.pdf"))
    t, a = make("design", 300)
    c.drawString(72, 750, t); c.drawString(72, 730, "https://doi.org/10.7777/pdfpaper.1")
    c.drawString(72, 700, "Abstract"); y = 685
    for k in range(0, len(a), 90):
        c.drawString(72, y, a[k:k + 90]); y -= 14
    c.save()
    (d / "Skip" / "notes.md").write_text("---\ndoi: 10.9999/meta.0\n---\n# Metagenomics paper\nalso https://doi.org/10.9999/meta.1 and 10.9999/meta.2\n")
    (d / "interest" / "notes.docx").write_text("x")

class FakeClient:
    """Labels by keyword, the way a sensible LLM reading the profile would."""
    def __init__(self): self.calls = []; self.messages = self
    def create(self, **kw):
        self.calls.append(kw)
        msg = kw["messages"][0]["content"].lower()
        if "nlr" in msg or "immune receptor" in msg or "effector" in msg or "resistosome" in msg:
            v = {"label": "mandatory", "projects": ["NLR engineering"], "reason": "NLR immunity."}
        elif "protein" in msg:
            v = {"label": "interest", "projects": [], "reason": "Protein design methods."}
        else:
            v = {"label": "skip", "projects": [], "reason": "Unrelated."}
        U = type("U", (), {"input_tokens": 200, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 0, "output_tokens": 40})
        return type("R", (), {"content": [type("B", (), {"text": json.dumps(v)})()], "usage": U()})()

def main():
    tmp = Path(tempfile.mkdtemp())
    store, papers = build_db(tmp / "db.sqlite")
    write_examples(tmp / "examples")

    summary, problems = triage.sync_examples(store, tmp / "examples", resolver=None)
    print("examples:", summary); print("problems:", *problems, sep="\n  ")
    assert summary == {"mandatory": 14, "interest": 16, "skip": 3}, summary
    assert any("random" in p for p in problems) and any("notes.docx" in p for p in problems)
    pdf = store.db.execute("SELECT doi, title, abstract FROM examples WHERE source LIKE '%paper.pdf'").fetchone()
    assert pdf[0] == "10.7777/pdfpaper.1" and pdf[2] and "protein engineering" in pdf[2], pdf
    # examples that are alert papers became judgments, with abstracts taken from the database
    assert len(triage.current_judgments(store)) == 5
    assert store.db.execute("SELECT abstract FROM examples WHERE doi='10.9999/nlr.0'").fetchone()[0]

    r = triage.run_filter(store, Tfidf(), keep_recall=0.95)
    print(f"filter: {r}")
    passed = {pid for (pid,) in store.db.execute("SELECT paper_id FROM assessments WHERE stage='embed' AND label='pass'")}
    rate = {t: len(passed & set(ids)) / len(ids) for t, ids in papers.items()}
    print("pass rate by topic:", {k: round(v, 2) for k, v in rate.items()})
    assert rate["nlr"] >= 0.9 and rate["design"] >= 0.8, rate
    assert rate["alz"] <= 0.05 and rate["meta"] <= 0.3, rate

    fake = FakeClient()
    c = clf.Classifier(clf.AnthropicBackend("claude-test", client=fake))
    profile = tmp / "profile.md"; profile.write_text("# Research profile\n### Project: NLR engineering\n")
    items = clf.candidates(store)
    assert all(it["id"] not in triage.current_judgments(store) for it in items)  # your labels aren't re-judged
    stats = clf.run_classification(store, c, profile, items, workers=3, progress=False)
    print("llm:", dict(stats.counts), "tokens in/cache/out:", stats.input_tokens, stats.cache_read_tokens, stats.output_tokens)
    system = fake.calls[0]["system"][0]
    assert system["cache_control"] == {"type": "ephemeral"} and "<mandatory_examples>" in system["text"]
    assert fake.calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert "Most similar papers they have labelled" in fake.calls[0]["messages"][0]["content"]
    assert len(clf.candidates(store)) == 0   # nothing left to do: no double billing

    final = triage.final_labels(store)
    by_topic = {t: {} for t in papers}
    for t, ids in papers.items():
        for pid in ids:
            lab = final[pid]["label"]; by_topic[t][lab] = by_topic[t].get(lab, 0) + 1
    print("final labels by topic:", by_topic)
    assert by_topic["nlr"].get("mandatory", 0) >= 36 and by_topic["alz"].get("filtered", 0) >= 150

    # You correct one: your label wins and becomes an example for the next run.
    victim = next(pid for pid in papers["design"] if final[pid]["label"] == "interest")
    triage.add_judgment(store, victim, "mandatory", note="actually core to project 2")
    assert triage.final_labels(store)[victim] ["label"] == "mandatory"
    assert triage.final_labels(store)[victim]["by"] == "you"
    assert any(t == store.db.execute("SELECT title FROM papers WHERE id=?", (victim,)).fetchone()[0]
               for t in clf.example_titles(store)["mandatory"])
    r2 = triage.run_filter(store, Tfidf(), keep_recall=0.95, progress=False)
    assert r2.n_pos == r.n_pos + 1

    # A merge carries assessments and labels along.
    a, b = papers["nlr"][5], papers["nlr"][6]
    store.merge(a, b); store.commit()
    assert store.db.execute("SELECT count(*) FROM assessments WHERE paper_id=?", (b,)).fetchone()[0] == 0
    assert store.db.execute("SELECT count(*) FROM assessments WHERE paper_id=?", (a,)).fetchone()[0] == 2
    store.close()

    # CLI: export filtered by label; label command by DOI; dry run.
    from paperflow.cli import cmd_export
    from paperflow.triage_cli import cmd_label, cmd_classify
    cfg = Config(database=tmp / "db.sqlite", cache=tmp / "cache.sqlite", profile=profile, examples_dir=tmp / "examples")
    cmd_export(cfg, Namespace(out=str(tmp / "m.csv"), status=None, since=None, new=False, added_since=None, label=["mandatory"]))
    import csv
    rows = list(csv.DictReader(open(tmp / "m.csv")))
    assert rows and all(r["triage"] == "mandatory" for r in rows) and rows[0]["projects"] in ("NLR engineering", "")
    cmd_label(cfg, Namespace(paper="10.9999/alz.3", label="interest", note=None))
    cmd_classify(cfg, Namespace(dry_run=True, redo=True, limit=1, model=None))
    shutil.rmtree(tmp)
    print("TRIAGE CHECKS PASSED")

if __name__ == "__main__":
    main()
