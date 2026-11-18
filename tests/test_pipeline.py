"""End-to-end test: ingest synthetic emails, resolve against mocked APIs, check dedup."""
import json, shutil, sys, tempfile
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.config import Config
from paperflow.cli import cmd_ingest
from paperflow.resolver import Resolver
from paperflow.store import Store

PREPRINT = "10.1101/2024.05.01.590001"
PUBLISHED = "10.1038/s41593-024-00001-x"

def oa_work(doi, title, authors, year, venue, typ="article", abstract=None, pmid=None):
    inv = {}
    for i, w in enumerate((abstract or "").split()):
        inv.setdefault(w, []).append(i)
    return {"doi": f"https://doi.org/{doi}", "display_name": title, "publication_year": year,
            "publication_date": f"{year}-05-01", "type": typ,
            "authorships": [{"author": {"display_name": a}} for a in authors],
            "primary_location": {"source": {"display_name": venue}, "landing_page_url": f"https://x/{doi}"},
            "abstract_inverted_index": inv or None,
            "ids": {"pmid": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"} if pmid else {}}

class FakeHttp:
    def __init__(self): self.calls = []
    def final_url(self, url): return None
    def get_html(self, url): return None
    def get_json(self, url, params=None, headers=None):
        self.calls.append((url, params))
        p = params or {}
        if url == f"https://api.biorxiv.org/details/biorxiv/{PREPRINT}":
            return {"collection": [
                {"doi": PREPRINT, "version": "1", "date": "2024-05-01", "title": "Dopamine neurons encode prediction errors across the whole striatum",
                 "authors": "Garcia, A.; Chen, B.; Okafor, C.", "abstract": "We recorded dopamine neurons.", "published": "NA"},
                {"doi": PREPRINT, "version": "2", "date": "2024-06-01", "title": "Dopamine neurons encode prediction errors across the whole striatum",
                 "authors": "Garcia, A.; Chen, B.; Okafor, C.", "abstract": "We recorded dopamine neurons (v2).", "published": PUBLISHED}]}
        if "api.biorxiv.org" in url: return {"collection": []}
        if url.startswith("https://api.openalex.org/works/doi:"):
            doi = url.split("doi:")[1]
            if doi == PREPRINT:
                return oa_work(PREPRINT, "Dopamine neurons encode prediction errors across the whole striatum",
                               ["Ana Garcia", "Bo Chen", "Chidi Okafor"], 2024, "bioRxiv", "preprint")
            if doi == PUBLISHED:
                return oa_work(PUBLISHED, "Dopamine neurons encode prediction errors across the whole striatum",
                               ["Ana Garcia", "Bo Chen", "Chidi Okafor"], 2024, "Nature Neuroscience",
                               abstract="Published abstract.", pmid="38000001")
            if doi == "10.1016/j.neuron.2024.01.001":
                return oa_work(doi, "Thalamic gating of cortical plasticity during sleep", ["Kenji Tanaka", "Marta Rossi"], 2024, "Neuron")
            if doi == "10.48550/arxiv.2405.01234":
                return oa_work(doi, "A normative model of exploration in foraging animals", ["Jin Park", "Luis Silva"], 2024, "arXiv",
                               "preprint", abstract="We propose a normative model.")
            if doi == "10.1523/jneurosci.0001-24.2024":
                return oa_work(doi, "Cholinergic interneurons pause during reward omission in mice", ["Ryo Ito", "Sunita Banerjee"], 2024,
                               "Journal of Neuroscience", abstract="Cholinergic interneurons pause.")
            return None
        if url == "https://api.openalex.org/works":
            f = p.get("filter", "")
            if "thalamic gating" in f:
                return {"results": [
                    oa_work("10.1016/j.neuron.2023.09.999", "Thalamic gating of cortical plasticity", ["Other Person"], 2023, "Neuron"),
                    oa_work("10.1016/j.neuron.2024.01.001", "Thalamic gating of cortical plasticity during sleep",
                            ["Kenji Tanaka", "Marta Rossi"], 2024, "Neuron")]}
            if "cholinergic interneurons" in f:
                return {"results": [oa_work("10.1523/jneurosci.0001-24.2024", "Cholinergic interneurons pause during reward omission in mice",
                                            ["Ryo Ito", "Sunita Banerjee"], 2024, "Journal of Neuroscience")]}
            return {"results": []}
        if "europepmc" in url:
            q = p.get("query", "")
            if q == "EXT_ID:38000001 AND SRC:MED":
                return {"resultList": {"result": [{"title": "Dopamine neurons encode prediction errors across the whole striatum.",
                        "doi": PUBLISHED, "pmid": "38000001", "pubYear": "2024", "source": "MED", "id": "38000001",
                        "authorString": "Garcia A, Chen B, Okafor C.", "abstractText": "<h4>Background</h4>Published abstract.",
                        "journalInfo": {"journal": {"title": "Nature Neuroscience"}}}]}}
            if q == 'DOI:"10.1016/j.neuron.2024.01.001"':
                return {"resultList": {"result": [{"title": "Thalamic gating of cortical plasticity during sleep.",
                        "doi": "10.1016/j.neuron.2024.01.001", "pmid": "38000002", "pubYear": "2024", "source": "MED", "id": "38000002",
                        "abstractText": "Sleep spindles gate plasticity."}]}}
            return {"resultList": {"result": []}}
        if "crossref" in url or "semanticscholar" in url:
            return None
        raise AssertionError(f"unexpected URL {url}")

def main():
    tmp = Path(tempfile.mkdtemp())
    cfg = Config(database=tmp / "db.sqlite", cache=tmp / "cache.sqlite", follow_redirects=True)
    samples = Path(__file__).parent / "samples"
    cmd_ingest(cfg, Namespace(imap=False, mbox=None, eml_dir=samples, outlook_csv=None, all_senders=True, limit=None))

    store = Store(cfg.database)
    pre = {r["id"]: (r["title"], r["doi"]) for r in store.db.execute("SELECT * FROM papers")}
    print("\nAfter ingest:", json.dumps(pre, indent=1))
    # Expected after ingest: preprint (scholar+biorxiv by DOI, stork published version by title), thalamic (2 scholar emails merged by title),
    # arxiv, astrocytes, cholinergic (stork title-only + plaintext DOI merged by title).
    assert len(pre) == 5, len(pre)

    resolver = Resolver(FakeHttp(), cfg)
    for it in store.pending_items():
        store.apply_resolution(resolver.resolve(it))
    store.dedupe_titles(); store.commit()

    rows = list(store.export_rows())
    print("\nAfter resolve:")
    for r in rows:
        print(f"- [{r['status']}] {r['title']} | {r['venue']} | doi={r['doi']} pub={r['published_doi']} "
              f"| abstract via {r['abstract_source']} | ids={r['identifiers']} "
              f"| seen {len(r['sightings'])}x in {sorted({s['source'] for s in r['sightings']})} | {r['status_detail']}")
    titles = {r["title"] for r in rows}
    assert len(rows) == 5, len(rows)   # preprint + its published version (Stork) merged
    dop = next(r for r in rows if r["title"].startswith("Dopamine"))
    assert sorted(dop["identifiers"]["doi"]) == sorted([PREPRINT, PUBLISHED]), dop["identifiers"]
    assert {s["source"] for s in dop["sightings"]} == {"scholar", "biorxiv", "stork"}
    assert dop["abstract"] == "We recorded dopamine neurons (v2).", dop["abstract"]
    th = next(r for r in rows if r["title"].startswith("Thalamic"))
    assert th["doi"] == "10.1016/j.neuron.2024.01.001" and th["abstract_source"] == "europepmc"
    assert len(th["sightings"]) == 2
    types = {r["title"][:20]: (r["article_type"], r["article_type_source"]) for r in rows}
    print("article types:", types)
    assert th["article_type"] == "research" and th["article_type_source"] == "europepmc", types
    assert dop["article_type"] == "research"
    assert all(r["article_type"] for r in rows if r["status"] != "unmatched"), types
    astro = next(r for r in rows if r["title"].startswith("Astrocytes"))
    assert astro["status"] == "unmatched"
    print("\nStats:", json.dumps(store.stats(), indent=1))
    store.close(); shutil.rmtree(tmp)
    print("\nALL CHECKS PASSED")

class FlakyHttp(FakeHttp):
    """Crossref and Semantic Scholar down; everything else up."""
    def get_json(self, url, params=None, headers=None):
        from paperflow.http import HttpError
        if "crossref" in url or "semanticscholar" in url:
            raise HttpError(url.split("/")[2] + ": ConnectionError")
        return super().get_json(url, params, headers)


def partial_outage():
    tmp = Path(tempfile.mkdtemp())
    cfg = Config(database=tmp / "db.sqlite", cache=tmp / "cache.sqlite")
    cmd_ingest(cfg, Namespace(imap=False, mbox=None, eml_dir=Path(__file__).parent / "samples",
                              outlook_csv=None, all_senders=True, limit=None))
    store = Store(cfg.database)
    resolver = Resolver(FlakyHttp(), cfg)
    for it in store.pending_items():
        store.apply_resolution(resolver.resolve(it))
    store.commit()
    status = {r["title"][:20]: (r["status"], r["status_detail"]) for r in store.export_rows()}
    print("\nPartial outage:", json.dumps(status, indent=1))
    assert status["Dopamine neurons enc"][0] == "resolved"      # didn't need the down sources
    assert status["Thalamic gating of c"][0] == "resolved"
    assert status["Astrocytes regulate "][0] == "error"         # nothing found and sources were down -> retry later
    store.close(); shutil.rmtree(tmp)
    print("PARTIAL-OUTAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
    partial_outage()
