"""Parsers against the real alert layouts seen in Outlook plain-text exports (content made up)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paperflow.parsers import biorxiv, scholar_text, stork
from paperflow.links import extract_ids

N = "\r\n"
SCHOLAR = N.join([
    "", "", "",
    "A test title about NLR immune receptors <https://scholar.google.com/scholar_url?url=https://academic.oup.com/plcell/advance-article-pdf/doi/10.1093/plcell/koag001/123456/koag001.pdf&hl=en&folt=kw-top> ",
    "", "A Author, B Author… - The Plant Cell, 2026",
    "… a snippet line …", " <https://scholar.google.com/citations?hl=en&update_op=email_library_add> \t <https://scholar.google.com/scholar_share?x>",
    "", "",
    "[HTML] A second, very long title that Scholar truncates before the end of the …"
    " <https://scholar.google.com/scholar_url?url=https://www.biorxiv.org/content/10.64898/2026.02.01.700001v1.abstract&hl=en&folt=kw-top> ",
    "", "C Author - bioRxiv, 2026", "… snippet …", " <https://scholar.google.com/citations?x>", "", "",
    "This message was sent by Google Scholar because you're following new results for [NLR] <https://scholar.google.com/scholar?q=NLR> . "
    "<https://scholar.google.com/scholar_url?url=https://scholar.google.com/scholar/images/cleardot.gif&hl=en&folt=kw-top> ",
])

COLLECTION = N.join([
    " <http://biorxiv.org/> ", "You have requested alerts for the following bioRxiv online collections: <http://biorxiv.org/cgi/collection?collection> ",
    "________________________________", "", "Plant Biology <http://biorxiv.org/cgi/collection/plant_biology>  ", "________________________________", "",
    "A collection item title", "Ana One, Bo Two, and Cy Three", "bioRxiv posted 6 January 2026 doi:10.64898/2026.01.05.697760",
    "[Abstract] <http://www.biorxiv.org/cgi/content/abstract/2026.01.05.697760v1??collection>  [PDF] <http://www.biorxiv.org/cgi/reprint/2026.01.05.697760v1??collection> ",
    "", "An older-prefix item", "Di Four", "bioRxiv posted 2 June 2025 doi:10.1101/2025.06.01.650000",
    " [Abstract] <http://www.biorxiv.org/cgi/content/abstract/2025.06.01.650000v2??collection> ", "",
])

CITETRACK = N.join([
    'Alert Results for "NLR pathogen effector resistance preprints" in bioRxiv',
    "Your CiteTrack Alert found 22 articles matching your criteria. Below are results 1 to 10.",
    "________________________________", "", "Search Criteria: ", "Anywhere in Article: NLR", "________________________________", "",
    "Results:", "A CiteTrack item", "Max A. One, Hong Two and Sue Three",
    "bioRxiv. posted 3 September 2026, 10.64898/2026.09.02.748808 ",
    "http://biorxiv.org/content/early/2026/10/09/2026.09.02.748808 <http://biorxiv.org/content/early/2026/10/09/2026.09.02.748808?ct> ", "",
])

STORK_NEW = N.join([
    "Oscar,", "Stork has brought you 1 new publications. Listen <https://www.storkapp.me/g/abc> ",
    " Protein language model\t Big Analysis › <https://www.storkapp.me/g/def> \t",
    "Direct Link to Journal <https://www.storkapp.me/g/xyz>   A Stork paper title. <https://www.storkapp.me/g/ghi> ",
    "Jo One, Kim Two, …, Lee Nine · 2026", "BioinformaticsIF 5.5", "PMID 42886502 · doi 10.1093/bioinformatics/btag778",
    " <mailto:?subject=Stork%3A%20A%20Stork%20paper%20title.&body=I%27d%20like%20to%20share%20a%20paper%20with%20you.%0A%0DA%20Stork%20paper%20title.%0A%0Dby%20Jo%20One%2C%20Kim%20Two%2C%20%E2%80%A6%2C%20Lee%20Nine%20%282026%29%0A%0Dhttps%3A%2F%2Fwww.storkapp.me%2Fpubpaper%2F42886502%0A%0DPowered%20by%20Stork>   <https://www.storkapp.me/g/s1> ",
])

STORK_OLD = N.join([
    "Oscar, ", "", "Stork has brought you 1 new publications.", "",
    "Protein language model     Big Analysis <https://www.storkapp.me/goTo.php?id=1&url=https://www.storkapp.me/meta/> ", "Papers", "",
    "An old-layout Stork title. <https://www.storkapp.me/goTo.php?id=1&url=https://www.storkapp.me/paper/showPaper.php?id=2374143843>  by Ge One, Su Two, …, Ro Three (2026) Nucleic Acids Res (impact factor: 13.1)",
    "PMID: 41495893 doi: 10.1093/nar/gkaf1448",
    " <mailto:?subject=Stork:%20An+old-layout+Stork+title.&body=I%27d+like+to+share+a+paper+with+you.%0A%0DAn+old-layout+Stork+title.%0A%0Dby+Ge+One%2C+Su+Two%2C+%E2%80%A6%2C+Ro+Three+%282026%29%0A%0Dhttps%3A%2F%2Fwww.storkapp.me%2Fpubpaper%2F41495893%0A%0DPowered+by+Stork>  ",
])

HOT_NEW = N.join(["Oscar,", "Hot papers in your field last month:",
                  "1\tA hot paper. <https://www.storkapp.me/g/Fip> ", "Pe One, Is Two, …, Je Three · 2026", "Science (New York, N.Y.)IF 47.3"])
HOT_OLD = N.join(["Oscar,", "", "Hot papers in your field last month:", "", "Rank\tPaper\t",
                  "1\tAn old hot paper. <https://www.storkapp.me/goTo.php?id=1&url=https://www.storkapp.me/paper/showPaper.php?id=255> ",
                  "Pr One, Yo Two, …, Ge Yeo et al. Molecular cell (impact factor: 16) (2026)\t"])


def test_scholar_text():
    name, kind, s = scholar_text.parse(SCHOLAR, "NLR - new results")
    assert (name, kind, len(s)) == ("NLR", "keyword", 2), (name, kind, len(s))
    assert s[0].ids == {"doi": "10.1093/plcell/koag001"}, s[0].ids          # Silverchair suffix trimmed
    assert s[0].authors_raw == "A Author, B Author…" and s[0].venue_raw == "The Plant Cell" and s[0].year == 2026
    assert s[1].title_truncated and s[1].title.startswith("A second") and not s[1].title.startswith("[HTML]")
    assert s[1].ids == {"biorxiv": "2026.02.01.700001", "doi": "10.64898/2026.02.01.700001"}


def test_biorxiv_collection():
    name, kind, s, n = biorxiv.parse(COLLECTION, "bioRxiv Subject Collection Alert")
    assert kind == "collection" and len(s) == 2 and n is None
    assert s[0].alert == "bioRxiv collection: Plant Biology"
    assert s[0].ids == {"biorxiv": "2026.01.05.697760", "doi": "10.64898/2026.01.05.697760"}
    assert s[0].authors_raw == "Ana One; Bo Two; Cy Three" and s[0].posted == "2026-01-06"
    assert s[1].ids["doi"] == "10.1101/2025.06.01.650000"


def test_biorxiv_citetrack():
    name, kind, s, n = biorxiv.parse(CITETRACK, "bioRxiv: NLR pathogen effector resistance preprints")
    assert (name, kind, n, len(s)) == ("NLR pathogen effector resistance preprints", "keyword", 22, 1)
    assert s[0].ids == {"biorxiv": "2026.09.02.748808", "doi": "10.64898/2026.09.02.748808"}
    assert s[0].authors_raw == "Max A. One; Hong Two; Sue Three"


def test_stork_layouts():
    for text, pmid, doi, venue in ((STORK_NEW, "42886502", "10.1093/bioinformatics/btag778", "Bioinformatics"),
                                   (STORK_OLD, "41495893", "10.1093/nar/gkaf1448", "Nucleic Acids Res")):
        name, kind, s, n = stork.parse(text, "Stork has brought you 1 new publications for Protein language model etc")
        assert (name, n, len(s)) == ("Stork: Protein language model", 1, 1), (name, n, len(s))
        assert s[0].ids == {"pmid": pmid, "doi": doi}, s[0].ids
        assert s[0].alert == "Stork: Protein language model" and s[0].venue_raw == venue and s[0].year == 2026
        assert s[0].authors_raw.startswith(("Jo One", "Ge One")) and "…" not in s[0].authors_raw


def test_stork_hot():
    for text, venue in ((HOT_NEW, "Science (New York, N.Y.)"), (HOT_OLD, "Molecular cell")):
        name, kind, s, n = stork.parse(text, "Hot papers in your field last month")
        assert kind == "hot" and len(s) == 1 and s[0].venue_raw == venue and s[0].year == 2026 and s[0].authors_raw


def test_doi_cleaning():
    assert extract_ids("https://link.springer.com/content/pdf/10.1186/s13321-025-01144-8_reference.pdf")["doi"] == "10.1186/s13321-025-01144-8"
    assert extract_ids("https://pubs.acs.org/jafcau/article/doi/10.1021/acs.jafc.6c07885/5431444")["doi"] == "10.1021/acs.jafc.6c07885"
    assert extract_ids("https://onlinelibrary.wiley.com/doi/10.1155/2024/1234567")["doi"] == "10.1155/2024/1234567"  # real slashes kept


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("ALL FORMAT TESTS PASSED")
