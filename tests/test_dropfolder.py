"""Matching PDFs you drop into the full-text folder to the right paper."""
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.store import Store
from paperflow.fulltext import scan_drop_folder

def pdf(path, lines_page1, body=300):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path)); y = 800
    def line(t):
        nonlocal y
        if y < 60: c.showPage(); y = 800
        c.drawString(40, y, t); y -= 14
    for t in lines_page1: line(t)
    for i in range(body): line(f"Body sentence {i} about immune receptors and pathogen effectors in plants.")
    c.save()

def main():
    tmp = Path(tempfile.mkdtemp())
    store = Store(tmp / "data" / "db.sqlite")
    papers = {
        1: ("Structural basis for effector recognition by a paired plant immune receptor", "10.1111/own.1"),
        2: ("A widely cited earlier paper on helper NLR networks in Solanaceae", "10.1111/cited.2"),
        3: ("Designed protein binders neutralise a fungal effector in rice leaves", "10.1111/nodoi.3"),
        4: ("Some preprint about resistosome assembly dynamics", "10.64898/2026.01.05.697760"),
        5: ("Paper for the id file name test", "10.1111/idname.5"),
        6: ("Title that won't extract from this publisher PDF at all", "10.1111/meta.6"),
    }
    for pid, (title, doi) in papers.items():
        store.db.execute("INSERT INTO papers (id, title, title_key, doi, status) VALUES (?,?,?,?,'resolved')",
                         (pid, title, title.lower(), doi))
        store.db.execute("INSERT INTO identifiers VALUES ('doi', ?, ?)", (doi, pid))
        if doi.startswith("10.64898"):
            store.db.execute("INSERT INTO identifiers VALUES ('biorxiv', '2026.01.05.697760', ?)", (pid,))
    store.commit()
    drop = tmp / "fulltext"; drop.mkdir()
    # 1: cites paper 2's DOI before printing its own: must still go to paper 1 (title on the page)
    pdf(drop / "nature_article.pdf", ["Structural basis for effector recognition by a paired plant immune",
                                      "receptor", "Building on doi:10.1111/cited.2 we show...", "https://doi.org/10.1111/own.1"])
    # only a cited DOI, and a title not in the database: not matched
    pdf(drop / "unrelated.pdf", ["A completely different paper about wheat rust epidemiology", "see 10.1111/cited.2"])
    # 3: no DOI at all; title matched (wrapped over two lines)
    pdf(drop / "download (3).pdf", ["Designed protein binders neutralise a fungal", "effector in rice leaves", "Abstract"])
    # 4: bioRxiv's default file name, DOI with version suffix
    pdf(drop / "2026.01.05.697760v2.full.pdf", ["Some preprint about resistosome assembly dynamics",
                                                "doi: https://doi.org/10.64898/2026.01.05.697760"])
    # 5: assigned by file name, whatever is inside
    pdf(drop / "id5 - from my library.pdf", ["Scanned-looking header without useful text"])
    pdf(drop / "id999.pdf", ["Whatever"])
    # 6: title garbled in the text layer, but the publisher's metadata carries the DOI
    from pypdf import PdfReader, PdfWriter
    pdf(tmp / "raw6.pdf", ["T1tle th@t w0n't extr@ct", "Cited: 10.1111/cited.2"])
    w = PdfWriter(clone_from=PdfReader(str(tmp / "raw6.pdf")))
    w.add_metadata({"/doi": "10.1111/meta.6"})
    with open(drop / "publisher.pdf", "wb") as f:
        w.write(f)
    added, problems = scan_drop_folder(store, drop)
    store.commit()
    got = {pid: (src, det) for pid, src, det in store.db.execute("SELECT paper_id, source, detail FROM fulltexts")}
    print("matched:", {pid: d for pid, (_, d) in got.items()})
    print("problems:", *problems, sep="\n  ")
    assert set(got) == {1, 3, 4, 5, 6}, got
    assert got[6][1] == "matched by DOI in the PDF's metadata"
    assert 2 not in got                                   # the cited paper didn't get paper 1's PDF
    assert got[1][1] == "matched by DOI and title" and got[3][1] == "matched by title"
    assert got[5][1] == "matched by file name"
    assert any("unrelated.pdf" in p and "other papers" in p for p in problems)
    assert any("id999.pdf" in p and "no paper with that id" in p for p in problems)
    assert all("Rename it id<number>.pdf" in p for p in problems)
    # rerun: matched files aren't re-read; unmatched ones are looked at again (after renaming, say)
    (drop / "unrelated.pdf").rename(drop / "id2.pdf")
    added2, problems2 = scan_drop_folder(store, drop)
    assert added2 == 1 and len(problems2) == 1, (added2, problems2)
    store.close(); shutil.rmtree(tmp)
    print("DROP FOLDER CHECKS PASSED")

if __name__ == "__main__":
    main()
