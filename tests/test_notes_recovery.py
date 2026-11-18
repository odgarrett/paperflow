"""Notes replies that are cut off or unreadable: split, salvage, or mark incomplete and redo."""
import json, re, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow.briefing import Paper, Writer
from paperflow.store import Store

GOOD = {"question": "Q?", "systems": ["Pik"], "approach": ["ITC"], "results": [], "limitations": [], "key_figures": []}

class Model:
    """Replies are cut off for pieces longer than `cut_above` characters (like a model that
    writes too many findings); `garbage_for` pieces get an unreadable reply."""
    provider, model, num_ctx = "ollama", "fake", 16384
    def __init__(self, cut_above=10**9, garbage_for=()):
        self.cut_above, self.garbage_for, self.calls = cut_above, garbage_for, []
    def fits(self, *a): return True
    def complete(self, system, user, schema=None, max_tokens=400, temperature=0.0):
        piece = user.split("\n\n", 1)[1]
        self.calls.append(len(piece))
        tag = re.search(r"PIECE(\d+)", piece)
        if any(f"PIECE{g}" in piece for g in self.garbage_for):
            return "Sorry, I lost track of the format here.", {"truncated": False}
        found = sorted(set(re.findall(r"PIECE\d+", piece)))
        reply = json.dumps({**GOOD, "results": [f"finding from {t}" for t in found]})
        if len(piece) > self.cut_above:
            return reply[: len(reply) - 25], {"truncated": True}   # cut off mid-list
        return reply, {"truncated": False}

def fulltext(n_sections, size):
    return "\n\n".join(f"## Section {i}\n" + "\n\n".join(f"PIECE{i} " + "word " * (size // 10) for _ in range(4))
                       for i in range(1, n_sections + 1))

def paper(pid, text=None, abstract=None):
    return Paper(id=pid, title=f"Paper {pid}", authors=[], venue=None, year=None, doi=None, is_preprint=False,
                 article_type=None, abstract=abstract, reason=None, project="P", state="new", score=0,
                 fulltext=text)

def main():
    tmp = Path(tempfile.mkdtemp())
    store = Store(tmp / "data" / "db.sqlite")
    for pid in (1, 2, 3):
        store.db.execute("INSERT INTO papers (id, title, status) VALUES (?, ?, 'resolved')", (pid, f"Paper {pid}"))
    store.commit()

    # 1. One big piece whose reply is cut off: redone in halves (and quarters) until it fits.
    m = Model(cut_above=9000)
    w = Writer(store, m, "profile", chunk_chars=40000)
    p = paper(1, fulltext(2, 4000))
    w.notes(p)
    found = sorted(p.notes["results"])
    print("split:", m.calls, found)
    assert found == ["finding from PIECE1", "finding from PIECE2"] and "_incomplete" not in p.notes
    assert len(m.calls) > 1 and w.recovered >= 1 and not w.issues

    # 2. A short abstract whose reply is cut off: the complete items are kept.
    m = Model(cut_above=50)
    w = Writer(store, m, "profile", chunk_chars=40000)
    p = paper(2, abstract="PIECE7 " + "word " * 40)
    w.notes(p)
    print("salvaged:", p.notes)
    assert p.notes["question"] == "Q?" and p.notes["systems"] == ["Pik"] and "_incomplete" not in p.notes

    # 3. An unreadable reply for one piece: kept out, marked incomplete, dumped, redone next run.
    m = Model(garbage_for=(2,))
    w = Writer(store, m, "profile", chunk_chars=5000)
    p = paper(3, fulltext(3, 4000))
    w.notes(p)
    print("incomplete:", p.notes.get("_incomplete"), "|", w.issues[0])
    assert p.notes["_incomplete"] >= 1 and "finding from PIECE1" in p.notes["results"]
    assert "finding from PIECE2" not in p.notes["results"] and w.debug_files and w.debug_files[0].exists()
    assert "lost track" in w.debug_files[0].read_text()
    m2 = Model()
    w2 = Writer(store, m2, "profile", chunk_chars=5000)
    p = paper(3, fulltext(3, 4000))
    w2.notes(p)                                                       # same inputs, but incomplete: redone
    assert m2.calls and "finding from PIECE2" in p.notes["results"] and "_incomplete" not in p.notes
    m3 = Model()
    Writer(store, m3, "profile", chunk_chars=5000).notes(paper(3, fulltext(3, 4000)))
    assert not m3.calls                                               # complete now: reused
    store.close(); shutil.rmtree(tmp)
    print("NOTES RECOVERY CHECKS PASSED")

if __name__ == "__main__":
    main()
