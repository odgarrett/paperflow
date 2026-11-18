"""Briefings end to end with a stand-in model and voice: notes from full texts in pieces,
write-ups by reading state, caching, ordering, EPUB, scripts, audio, registration, failures."""
import io, json, re, shutil, sys, tempfile, zipfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lxml import etree
from paperflow import classify, tts as ptts
from paperflow.briefing import chunks_for
from paperflow.config import Config
from paperflow.store import Store
from paperflow import fulltext as ft, reading

PROFILE = """# Research profile
### Project: Protein design
- Question: binders
### Project: NLR engineering
- Question: Pik
"""

def long_fulltext(title):
    secs = ["## Introduction\n" + "NLRs are intracellular immune receptors in plants. " * 120,
            "## Results\n" + "The Pik HMA domain bound AVR-Pik with a KD of 3.5 µM (Fig. 2). " * 150,
            "## Materials and methods\n" + "Proteins were expressed in E. coli and purified. " * 400,
            "## Acknowledgements\nWe thank everyone. " * 5,
            "## Discussion\n" + "This supports engineering et al. approaches at ~40 kDa. " * 100]
    return f"# {title}\n\n" + "\n\n".join(secs)

def make_db(tmp):
    store = Store(tmp / "data" / "db.sqlite")
    rows = [  # id, title, project, state, has_fulltext, abstract
        (1, "Structure of the Pik resistosome", "NLR engineering", "new", True, "We solved it."),
        (2, "Sensor NLR evolution", "NLR engineering", "familiar", False, "Sensors evolved twice."),
        (3, "A landmark ZAR1 paper", "NLR engineering", "read", False, "ZAR1 forms a pentamer."),
        (4, "De novo binders for effectors", "Protein design", "saved", False, "We designed binders."),
        (5, "Untitled-ish paper with nothing", "", "new", False, None),
        (6, "Paper that breaks the model", "Protein design", "new", False, "BREAK this one."),
    ]
    for pid, title, project, state, has_ft, abstract in rows:
        store.db.execute("INSERT INTO papers (id, title, title_key, authors, venue, year, doi, abstract, status, "
                         "article_type, is_preprint) VALUES (?,?,?,?,?,?,?,?,'resolved','research',?)",
                         (pid, title, title.lower(), json.dumps(["Ana Garcia", "Bo Chen", "Cy Lee"]), "Nature Plants",
                          2026, f"10.5555/b{pid}", abstract, int(pid == 4)))
        store.db.execute("INSERT INTO identifiers VALUES ('doi', ?, ?)", (f"10.5555/b{pid}", pid))
        store.db.execute("INSERT INTO assessments VALUES (?, 'embed', 'pass', ?, '{}', 'x')", (pid, 1 - pid / 10))
        store.db.execute("INSERT INTO assessments VALUES (?, 'llm', 'mandatory', NULL, ?, 'x')",
                         (pid, json.dumps({"projects": [project] if project else [], "reason": f"Core to {project}."})))
        if state != "new":
            reading.set_state(store, pid, state)
        if has_ft:
            ft.save(store, pid, ft.FullText("ok", "europepmc", "u", long_fulltext(title)))
    store.commit()
    return store

class FakeBackend:
    provider, model, num_ctx = "ollama", "fake:1b", 4096
    def __init__(self): self.calls = []
    def fits(self, chars, max_tokens=400): return chars // 3 + max_tokens + 100 <= self.num_ctx
    def complete(self, system, user, schema=None, max_tokens=400, temperature=0.0):
        self.calls.append({"schema": schema is not None, "user": user, "max_tokens": max_tokens,
                           "system": system[0]["text"]})
        if "BREAK" in user:
            raise RuntimeError("model fell over")
        if schema is not None:
            assert self.fits(len(system[0]["text"]) + len(user), max_tokens), "notes piece too big for context"
            part = re.search(r"part (\d+) of (\d+)", user)
            return json.dumps({"question": "Does Pik bind AVR-Pik?" if part and part.group(1) == "1" else "",
                               "systems": ["Pik", "AVR-Pik"],
                               "approach": ["Crystallography"] if "methods" in user.lower() else [],
                               "results": [f"Finding from piece {part.group(1) if part else 'abstract'} at 3.5 µM"],
                               "limitations": [], "key_figures": ["Figure 2: binding"]}), {}
        length = "full briefing" if "500 to 700 words" in user else re.search(r"(refresher|reminder)", user).group(1)
        body = {"full briefing": "## Why it matters\n**This matters** for your Pik work, et al. said.\n\n"
                                 "- The KD was 3.5 µM (Fig. 2).\n\nSecond paragraph about α-helix packing.",
                "refresher": "Refresher: sensors evolved twice.", "reminder": "Reminder: ZAR1 forms a pentamer."}[length]
        return body, {"input": 100, "output": 50}

class FakeVoice:
    def __init__(self, voice, speed): self.said = []
    def say(self, text):
        self.said.append(text)
        return np.sin(np.linspace(0, 200, 2400)).astype(np.float32) * 0.1

def run(cfg, backend, voice_box, **kw):
    args = dict(papers=None, limit=None, name=None, dry_run=False, no_audio=False, voice=None, rewrite=False, redo=False)
    args.update(kw)
    real = (classify.make_backend, classify.unload, ptts.Kokoro)
    classify.make_backend = lambda cfg, m=None: backend
    classify.unload = lambda b: voice_box.setdefault("unloaded", True)
    ptts.Kokoro = lambda v, s: voice_box.setdefault("voice", FakeVoice(v, s))
    try:
        from paperflow.brief_cli import cmd_brief
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_brief(cfg, Namespace(**args))
        return buf.getvalue()
    finally:
        classify.make_backend, classify.unload, ptts.Kokoro = real

def main():
    tmp = Path(tempfile.mkdtemp())
    make_db(tmp)
    (tmp / "profile.md").write_text(PROFILE)
    cfg = Config(database=tmp / "data" / "db.sqlite", cache=tmp / "c.sqlite", profile=tmp / "profile.md",
                 briefings_dir=tmp / "briefings", pronunciations=tmp / "pronunciations.txt", tts="kokoro")

    # chunking: back matter dropped, long methods shortened, pieces fit
    pieces = chunks_for(long_fulltext("T"), 9000)
    joined = "\n".join(pieces)
    assert "We thank everyone" not in joined and "[…]" in joined and all(len(p) <= 9000 for p in pieces)

    out = run(cfg, FakeBackend(), {}, dry_run=True)
    print(out.strip().split("\n\n")[0])
    assert "Protein design" in out.split("NLR engineering")[0]           # profile order
    assert re.search(r"1\. id 4\s+full briefing\s+from abstract\s+De novo binders", out)   # saved -> full
    assert re.search(r"id 1\s+full briefing\s+from full text \(\d+ pieces?\)\s+Structure of the Pik", out)
    assert re.search(r"id 2\s+refresher\s+from abstract\s+Sensor NLR", out)
    assert re.search(r"id 3\s+reminder\s+from abstract\s+A landmark", out)
    assert re.search(r"id 5\s+full briefing\s+from title only", out)
    assert "would be briefed without a full text" in out and "id<number>.pdf" in out

    backend, box = FakeBackend(), {}
    out = run(cfg, backend, box)
    print(out.strip())
    assert "1 paper(s) failed" in out and "[6]" in out
    notes_calls = [c for c in backend.calls if c["schema"]]
    assert sum("part " in c["user"] for c in notes_calls) >= 3           # the full text went in pieces
    write_calls = {re.search(r"Paper: (.+)", c["user"]).group(1): c for c in backend.calls if not c["schema"]}
    assert "500 to 700 words" in write_calls["Structure of the Pik resistosome"]["user"]
    for c in write_calls.values():   # content only: no profile, no relevance framing
        assert "<research_profile>" not in c["system"] and "Do not comment on why the paper matters" in c["system"]
        assert "flagged" not in c["user"] and "Project:" not in c["user"]
    assert "from the full text" in write_calls["Structure of the Pik resistosome"]["user"]
    assert "from the abstract only" in write_calls["Sensor NLR evolution"]["user"]

    bdir = next((tmp / "briefings").iterdir())
    epub = next(bdir.glob("*.epub"))
    with zipfile.ZipFile(epub) as z:
        names = z.namelist()
        assert names[0] == "mimetype" and z.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        for n in names:
            if n.endswith((".xhtml", ".opf", ".xml")):
                etree.fromstring(z.read(n))                              # well-formed XML
        ch1 = z.read("OEBPS/ch01.xhtml").decode()
        nav = z.read("OEBPS/nav.xhtml").decode()
    assert "Protein design" in ch1 and "1. De novo binders" in nav and "##" not in ch1 and "**" not in ch1
    md = (bdir / "briefing.md").read_text()
    assert "### 1. De novo binders" in md and "you: saved" in md and "Paper that breaks" not in md
    assert md.index("De novo binders") < md.index("Structure of the Pik") < md.index("Sensor NLR") < md.index("A landmark")
    script = (bdir / "script.txt").read_text()
    assert "Garcia and colleagues, in Nature Plants, 2026." in script
    assert "3.5 micromolar (Figure 2)" in script and "alpha-helix" in script and "et al" not in script
    assert "This briefing is based on the abstract only." in script
    assert script.index("[Paper 3: Sensor NLR evolution]") < script.index("Paper 3. Sensor NLR evolution.")
    audio = bdir / f"{bdir.name}.mp3"
    assert audio.exists() and audio.stat().st_size > 1000 and not (bdir / "audio").exists() and box.get("unloaded")
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(audio)],
                           capture_output=True, text=True).stdout
    chapters = [c["tags"]["title"] for c in json.loads(probe)["chapters"]]
    print("   chapters:", chapters)
    assert chapters[0] == "Introduction" and chapters[1] == "Paper 1: De novo binders for effectors"
    assert len(chapters) == 6 and chapters[-1].startswith("Paper 5:")
    assert (tmp / "pronunciations.txt").exists()

    store = Store(cfg.database)
    name, pids = reading.briefing_papers(store, "latest")
    assert pids == [4, 1, 2, 3, 5], pids                                 # numbering = briefing order, failure removed
    store.close()
    print("ok: notes in pieces, lengths by reading state, profile order, EPUB valid, scripts, MP3s, registered")

    # Rerun: nothing changed, so no model calls except the paper that failed before.
    backend2 = FakeBackend()
    run(cfg, backend2, {}, no_audio=True)
    assert all("BREAK" in c["user"] for c in backend2.calls), [c["user"][:60] for c in backend2.calls]
    # Mark paper 1 familiar: only its write-up is redone, as a refresher; notes reused.
    store = Store(cfg.database); reading.set_state(store, 1, "familiar"); store.commit(); store.close()
    backend3 = FakeBackend()
    run(cfg, backend3, {}, no_audio=True)
    redone = [c for c in backend3.calls if "BREAK" not in c["user"]]
    assert len(redone) == 1 and not redone[0]["schema"] and "refresher" in redone[0]["user"], redone
    print("ok: rerun reuses notes and write-ups; a changed reading state redoes just that write-up")

    out = run(cfg, FakeBackend(), {}, papers="2,10.5555/b3", no_audio=True, name="pilot")
    assert "Briefing 'pilot': 2 mandatory, 0 of interest" in out
    cfg.audio_format = "m4b"
    run(cfg, FakeBackend(), {}, papers="2,10.5555/b3", name="pilot-m4b")
    m4b = tmp / "briefings" / "pilot-m4b" / "pilot-m4b.m4b"
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(m4b)],
                           capture_output=True, text=True).stdout
    assert [c["tags"]["title"] for c in json.loads(probe)["chapters"]][1] == "Paper 1: Sensor NLR evolution"
    print("ok: M4B with chapters")
    shutil.rmtree(tmp)
    print("BRIEFING CHECKS PASSED")

if __name__ == "__main__":
    main()
