"""Briefings: a readable (and listenable) section per paper, grouped by project.

For each paper:
  1. notes: the LLM reads the full text in pieces that fit its context window (or the
     abstract, when there's no full text) and records the question, systems, approach,
     results (with numbers), limitations and key figures, using only what the text says;
  2. write-up: the LLM turns the notes into prose about the paper's content alone (no commentary
     on relevance; your research profile isn't used here), as long as your reading state calls for:
     new/saved/unmarked: a full briefing (~500–700 words); familiar: a refresher (~100–150);
     read: a one- or two-sentence reminder.
Both are stored and reused; they're only redone when their inputs change (new full text,
different reading state, other model) or when you ask.

Papers are grouped by project, in the order your research profile lists them, and numbered
1, 2, 3… across the briefing, which is recorded so `paperflow review --briefing latest` can
walk through it in the same order.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .store import utcnow

log = logging.getLogger(__name__)

NOTES_VERSION = "2"   # 2: longer replies, conciseness rules, recovery from cut-off replies
WRITE_VERSION = "2"   # 2: content only, no relevance commentary, no research profile

NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "systems": {"type": "array", "items": {"type": "string"}},
        "approach": {"type": "array", "items": {"type": "string"}},
        "results": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "key_figures": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "systems", "approach", "results", "limitations", "key_figures"],
    "additionalProperties": False,
}
NOTE_LISTS = ("systems", "approach", "results", "limitations", "key_figures")
NOTE_CAPS = {"systems": 12, "approach": 8, "results": 16, "limitations": 8, "key_figures": 8}
NOTES_MAX_TOKENS = 1600
MIN_SPLIT_CHARS = 6000   # a piece shorter than this is salvaged rather than split again

NOTES_SYSTEM = """You take notes on a scientific paper for a researcher who will be briefed on it.
You are given one part of the paper, or its abstract. Record only what this text states: do not
add background knowledge, and do not guess.

- question: the question or aim of the paper, if this part states it; otherwise "".
- systems: organisms, proteins, genes, pathogens, cell types or other systems studied.
- approach: the methods and experimental approach, briefly.
- results: the specific findings, with numbers, effect sizes and statistics where given. One
  finding per item, each understandable on its own.
- limitations: caveats, limitations or open questions the authors acknowledge.
- key_figures: figures or tables that carry the main findings, as "Figure 2: what it shows".

Leave a list empty when this part says nothing about it. Be concise: one short sentence per
item, at most 12 results and at most 6 items in each other list; keep the most important."""

WRITE_SYSTEM = """You write one section of a research briefing: a section about one paper. Its purpose is
to convey as much of the paper's substance as the length allows, so that the reader understands
what the paper did and found without reading it.

- Spend every sentence on the paper's content: its question, approach, results and conclusions.
  Do not comment on why the paper matters, who should read it, or how it relates to the reader's
  work, and don't open with a summary of its importance.
- Write plain prose paragraphs. No headings, lists, tables, markdown, links or citation brackets.
  The text is read on an e-reader and also listened to as audio.
- Spell out each abbreviation the first time you use it, except very common ones (DNA, RNA, PCR).
- Use the exact numbers from the notes, rounding long ones. Refer to a figure only as a pointer
  ("Figure 3 shows this"); never rely on the reader seeing it.
- Use only the notes provided. Never add findings, numbers, methods or claims that are not in them.
  If the notes come from the abstract only, say so once, briefly."""

LENGTHS = {
    "full": ("Write about 500 to 700 words: the question and why it was open, the systems and approach, "
             "the main findings with their numbers and how they fit together, the conclusions the authors "
             "draw, and the limitations and open questions.", 1300),
    "refresher": ("The reader already knows this paper's gist. Write a refresher of about 100 to 150 words: "
                  "the core finding, and the one or two details most worth remembering.", 350),
    "reminder": ("The reader has read this paper. Write a reminder of one or two sentences saying what it "
                 "showed.", 150),
}
STATE_LENGTH = {"new": "full", "unmarked": "full", "saved": "full", "familiar": "refresher", "read": "reminder"}
STATE_ORDER = {"full": 0, "refresher": 1, "reminder": 2}

_DROP_SECTIONS = re.compile(r"acknowledg|author contribution|funding|competing|conflicts? of interest|"
                            r"data availab|code availab|supplementa|declaration|ethics|abbreviations", re.I)
_METHODS = re.compile(r"method|materials|experimental procedures", re.I)
METHODS_MAX_CHARS = 4000


def _hash(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


# -- chunking ------------------------------------------------------------------------------

def _sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs from '## Heading' markup (JATS full texts); one block otherwise."""
    parts = re.split(r"^(#{1,3} .+)$", text, flags=re.M)
    if len(parts) == 1:
        return [("", text)]
    out, head = [], ""
    if parts[0].strip():
        out.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        head = parts[i].lstrip("#").strip()
        out.append((head, parts[i + 1] if i + 1 < len(parts) else ""))
    return out


def chunks_for(text: str, max_chars: int) -> list[str]:
    """Split a full text into pieces of at most max_chars, at section then paragraph boundaries.
    Back-matter is dropped and long methods sections are shortened."""
    blocks = []
    for head, body in _sections(text):
        if head and _DROP_SECTIONS.search(head):
            continue
        body = body.strip()
        if head and _METHODS.search(head) and len(body) > METHODS_MAX_CHARS:
            body = body[:METHODS_MAX_CHARS] + " […]"
        block = (f"## {head}\n" if head else "") + body
        if block.strip():
            blocks.append(block)
    pieces: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            pieces.append(block)
            continue
        cur = ""
        for para in re.split(r"\n\s*\n", block):
            while len(para) > max_chars:  # a single enormous paragraph (PDF text): hard split
                pieces.append(para[:max_chars])
                para = para[max_chars:]
            if len(cur) + len(para) + 2 > max_chars:
                pieces.append(cur)
                cur = ""
            cur += ("\n\n" if cur else "") + para
        if cur:
            pieces.append(cur)
    out, cur = [], ""
    for p in pieces:  # pack small neighbours together
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + p
    if cur.strip():
        out.append(cur)
    return out


def merge_notes(parts: list[dict]) -> dict:
    merged = {"question": "", **{k: [] for k in NOTE_LISTS}}
    seen = {k: set() for k in NOTE_LISTS}
    for n in parts:
        q = (n.get("question") or "").strip()
        if q and not merged["question"]:
            merged["question"] = q
        for k in NOTE_LISTS:
            for item in n.get(k) or []:
                item = str(item).strip()
                key = re.sub(r"\W+", " ", item.lower()).strip()
                if item and key not in seen[k] and len(merged[k]) < NOTE_CAPS[k]:
                    seen[k].add(key)
                    merged[k].append(item)
    return merged


def notes_as_text(n: dict) -> str:
    lines = []
    if n.get("question"):
        lines.append(f"Question: {n['question']}")
    if n.get("systems"):
        lines.append("Systems: " + "; ".join(n["systems"]))
    for k, title in (("approach", "Approach"), ("results", "Results"), ("limitations", "Limitations"),
                     ("key_figures", "Key figures")):
        if n.get(k):
            lines.append(f"{title}:\n" + "\n".join(f"- {x}" for x in n[k]))
    return "\n".join(lines)


# -- the papers ----------------------------------------------------------------------------

@dataclass
class Paper:
    id: int
    title: str
    authors: list
    venue: str | None
    year: int | None
    doi: str | None
    is_preprint: bool
    article_type: str | None
    abstract: str | None
    reason: str | None
    project: str
    state: str
    score: float
    fulltext: str | None = None
    number: int = 0
    notes: dict | None = None
    notes_source: str | None = None
    text: str | None = None

    @property
    def length(self) -> str:
        return STATE_LENGTH.get(self.state, "full")


def profile_projects(profile_text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"^#{2,4}\s*Project:\s*(.+?)\s*$", profile_text, re.M)
            if "<" not in m.group(1)]


def _match_project(name: str | None, known: list[str]) -> str | None:
    if not name:
        return None
    best = max(known, key=lambda k: fuzz.ratio(k.lower(), name.lower()), default=None)
    if best and fuzz.ratio(best.lower(), name.lower()) >= 80:
        return best
    return name


OTHER = "Other papers"


def gather(store, pids: list[int], profile_text: str, start: int = 0) -> list[tuple[str, list[Paper]]]:
    """Load the papers and group them into (project, papers) in briefing order, numbered from start + 1."""
    from .fulltext import load as load_fulltext
    from .reading import current_states
    from .triage import final_labels
    finals = final_labels(store)
    states = current_states(store)
    known = profile_projects(profile_text)
    papers = []
    for pid in pids:
        r = store.db.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        if r is None:
            continue
        f = finals.get(pid) or {}
        llm = (f.get("llm") or {}).get("detail") or {}
        projects = llm.get("projects") or []
        has_ft = store.db.execute("SELECT 1 FROM fulltexts WHERE paper_id=? AND status='ok'", (pid,)).fetchone()
        papers.append(Paper(
            id=pid, title=r["title"] or "(untitled)", authors=json.loads(r["authors"]) if r["authors"] else [],
            venue=r["venue"], year=r["year"], doi=r["doi"], is_preprint=bool(r["is_preprint"]),
            article_type=r["article_type"], abstract=r["abstract"], reason=llm.get("reason"),
            project=_match_project(projects[0] if projects else None, known) or OTHER,
            state=states.get(pid, "unmarked"), score=((f.get("embed") or {}).get("score") or 0.0),
            fulltext=load_fulltext(store, pid) if has_ft else None,
        ))
    order = {name: i for i, name in enumerate(known)}
    groups: dict[str, list[Paper]] = {}
    for p in papers:
        groups.setdefault(p.project, []).append(p)
    sections = sorted(groups.items(), key=lambda kv: (kv[0] == OTHER, order.get(kv[0], len(order)), kv[0]))
    n = start
    out = []
    for project, ps in sections:
        ps.sort(key=lambda p: (STATE_ORDER[p.length], -p.score))
        for p in ps:
            n += 1
            p.number = n
        out.append((project, ps))
    return out


# -- LLM steps -----------------------------------------------------------------------------

def _clean_prose(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:#+\s*|[-*•]\s+|\d+[.)]\s+)", "", line)       # headings, bullets
        line = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), line)
        lines.append(line.strip())
    text = "\n".join(lines)
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p)


class Writer:
    def __init__(self, store, backend, profile_text: str, chunk_chars: int):
        self.store, self.backend = store, backend
        self.model = f"{backend.provider}:{backend.model}"
        self.profile_text = profile_text   # used only to order papers by project, not in the prompts
        self.chunk_chars = chunk_chars
        self.calls = 0
        self.recovered = 0                   # replies rescued by splitting or salvaging
        self.issues: list[str] = []
        self.debug_files: list = []

    # notes --------------------------------------------------------------------------------
    def notes(self, p: Paper, redo: bool = False) -> None:
        source = "full text" if p.fulltext else "abstract"
        material = p.fulltext or p.abstract or ""
        if not material.strip():
            p.notes, p.notes_source = merge_notes([]), "title only"
            return
        h = _hash(NOTES_VERSION, self.model, source, material)
        row = self.store.db.execute("SELECT input_hash, notes, source FROM paper_notes WHERE paper_id=?",
                                    (p.id,)).fetchone()
        if row and row[0] == h and not redo:
            cached = json.loads(row[1])
            if not cached.get("_incomplete"):        # notes with a lost piece are always redone
                p.notes, p.notes_source = cached, row[2]
                return
        pieces = chunks_for(material, self.chunk_chars) if p.fulltext else [material]
        parts, lost = [], 0
        for i, piece in enumerate(pieces, 1):
            where = f"part {i} of {len(pieces)} of the full text" if p.fulltext else "the abstract"
            got, n_lost = self._piece_notes(p, piece, where, depth=0)
            parts += got
            lost += n_lost
        p.notes, p.notes_source = merge_notes(parts), source
        if lost:
            p.notes["_incomplete"] = lost
            self.issues.append(f"[{p.id}] {lost} piece(s) of {_short(p.title)} couldn't be read; its briefing "
                               "is based on the rest, and the next run tries those pieces again")
        self.store.db.execute("INSERT OR REPLACE INTO paper_notes VALUES (?,?,?,?,?,?)",
                              (p.id, source, self.model, h, json.dumps(p.notes), utcnow()))
        self.store.commit()

    def _piece_notes(self, p: Paper, piece: str, where: str, depth: int) -> tuple[list[dict], int]:
        """Notes for one piece. A reply cut off at the length limit is redone as two smaller
        pieces; failing that, every complete item in it is kept. Returns (notes, pieces lost)."""
        system = [{"type": "text", "text": NOTES_SYSTEM}]
        user = f"Paper: {p.title}\nThis is {where}.\n\n{piece}"
        text, usage = self.backend.complete(system, user, schema=NOTES_SCHEMA, max_tokens=NOTES_MAX_TOKENS)
        self.calls += 1
        parsed = parse_json(text)
        cut_off = bool(usage.get("truncated"))
        if parsed is not None and not cut_off:
            return [parsed], 0
        if cut_off and len(piece) >= MIN_SPLIT_CHARS and depth < 2:
            self.recovered += 1
            halves = _halve(piece)
            out, lost = [], 0
            for j, half in enumerate(halves, 1):
                got, n = self._piece_notes(p, half, f"{where} (section {j} of {len(halves)})", depth + 1)
                out += got
                lost += n
            return out, lost
        salvaged = repair_json(text)
        if salvaged:
            self.recovered += 1
            return [salvaged], 0
        self._dump(p, where, text, usage)
        return [], 1

    def overview(self, topic: str, papers: list[Paper], redo: bool = False) -> str:
        """A few paragraphs covering these papers, each mentioned as "Paper N"."""
        entries = "\n\n".join(_overview_entry(p) for p in papers)
        h = _hash(OVERVIEW_VERSION, self.model, topic, entries)
        row = self.store.db.execute("SELECT text FROM overviews WHERE input_hash=?", (h,)).fetchone()
        if row and not redo:
            return row[0]
        system = [{"type": "text", "text": OVERVIEW_SYSTEM}]
        words = 40 * len(papers) + 60
        user = (f"Topic: {topic}\nWrite an overview of these {len(papers)} papers in about {words} words.\n\n"
                + entries)
        max_tokens = min(2400, 3 * words + 200)
        if not self.backend.fits(len(system[0]["text"]) + len(user), max_tokens):
            raise RuntimeError("an overview request doesn't fit the model's context window; raise ollama_num_ctx")
        text, _ = self.backend.complete(system, user, schema=None, max_tokens=max_tokens, temperature=0.3)
        self.calls += 1
        text = _clean_prose(text)
        # Every paper must be mentioned by number, or it would be invisible in the audio.
        mentioned = {int(n) for grp in re.findall(r"\bPapers?\s+(\d+(?:\s*(?:,\s*(?:and\s+)?|and|&|to)\s*\d+)*)", text)
                     for n in re.findall(r"\d+", grp)}
        missing = [p for p in papers if p.number not in mentioned]
        if missing:
            text += "\n\n" + " ".join(f"Paper {p.number}: {p.title}." for p in missing)
            self.issues.append(f"overview of {topic!r} left out {len(missing)} paper(s); their titles were added")
        self.store.db.execute("INSERT OR REPLACE INTO overviews VALUES (?,?,?,?)", (h, self.model, text, utcnow()))
        self.store.commit()
        return text

    def _dump(self, p: Paper, where: str, text: str, usage: dict) -> None:
        d = self.store.path.parent / "debug"
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\w]+", "-", where).strip("-")[:40]
        path = d / f"notes-{p.id}-{name}.txt"
        path.write_text(f"paper {p.id}: {p.title}\n{where}\ncut off at length limit: {usage.get('truncated')}\n"
                        f"---- raw reply ----\n{text}", encoding="utf-8")
        self.debug_files.append(path)

    # write-up -----------------------------------------------------------------------------
    def write(self, p: Paper, kind: str, redo: bool = False) -> None:
        instruction, max_tokens = LENGTHS[p.length]
        notes_text = notes_as_text(p.notes or {}) or "(no notes: only the title is known)"
        h = _hash(WRITE_VERSION, self.model, p.length, notes_text, p.title)
        row = self.store.db.execute("SELECT input_hash, text FROM writeups WHERE paper_id=? AND kind=?",
                                    (p.id, kind)).fetchone()
        if row and row[0] == h and not redo:
            p.text = row[1]
            return
        system = [{"type": "text", "text": WRITE_SYSTEM}]
        a = p.authors
        who = ", ".join(a[:3]) + (f", … {a[-1]}" if len(a) > 4 else (", " + a[3] if len(a) == 4 else ""))
        meta = [f"Paper: {p.title}"]
        if who:
            meta.append(f"Authors: {who}")
        venue = " ".join(str(x) for x in (p.venue, p.year) if x)
        if venue:
            meta.append(f"Published in: {venue}" + (" (preprint)" if p.is_preprint else ""))
        if p.article_type and p.article_type not in ("research", "unknown"):
            meta.append(f"Article type: {p.article_type}")
        src = {"full text": "the full text", "abstract": "the abstract only", "title only": "the title only"}
        user = ("\n".join(meta) + f"\n\n{instruction}\n\nNotes (from {src.get(p.notes_source, 'the abstract only')}):\n"
                + notes_text)
        if not self.backend.fits(len(system[0]["text"]) + len(user), max_tokens):
            raise RuntimeError("the prompt (notes) doesn't fit the model's context window; "
                               "raise ollama_num_ctx in config.yaml")
        text, _ = self.backend.complete(system, user, schema=None, max_tokens=max_tokens, temperature=0.3)
        self.calls += 1
        p.text = _clean_prose(text)
        if not p.text:
            raise RuntimeError("the model returned an empty write-up")
        self.store.db.execute("INSERT OR REPLACE INTO writeups VALUES (?,?,?,?,?,?,?)",
                              (p.id, kind, p.length, self.model, h, p.text, utcnow()))
        self.store.commit()


def _strip_reply(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    return re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()


def parse_json(text: str) -> dict | None:
    """The JSON object in a reply, or None if there isn't a complete one."""
    text = _strip_reply(text)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0) if m else text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def repair_json(text: str) -> dict | None:
    """Rescue a JSON object that was cut off: keep every complete item, close the brackets."""
    text = _strip_reply(text)
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    stack, in_str, esc = [], False, False
    cuts = []                                   # (position, open brackets there)
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                cuts.append((i + 1, list(stack)))
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, list(stack)))
        elif ch == ",":
            cuts.append((i, list(stack)))
    for pos, open_ in reversed(cuts[-400:]):
        closers = "".join("]" if c == "[" else "}" for c in reversed(open_))
        try:
            v = json.loads(s[:pos] + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict) and any(v.get(k) for k in ("question", *NOTE_LISTS)):
            return v
    return None


def _halve(piece: str) -> list[str]:
    """Split a piece in two at the paragraph boundary nearest its middle."""
    mid = len(piece) // 2
    breaks = [m.start() for m in re.finditer(r"\n\s*\n", piece)]
    cut = min(breaks, key=lambda b: abs(b - mid)) if breaks else mid
    if cut < len(piece) * 0.2 or cut > len(piece) * 0.8:
        cut = mid
    return [piece[:cut].strip(), piece[cut:].strip()]


def _short(title: str, n: int = 60) -> str:
    return title if len(title) <= n else title[: n - 1] + "…"


MAX_PIECE_CHARS = 24_000   # ~6–7k tokens: big enough for a whole section, small enough that the
                           # notes reply stays short and the request stays well inside memory


def chunk_chars_for(backend) -> int:
    """How much paper text goes into one notes request."""
    num_ctx = getattr(backend, "num_ctx", None)
    if not num_ctx:
        return MAX_PIECE_CHARS
    overhead = len(NOTES_SYSTEM) // 3 + 200
    return max(4000, min(MAX_PIECE_CHARS, (num_ctx - NOTES_MAX_TOKENS - overhead - 300) * 3))


# -- what's new since the last weekly briefing -------------------------------------------

WINDOW_KEY = "weekly_window_start"


def new_papers(store) -> dict:
    """Papers for the next weekly briefing: added since the last one (or, the first time, by the
    latest ingest) and not yet in any briefing. Also lists papers in that window that haven't
    been triaged, so nothing is passed over because classification hadn't run yet."""
    from .triage import final_labels
    since = store.get_meta(WINDOW_KEY) or store.get_meta("last_ingest_started") or ""
    briefed = {pid for (pid,) in store.db.execute("SELECT DISTINCT paper_id FROM briefing_items")}
    finals = final_labels(store)
    out = {"since": since, "mandatory": [], "interest": [], "untriaged": []}
    for pid, added in store.db.execute("SELECT id, added_at FROM papers WHERE added_at >= ?", (since,)):
        if pid in briefed:
            continue
        label = (finals.get(pid) or {}).get("label")
        if label in ("mandatory", "interest"):
            out[label].append(pid)
        elif label in ("pending_llm", "unscored"):
            out["untriaged"].append(pid)
    return out


# -- "also of interest" overviews ---------------------------------------------------------

OVERVIEW_VERSION = "1"
OVERVIEW_SYSTEM = """You write the "also of interest" part of a research briefing: a short overview of several
recent papers on one topic, written from their abstracts.

- Mention every paper given, by its number, written exactly as "Paper 12". For each, say in one or
  two sentences what it did and found. Where papers are related, discuss them together and say how
  they relate, but only when the abstracts make that clear.
- Spend every sentence on the papers' content. Do not comment on why they matter or how they relate
  to the reader's work.
- Write plain prose paragraphs. No headings, lists, tables, markdown, links or citation brackets.
  The text is read on an e-reader and also listened to as audio.
- Spell out each abbreviation the first time you use it, except very common ones (DNA, RNA, PCR).
- Use only the abstracts given. Never add findings or claims that are not in them."""
OVERVIEW_ABSTRACT_CHARS = 1200


def _overview_entry(p: Paper) -> str:
    a = p.authors
    who = (str(a[0]).split(",")[0].split()[-1] + (" and colleagues" if len(a) > 1 else "")) if a else ""
    venue = " ".join(str(x) for x in (p.venue, p.year) if x) + (" (preprint)" if p.is_preprint else "")
    text = (p.abstract or "").strip()
    if len(text) > OVERVIEW_ABSTRACT_CHARS:
        text = text[:OVERVIEW_ABSTRACT_CHARS].rsplit(" ", 1)[0] + " …"
    lines = [f"Paper {p.number}: {p.title}"]
    if who or venue:
        lines.append(" · ".join(x for x in (who, venue) if x))
    lines.append(f"Abstract: {text}" if text else "(No abstract available; only the title is known.)")
    return "\n".join(lines)


def overview_chars_for(backend) -> int:
    """How much abstract text goes into one overview request."""
    num_ctx = getattr(backend, "num_ctx", None)
    if not num_ctx:
        return 18_000
    return max(3000, min(18_000, (num_ctx - 2400 - len(OVERVIEW_SYSTEM) // 3 - 300) * 3))


def overview_batches(papers: list[Paper], max_chars: int) -> list[list[Paper]]:
    """Split a topic's papers into groups whose abstracts fit one request."""
    batches, cur, size = [], [], 0
    for p in papers:
        n = len(_overview_entry(p)) + 2
        if cur and (size + n > max_chars or len(cur) >= 15):
            batches.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += n
    if cur:
        batches.append(cur)
    return batches


# -- recording the briefing ----------------------------------------------------------------

def register(store, name: str, kind: str, sections: list[tuple[str, list[Paper]]], path: str) -> int:
    cur = store.db.execute("INSERT INTO briefings (name, kind, created_at, path) VALUES (?,?,?,?)",
                           (name, kind, utcnow(), path))
    bid = cur.lastrowid
    for _, ps in sections:
        for p in ps:
            store.db.execute("INSERT INTO briefing_items VALUES (?,?,?)", (bid, p.number, p.id))
    store.commit()
    return bid
