"""Your reading state of each paper, and the interactive review pass.

States (history is kept; the latest one counts):
  new       you don't recognise it: gets a full briefing
  familiar  you know the gist: gets a short refresher
  read      you've read it properly: gets a one-line reminder
  saved     you added it to Zotero / want to read it: full briefing, flagged as queued
Papers with no recorded state are "unmarked" (treated like new).

The reading state is separate from the triage label (mandatory / interest / skip): one says how
well you know a paper, the other how relevant it is.
"""
from __future__ import annotations

import json
import sys
import textwrap

from .store import utcnow

STATES = ("new", "familiar", "read", "saved")


def current_states(store) -> dict[int, str]:
    out = {}
    for pid, state in store.db.execute("SELECT paper_id, state FROM reading ORDER BY id"):
        out[pid] = state
    return out


def set_state(store, paper_id: int, state: str, note: str | None = None) -> None:
    if state not in STATES:
        raise ValueError(state)
    store.db.execute("INSERT INTO reading (paper_id, state, note, created_at) VALUES (?,?,?,?)",
                     (paper_id, state, note, utcnow()))


def history(store, paper_id: int) -> list[tuple[str, str]]:
    return [(s, t) for s, t in store.db.execute(
        "SELECT state, created_at FROM reading WHERE paper_id=? ORDER BY id", (paper_id,))]


# -- choosing papers -----------------------------------------------------------------------

def briefing_papers(store, which: str) -> tuple[str, list[int]]:
    """Papers of a briefing in announced order. `which` is 'latest', a briefing id or a name."""
    db = store.db
    if which == "latest":
        row = db.execute("SELECT id, name FROM briefings ORDER BY id DESC LIMIT 1").fetchone()
    elif which.isdigit():
        row = db.execute("SELECT id, name FROM briefings WHERE id=?", (int(which),)).fetchone()
    else:
        row = db.execute("SELECT id, name FROM briefings WHERE name=? ORDER BY id DESC LIMIT 1", (which,)).fetchone()
    if not row:
        raise SystemExit("No briefings yet." if which == "latest" else f"No briefing {which!r}.")
    pids = [r[0] for r in db.execute("SELECT paper_id FROM briefing_items WHERE briefing_id=? ORDER BY position",
                                     (row[0],))]
    return row[1] or f"briefing {row[0]}", pids


def select_papers(store, labels: list[str] | None, states: list[str] | None, new_batch: bool,
                  include_marked: bool) -> list[int]:
    """Papers matching the triage labels / reading states, grouped by project, newest first."""
    from .triage import final_labels
    finals = final_labels(store)
    current = current_states(store)
    added_since = store.get_meta("last_ingest_started") if new_batch else None
    rows = store.db.execute("SELECT id, year, added_at FROM papers").fetchall()
    picked = []
    for pid, year, added_at in rows:
        f = finals.get(pid) or {}
        if labels and f.get("label") not in labels:
            continue
        st = current.get(pid, "unmarked")
        if states:
            if st not in states:
                continue
        elif not include_marked and pid in current:
            continue  # the default: only papers you haven't marked yet, so a pass can be resumed
        if added_since and (added_at or "") < added_since:
            continue
        projects = ((f.get("llm") or {}).get("detail") or {}).get("projects") or []
        picked.append((projects[0] if projects else "~", -(year or 0), pid))
    return [pid for _, _, pid in sorted(picked)]


# -- one keypress at a time ----------------------------------------------------------------

def getch() -> str:
    """Read one keypress without waiting for Enter (falls back to a line of input)."""
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    except (ImportError, OSError, ValueError):
        line = input("> ")
        return line[:1] if line else "\r"


# -- the review screen ---------------------------------------------------------------------

KEYS = {"n": "new", "f": "familiar", "r": "read", "s": "saved"}
RELABEL = {"m": "mandatory", "i": "interest", "x": "skip"}
TYPE_KEYS = {"1": "research", "2": "review", "3": "commentary", "4": "editorial",
             "5": "correction", "6": "retraction", "7": "other"}
HELP = ("  [n]ew  [f]amiliar  [r]ead  [s]aved  ·  relabel: [m]andatory [i]nterest [x]skip  ·  "
        "[t]ype  [a]bstract  [Enter] leave as is  [b]ack  [q]uit")


def _card(store, pid: int, pos: int, total: int, finals: dict, states: dict, number: int | None,
          show_abstract: bool, width: int = 100) -> str:
    p = store.db.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
    f = finals.get(pid) or {}
    llm = (f.get("llm") or {}).get("detail") or {}
    authors = json.loads(p["authors"]) if p["authors"] else []
    if len(authors) > 4:
        authors = authors[:3] + ["…", authors[-1]]
    head = f"[{pos}/{total}]"
    if number is not None:
        head += f"  Paper {number}"
    projects = ", ".join(llm.get("projects") or [])
    head += "  " + " · ".join(x for x in (projects, p["doi"], f"id {pid}") if x)
    kind = p["article_type"] if p["article_type"] not in (None, "unknown") else ""
    if p["is_preprint"]:
        kind = (kind + " preprint").strip()
    meta = " · ".join(x for x in (", ".join(authors), " ".join(str(v) for v in (p["venue"], p["year"]) if v), kind) if x)
    lines = [head, textwrap.fill(p["title"] or "(no title)", width), textwrap.fill(meta, width)]
    label = f.get("label")
    lines.append(f"Triage: {label} (by {f.get('by') or '-'})   Your state: {states.get(pid, 'unmarked')}")
    if llm.get("reason"):
        lines.append(textwrap.fill("Why: " + llm["reason"], width))
    abstract = p["abstract"] or ""
    if abstract:
        if show_abstract:
            lines.append(textwrap.fill(abstract, width))
        else:
            lines.append(textwrap.fill(abstract[:280] + ("…" if len(abstract) > 280 else ""), width))
    return "\n".join(lines)


def review_loop(store, pids: list[int], numbers: dict[int, int] | None = None,
                read_key=None, out=print) -> dict:
    """Walk through papers, one keypress each. Returns counts of what changed."""
    read_key = read_key or (lambda: getch())
    from .triage import add_judgment, final_labels
    finals = final_labels(store)
    states = current_states(store)
    changed = {"state": 0, "label": 0, "type": 0}
    i, show_abstract = 0, False
    while 0 <= i < len(pids):
        pid = pids[i]
        out("\n" + _card(store, pid, i + 1, len(pids), finals, states,
                         (numbers or {}).get(pid), show_abstract) + "\n" + HELP)
        key = read_key().lower()
        if key == "q":
            break
        if key == "a":
            show_abstract = not show_abstract
            continue
        show_abstract = False
        if key == "b":
            i = max(0, i - 1)
            continue
        if key in KEYS:
            set_state(store, pid, KEYS[key])
            states[pid] = KEYS[key]
            changed["state"] += 1
        elif key in RELABEL:
            add_judgment(store, pid, RELABEL[key], note="review")
            finals = final_labels(store)
            changed["label"] += 1
            out(f"  → labelled {RELABEL[key]}; now choose a reading state (or Enter to move on)")
            continue  # stay on the paper so a state can be set too
        elif key == "t":
            out("  type: " + "  ".join(f"[{k}] {v}" for k, v in TYPE_KEYS.items()))
            t = TYPE_KEYS.get(read_key())
            if t:
                store.set_article_type(pid, t, "you", None)
                changed["type"] += 1
            continue
        elif key not in ("\r", "\n", " "):
            out("  (unknown key)")
            continue
        store.commit()
        i += 1
    store.commit()
    return changed
