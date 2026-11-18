"""Stage 2a: sync your examples, then filter papers by similarity to them.

Scoring (per paper, cosine similarity of embeddings):
  pos  = mean of the k highest similarities to your positive examples (mandatory + interest)
  skip = the same, against your skip examples (if any)

A paper passes when pos >= threshold, unless it sits clearly closer to your skip examples
(skip - pos > skip_margin). The threshold is calibrated on your own positives: each positive
example is scored against the *other* positives (leave-one-out), and the threshold is set
so that `keep_recall` of them (default 95%) would pass. This stage is meant to be generous;
the LLM stage that follows is the precise one.

Labels you give papers yourself (`paperflow label`, or papers matching an example) count as
examples too, so corrections feed straight back into the filter.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from .embed import embed_all
from .examples import Example, scan
from .models import Meta
from .store import utcnow

log = logging.getLogger(__name__)

POSITIVE = ("mandatory", "interest")
TOP_K = 3


# -- examples ------------------------------------------------------------------------------

def sync_examples(store, examples_dir, resolver=None) -> tuple[dict, list[str]]:
    """Mirror the examples folders into the database and look up missing metadata.

    Returns (summary, problems)."""
    found, problems = scan(examples_dir)
    db = store.db
    keys = {e.key for e in found}
    existing = {r[0]: r for r in db.execute("SELECT key, label, source, resolved, title, abstract FROM examples")}
    for key in set(existing) - keys:  # file removed or paper moved out
        db.execute("DELETE FROM examples WHERE key=?", (key,))
        db.execute("DELETE FROM embeddings WHERE kind='example' AND ref=?", (key,))
    for e in found:
        old = existing.get(e.key)
        if old:
            db.execute("UPDATE examples SET label=?, source=?, updated_at=? WHERE key=?",
                       (e.label, e.source, utcnow(), e.key))
            # keep looked-up metadata, but let a file's own title/abstract fill gaps
            db.execute("UPDATE examples SET title=coalesce(title, ?), abstract=coalesce(abstract, ?) WHERE key=?",
                       (e.title, e.abstract, e.key))
        else:
            db.execute("INSERT INTO examples VALUES (?,?,?,?,?,?,0,?)",
                       (e.key, e.label, e.source, e.doi, e.title, e.abstract, utcnow()))

    # Fill in metadata: first from papers already in the database, then via the APIs.
    for key, doi in db.execute("SELECT key, doi FROM examples WHERE doi IS NOT NULL "
                               "AND (title IS NULL OR abstract IS NULL)").fetchall():
        row = db.execute("SELECT p.title, p.abstract FROM identifiers i JOIN papers p ON p.id=i.paper_id "
                         "WHERE i.scheme='doi' AND i.value=?", (doi,)).fetchone()
        if row:
            db.execute("UPDATE examples SET title=coalesce(title, ?), abstract=coalesce(abstract, ?), "
                       "resolved=CASE WHEN coalesce(abstract, ?) IS NOT NULL THEN 1 ELSE resolved END "
                       "WHERE key=?", (row[0], row[1], row[1], key))
    todo = db.execute("SELECT key, doi, title FROM examples WHERE abstract IS NULL AND resolved=0").fetchall()
    if todo and resolver is not None:
        from tqdm import tqdm
        for key, doi, title in tqdm(todo, desc="looking up examples", unit="paper"):
            item = {"id": -1, "title": title, "title_truncated": False, "year": None,
                    "ids": {"doi": doi} if doi else {}, "urls": [], "surnames": []}
            res = resolver.resolve(item)
            m: Meta = res.meta
            if res.status != "error":
                db.execute("UPDATE examples SET title=coalesce(?, title), abstract=?, doi=coalesce(doi, ?), "
                           "resolved=1 WHERE key=?", (m.title, m.abstract, m.doi, key))
    _link_examples_to_papers(store)
    db.commit()

    summary = {lab: n for lab, n in db.execute("SELECT label, count(*) FROM examples GROUP BY label")}
    missing = db.execute("SELECT source, doi FROM examples WHERE title IS NULL ORDER BY source").fetchall()
    if missing:
        shown = ", ".join(f"{d} ({src})" for src, d in missing[:3]) + (", …" if len(missing) > 3 else "")
        problems.append(f"{len(missing)} example(s) couldn't be looked up, so they're ignored for now "
                        f"(retried next run): {shown}")
    n_no_abs = db.execute("SELECT count(*) FROM examples WHERE title IS NOT NULL AND abstract IS NULL").fetchone()[0]
    if n_no_abs:
        problems.append(f"{n_no_abs} example(s) have no abstract; they're compared by title only")
    return summary, problems


def _link_examples_to_papers(store) -> None:
    """An example that is also an alert paper becomes your judgment of that paper."""
    db = store.db
    db.execute("DELETE FROM judgments WHERE source LIKE 'example:%'")
    rows = db.execute(
        "SELECT e.key, e.label, e.source, i.paper_id FROM examples e "
        "JOIN identifiers i ON i.scheme='doi' AND i.value=e.doi").fetchall()
    for key, label, source, pid in rows:
        db.execute("INSERT INTO judgments (paper_id, label, source, created_at) VALUES (?,?,?,?)",
                   (pid, label, f"example:{source}", utcnow()))


# -- labels you give papers ----------------------------------------------------------------

def current_judgments(store) -> dict[int, tuple[str, str]]:
    """paper_id -> (label, source) from your most recent label of each paper."""
    out = {}
    for pid, label, source in store.db.execute("SELECT paper_id, label, source FROM judgments ORDER BY id"):
        out[pid] = (label, source)
    return out


# -- the embedding filter ------------------------------------------------------------------

@dataclass
class FilterResult:
    n_scored: int
    n_pass: int
    threshold: float
    n_pos: int
    n_skip: int
    recall_est: float | None
    skip_pass_rate: float | None
    newly_passed: int
    n_vetoed: int = 0


def _paper_texts(store) -> list[tuple[int, str, str | None]]:
    """(paper_id, title, abstract-or-best-available-text) for every paper."""
    out = []
    snippets = {}
    for pid, snip in store.db.execute(
            "SELECT paper_id, snippet FROM sightings WHERE snippet IS NOT NULL ORDER BY length(snippet)"):
        snippets[pid] = snip  # longest wins
    for pid, title, abstract in store.db.execute("SELECT id, title, abstract FROM papers WHERE title IS NOT NULL"):
        out.append((pid, title, abstract or snippets.get(pid)))
    return out


def _topk_mean(sims: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest values along axis 1."""
    k = max(1, min(k, sims.shape[1]))
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def run_filter(store, backend, keep_recall: float = 0.95, skip_margin: float = 0.02,
               threshold: float | None = None, progress: bool = True) -> FilterResult:
    db = store.db
    judged = current_judgments(store)
    papers = _paper_texts(store)
    examples = [(k, lab, t, a) for k, lab, t, a in
                db.execute("SELECT key, label, title, abstract FROM examples WHERE title IS NOT NULL")]

    items = [("paper", str(pid), t, a) for pid, t, a in papers] + \
            [("example", k, t, a) for k, _, t, a in examples]
    vecs = embed_all(store, backend, items, progress=progress)

    # Reference sets: example files + papers you've labelled yourself.
    ref_vecs, ref_labels, ref_titles, ref_ids = [], [], [], []
    for k, lab, t, _ in examples:
        ref_vecs.append(vecs[("example", k)]); ref_labels.append(lab); ref_titles.append(t); ref_ids.append(k)
    paper_title = {pid: t for pid, t, _ in papers}
    for pid, (lab, src) in judged.items():
        if src.startswith("example:") or ("paper", str(pid)) not in vecs:
            continue  # example-linked papers are already represented by the example itself
        ref_vecs.append(vecs[("paper", str(pid))]); ref_labels.append(lab)
        ref_titles.append(paper_title.get(pid)); ref_ids.append(f"paper:{pid}")
    ref_labels_arr = np.array(ref_labels)
    pos_mask = np.isin(ref_labels_arr, POSITIVE)
    skip_mask = ref_labels_arr == "skip"
    n_pos, n_skip = int(pos_mask.sum()), int(skip_mask.sum())
    if n_pos < 2:
        raise SystemExit("Need at least 2 positive examples (mandatory or interest) to filter. "
                         "Add some to the examples folders, then run `paperflow examples`.")
    R = np.vstack(ref_vecs)
    P, S = R[pos_mask], R[skip_mask] if n_skip else None

    # Calibrate on the positives themselves (leave-one-out).
    pp = P @ P.T
    np.fill_diagonal(pp, -np.inf)
    loo = _topk_mean(pp, TOP_K)
    if threshold is None:
        threshold = float(np.quantile(loo, 1 - keep_recall))
    recall_est = float((loo >= threshold).mean())
    skip_pass_rate = None
    if S is not None:
        sp = _topk_mean(S @ P.T, TOP_K)
        ss = S @ S.T
        np.fill_diagonal(ss, -np.inf)
        s_skip = _topk_mean(ss, TOP_K) if n_skip > 1 else np.full(n_skip, -np.inf)
        skip_pass_rate = float(((sp >= threshold) & ~(s_skip - sp > skip_margin)).mean())

    # Score every paper.
    pids = [pid for pid, _, _ in papers]
    X = np.vstack([vecs[("paper", str(pid))] for pid in pids])
    sims = X @ R.T
    pos = _topk_mean(sims[:, pos_mask], TOP_K)
    skip = _topk_mean(sims[:, skip_mask], TOP_K) if S is not None else np.full(len(pids), -np.inf)

    previously = {pid for (pid,) in db.execute("SELECT paper_id FROM assessments WHERE stage='embed' AND label='pass'")}
    n_pass = newly = n_vetoed = 0
    now = utcnow()
    for row, pid in enumerate(pids):
        near = np.argsort(-sims[row])[:3]
        nearest = [{"label": ref_labels[j], "title": ref_titles[j], "sim": round(float(sims[row, j]), 3)}
                   for j in near]
        pos_near = [j for j in np.argsort(-sims[row]) if pos_mask[j]][:1]
        hint = ref_labels[pos_near[0]] if pos_near else None
        vetoed = S is not None and (skip[row] - pos[row] > skip_margin)
        passed = bool(pos[row] >= threshold and not vetoed)
        detail = {"pos": round(float(pos[row]), 4),
                  "skip": round(float(skip[row]), 4) if S is not None else None,
                  "threshold": round(threshold, 4), "nearest": nearest, "hint": hint,
                  "vetoed_by_skip": bool(vetoed and pos[row] >= threshold), "model": backend.name}
        db.execute("INSERT OR REPLACE INTO assessments VALUES (?,?,?,?,?,?)",
                   (pid, "embed", "pass" if passed else "drop", float(pos[row]), json.dumps(detail), now))
        if passed:
            n_pass += 1
            newly += pid not in previously
        n_vetoed += detail["vetoed_by_skip"]
    db.commit()
    return FilterResult(len(pids), n_pass, threshold, n_pos, n_skip, recall_est, skip_pass_rate, newly, n_vetoed)


# -- combined view -------------------------------------------------------------------------

FINAL_ORDER = ("mandatory", "interest", "skip", "filtered", "pending_llm", "unscored")


def final_labels(store) -> dict[int, dict]:
    """paper_id -> {"label", "by", "embed", "llm", "judgment"}.

    Your own label wins, then the LLM's, then the embedding filter's verdict."""
    db = store.db
    a = {}
    for pid, stage, label, score, detail in db.execute("SELECT paper_id, stage, label, score, detail FROM assessments"):
        a.setdefault(pid, {})[stage] = {"label": label, "score": score, "detail": json.loads(detail or "{}")}
    judged = current_judgments(store)
    out = {}
    for (pid,) in db.execute("SELECT id FROM papers"):
        st = a.get(pid, {})
        j = judged.get(pid)
        if j:
            label, by = j[0], "you"
        elif "llm" in st:
            label, by = st["llm"]["label"], "llm"
        elif st.get("embed", {}).get("label") == "drop":
            label, by = "filtered", "embedding"
        elif st.get("embed", {}).get("label") == "pass":
            label, by = "pending_llm", "embedding"
        else:
            label, by = "unscored", None
        out[pid] = {"label": label, "by": by, "embed": st.get("embed"), "llm": st.get("llm"),
                    "judgment": {"label": j[0], "source": j[1]} if j else None}
    return out


def add_judgment(store, paper_id: int, label: str, note: str | None = None) -> None:
    store.db.execute("INSERT INTO judgments (paper_id, label, source, note, created_at) VALUES (?,?,?,?,?)",
                     (paper_id, label, "manual", note, utcnow()))
    store.db.commit()
