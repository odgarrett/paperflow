"""SQLite storage.

Tables
  emails       one row per ingested alert email (raw message kept, compressed, for re-parsing)
  papers       one row per distinct paper
  identifiers  (scheme, value) -> paper; every DOI/PMID/PMCID/arXiv id is owned by one paper
  sightings    every appearance of a paper in an email (which alert, when, snippet)

Deduplication: a paper is identified by any of its identifiers. When a new identifier turns
out to belong to another paper (e.g. a Scholar title-only entry resolves to a DOI already
seen in a bioRxiv alert, or a preprint's published DOI), the two rows are merged.
Title-only entries are matched on normalized titles, including Scholar's truncated ones.
"""
from __future__ import annotations

import json
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

from .models import ParsedEmail, Sighting
from .resolver import Resolution
from .links import is_preprint_doi
from .text import surnames_from_list, title_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    message_id   TEXT PRIMARY KEY,
    source       TEXT,
    parser       TEXT,
    sender       TEXT,
    subject      TEXT,
    alert_name   TEXT,
    alert_kind   TEXT,
    received_at  TEXT,
    ingested_at  TEXT,
    n_sightings  INTEGER,
    n_reported   INTEGER,
    raw          BLOB
);
CREATE TABLE IF NOT EXISTS papers (
    id              INTEGER PRIMARY KEY,
    title           TEXT,
    title_key       TEXT,
    title_truncated INTEGER DEFAULT 0,
    authors         TEXT,      -- JSON list
    year            INTEGER,
    pub_date        TEXT,
    venue           TEXT,
    doi             TEXT,
    published_doi   TEXT,
    is_preprint     INTEGER,
    abstract        TEXT,
    abstract_source TEXT,
    oa_pdf_url      TEXT,
    landing_url     TEXT,
    work_type       TEXT,
    meta_sources    TEXT,      -- JSON list
    status          TEXT DEFAULT 'pending',
    status_detail   TEXT,
    first_seen      TEXT,
    last_seen       TEXT,
    resolved_at     TEXT,
    added_at        TEXT,      -- when the paper first entered the database
    article_type    TEXT,      -- research | review | commentary | editorial | correction | retraction | other | unknown
    article_type_source TEXT,  -- what decided it: europepmc | openalex | semanticscholar | crossref | title | journal | …
    related         TEXT       -- JSON: PubMed links to other papers (comment on, erratum in, retraction in…)
);
CREATE INDEX IF NOT EXISTS papers_title_key ON papers(title_key);
CREATE INDEX IF NOT EXISTS papers_status ON papers(status);
CREATE TABLE IF NOT EXISTS identifiers (
    scheme   TEXT NOT NULL,
    value    TEXT NOT NULL,
    paper_id INTEGER NOT NULL,
    PRIMARY KEY (scheme, value)
);
CREATE INDEX IF NOT EXISTS identifiers_paper ON identifiers(paper_id);
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY,
    paper_id    INTEGER NOT NULL,
    message_id  TEXT NOT NULL,
    source      TEXT,
    alert_name  TEXT,
    alert_kind  TEXT,
    raw_title   TEXT,
    raw_authors TEXT,
    raw_venue   TEXT,
    raw_url     TEXT,
    raw_ids     TEXT,
    snippet     TEXT,
    seen_at     TEXT
);
CREATE INDEX IF NOT EXISTS sightings_paper ON sightings(paper_id);
CREATE INDEX IF NOT EXISTS sightings_message ON sightings(message_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- Triage (stage 2) ------------------------------------------------------------------------
-- Your example papers, synced from the examples/ folders.
CREATE TABLE IF NOT EXISTS examples (
    key        TEXT PRIMARY KEY,   -- doi:… or title:…
    label      TEXT NOT NULL,      -- mandatory | interest | skip
    source     TEXT,               -- file it came from
    doi        TEXT,
    title      TEXT,
    abstract   TEXT,
    resolved   INTEGER DEFAULT 0,  -- metadata lookup attempted
    updated_at TEXT
);
-- Cached embedding vectors (float32), per model.
CREATE TABLE IF NOT EXISTS embeddings (
    kind      TEXT NOT NULL,       -- paper | example
    ref       TEXT NOT NULL,       -- paper id or example key
    model     TEXT NOT NULL,
    text_hash TEXT,
    vec       BLOB,
    PRIMARY KEY (kind, ref, model)
);
-- Latest automatic assessment of each paper at each stage.
CREATE TABLE IF NOT EXISTS assessments (
    paper_id   INTEGER NOT NULL,
    stage      TEXT NOT NULL,      -- embed (pass | drop) | llm (mandatory | interest | skip)
    label      TEXT,
    score      REAL,
    detail     TEXT,               -- JSON: nearest examples, reason, projects, model…
    created_at TEXT,
    PRIMARY KEY (paper_id, stage)
);
-- Your own labels (from the examples folders or `paperflow label`). These always win.
CREATE TABLE IF NOT EXISTS judgments (
    id         INTEGER PRIMARY KEY,
    paper_id   INTEGER NOT NULL,
    label      TEXT NOT NULL,
    source     TEXT,               -- example:<file> | manual
    note       TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS judgments_paper ON judgments(paper_id);

-- Your reading state of each paper, as history (latest row wins): new | familiar | read | saved
CREATE TABLE IF NOT EXISTS reading (
    id         INTEGER PRIMARY KEY,
    paper_id   INTEGER NOT NULL,
    state      TEXT NOT NULL,
    note       TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS reading_paper ON reading(paper_id);
-- Briefings and the order their papers appear in (papers are announced as 1, 2, 3…)
CREATE TABLE IF NOT EXISTS briefings (
    id         INTEGER PRIMARY KEY,
    name       TEXT,
    kind       TEXT,               -- backlog-mandatory | weekly | …
    created_at TEXT,
    path       TEXT
);
CREATE TABLE IF NOT EXISTS briefing_items (
    briefing_id INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    paper_id    INTEGER NOT NULL,
    PRIMARY KEY (briefing_id, position)
);
-- Briefing material, reused until its inputs change
CREATE TABLE IF NOT EXISTS paper_notes (
    paper_id   INTEGER PRIMARY KEY,
    source     TEXT,               -- full text | abstract | title only
    model      TEXT,
    input_hash TEXT,
    notes      TEXT,               -- JSON: question, systems, approach, results, limitations, key_figures
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS writeups (
    paper_id     INTEGER NOT NULL,
    kind         TEXT NOT NULL,    -- which kind of briefing it was written for
    length_class TEXT,             -- full | refresher | reminder
    model        TEXT,
    input_hash   TEXT,
    text         TEXT,
    created_at   TEXT,
    PRIMARY KEY (paper_id, kind)
);
-- "Also of interest" overviews, reused while the same papers are covered
CREATE TABLE IF NOT EXISTS overviews (
    input_hash TEXT PRIMARY KEY,
    model      TEXT,
    text       TEXT,
    created_at TEXT
);
-- Saved papers you've added to Zotero (via `paperflow zotero`)
CREATE TABLE IF NOT EXISTS zotero_sent (
    paper_id INTEGER PRIMARY KEY,
    sent_at  TEXT
);
-- Full texts retrieved for papers (the text itself lives in data/fulltext/<paper_id>.txt)
CREATE TABLE IF NOT EXISTS fulltexts (
    paper_id   INTEGER PRIMARY KEY,
    status     TEXT NOT NULL,      -- ok | unavailable | error (retried next run)
    source     TEXT,               -- your pdf | europepmc | biorxiv | unpaywall | openalex
    url        TEXT,
    chars      INTEGER,
    detail     TEXT,               -- why not, when unavailable
    fetched_at TEXT
);
"""

META_COLS = ("title", "title_key", "title_truncated", "authors", "year", "pub_date", "venue", "doi",
             "published_doi", "is_preprint", "abstract", "abstract_source", "oa_pdf_url",
             "landing_url", "work_type", "meta_sources", "article_type", "article_type_source", "related")
STATUS_RANK = {"resolved": 3, "no_abstract": 2, "unmatched": 1, "error": 1, "pending": 0}
MIN_TITLE_KEY = 15  # don't title-match very short titles


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(papers)")}
        for col in ("added_at", "article_type", "article_type_source", "related"):
            if col not in cols:  # databases created by earlier versions
                self.db.execute(f"ALTER TABLE papers ADD COLUMN {col} TEXT")
        self.merges = 0

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def get_meta(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def _paper(self, pid: int):
        return self.db.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()

    def _owner(self, scheme: str, value: str) -> int | None:
        r = self.db.execute("SELECT paper_id FROM identifiers WHERE scheme=? AND value=?",
                            (scheme, value)).fetchone()
        return r[0] if r else None

    # -- emails ----------------------------------------------------------------------------

    def known_message_ids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT message_id FROM emails")}

    def add_email(self, parsed: ParsedEmail, raw: bytes) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (parsed.message_id, parsed.source, parsed.parser, parsed.sender, parsed.subject,
             parsed.alert_name, parsed.alert_kind, parsed.received_at, utcnow(),
             len(parsed.sightings), parsed.n_reported, zlib.compress(raw)),
        )
        for s in parsed.sightings:
            self._add_sighting(parsed, s)

    def iter_raw_emails(self):
        for r in self.db.execute("SELECT message_id, raw FROM emails WHERE raw IS NOT NULL"):
            yield r["message_id"], zlib.decompress(r["raw"])

    def clear_sightings(self, message_id: str) -> None:
        self.db.execute("DELETE FROM sightings WHERE message_id=?", (message_id,))

    # -- sightings and matching ------------------------------------------------------------

    def _add_sighting(self, parsed: ParsedEmail, s: Sighting) -> None:
        seen = s.posted or parsed.received_at  # may be None (e.g. Outlook CSV)
        pid = self._match_or_create(s, seen)
        self.db.execute(
            "INSERT INTO sightings (paper_id, message_id, source, alert_name, alert_kind, raw_title,"
            " raw_authors, raw_venue, raw_url, raw_ids, snippet, seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, parsed.message_id, s.source, s.alert or parsed.alert_name,
             s.alert_kind or parsed.alert_kind, s.title,
             s.authors_raw, s.venue_raw, s.url, json.dumps(s.ids) if s.ids else None, s.snippet, seen),
        )
        p = self._paper(pid)
        updates = {
            "first_seen": min(filter(None, [p["first_seen"], seen]), default=None),
            "last_seen": max(filter(None, [p["last_seen"], seen]), default=None),
        }
        # A full title from another alert beats Scholar's truncated one (until resolved).
        if p["title_truncated"] and s.title and not s.title_truncated and p["status"] != "resolved":
            updates.update(title=s.title, title_key=title_key(s.title), title_truncated=0)
        if p["year"] is None and s.year:
            updates["year"] = s.year
        self._update(pid, updates)

    def _match_or_create(self, s: Sighting, seen: str) -> int:
        owners = {o for o in (self._owner(k, v) for k, v in s.ids.items()) if o is not None}
        pid = None
        if owners:
            pid = min(owners)
            for other in owners - {pid}:
                self.merge(pid, other)
        elif s.title:
            pid = self._match_title(title_key(s.title), s.title_truncated, s.ids.get("doi"))
        if pid is None:
            cur = self.db.execute(
                "INSERT INTO papers (title, title_key, title_truncated, year, venue, doi, status,"
                " first_seen, last_seen, added_at) VALUES (?,?,?,?,?,?,'pending',?,?,?)",
                (s.title, title_key(s.title), int(s.title_truncated), s.year, s.venue_raw,
                 s.ids.get("doi"), seen, seen, utcnow()),
            )
            pid = cur.lastrowid
        for scheme, value in s.ids.items():
            self._claim(pid, scheme, value)
        return pid

    def _match_title(self, key: str, truncated: bool, doi: str | None) -> int | None:
        if len(key) < MIN_TITLE_KEY:
            return None
        rows = list(self.db.execute("SELECT id, doi FROM papers WHERE title_key=?", (key,)))
        if truncated:
            rows += self.db.execute("SELECT id, doi FROM papers WHERE title_key LIKE ?", (key + "%",))
        rows += self.db.execute(
            "SELECT id, doi FROM papers WHERE title_truncated=1 AND length(title_key)>=? "
            "AND ? LIKE title_key || '%'", (MIN_TITLE_KEY, key))
        for r in rows:
            if doi is None or r["doi"] is None or r["doi"] == doi:
                return r["id"]
        return None

    def _claim(self, pid: int, scheme: str, value: str) -> int:
        """Attach an identifier to a paper, merging if another paper already owns it."""
        owner = self._owner(scheme, value)
        if owner is None:
            self.db.execute("INSERT INTO identifiers VALUES (?,?,?)", (scheme, value, pid))
        elif owner != pid:
            self.merge(pid, owner)
        if scheme == "doi":
            self.db.execute("UPDATE papers SET doi=? WHERE id=? AND doi IS NULL", (value, pid))
        return pid

    def _update(self, pid: int, values: dict) -> None:
        if values:
            cols = ", ".join(f"{k}=?" for k in values)
            self.db.execute(f"UPDATE papers SET {cols} WHERE id=?", (*values.values(), pid))

    @property
    def fulltext_dir(self) -> Path:
        return self.path.parent / "fulltext"

    def _merge_fulltext(self, keep: int, drop: int) -> None:
        """Keep whichever paper has a full text (the kept paper's if both do)."""
        k = self.db.execute("SELECT status FROM fulltexts WHERE paper_id=?", (keep,)).fetchone()
        d = self.db.execute("SELECT status FROM fulltexts WHERE paper_id=?", (drop,)).fetchone()
        drop_file = self.fulltext_dir / f"{drop}.txt"
        if d and d[0] == "ok" and not (k and k[0] == "ok"):
            self.db.execute("DELETE FROM fulltexts WHERE paper_id=?", (keep,))
            self.db.execute("UPDATE fulltexts SET paper_id=? WHERE paper_id=?", (keep, drop))
            if drop_file.exists():
                drop_file.replace(self.fulltext_dir / f"{keep}.txt")
        else:
            self.db.execute("DELETE FROM fulltexts WHERE paper_id=?", (drop,))
            drop_file.unlink(missing_ok=True)

    def merge(self, keep: int, drop: int) -> None:
        """Fold paper `drop` into `keep`. The better-resolved record's metadata wins."""
        if keep == drop:
            return
        k, d = self._paper(keep), self._paper(drop)
        if k is None or d is None:
            return
        updates = {}
        drop_better = STATUS_RANK.get(d["status"], 0) > STATUS_RANK.get(k["status"], 0)
        for col in META_COLS:
            if d[col] is not None and (k[col] is None or drop_better):
                updates[col] = d[col]
        if drop_better:
            updates.update(status=d["status"], status_detail=d["status_detail"],
                           resolved_at=d["resolved_at"])
        updates["first_seen"] = min(filter(None, [k["first_seen"], d["first_seen"]]), default=None)
        updates["last_seen"] = max(filter(None, [k["last_seen"], d["last_seen"]]), default=None)
        updates["added_at"] = min(filter(None, [k["added_at"], d["added_at"]]), default=None)
        self._update(keep, updates)
        self.db.execute("UPDATE sightings SET paper_id=? WHERE paper_id=?", (keep, drop))
        self.db.execute("UPDATE identifiers SET paper_id=? WHERE paper_id=?", (keep, drop))
        self.db.execute("UPDATE judgments SET paper_id=? WHERE paper_id=?", (keep, drop))
        self.db.execute("UPDATE reading SET paper_id=? WHERE paper_id=?", (keep, drop))
        self.db.execute("UPDATE briefing_items SET paper_id=? WHERE paper_id=?", (keep, drop))
        self._merge_fulltext(keep, drop)
        self.db.execute("UPDATE OR IGNORE zotero_sent SET paper_id=? WHERE paper_id=?", (keep, drop))
        self.db.execute("DELETE FROM zotero_sent WHERE paper_id=?", (drop,))
        for table in ("paper_notes", "writeups"):  # regenerated from the merged paper's material
            self.db.execute(f"DELETE FROM {table} WHERE paper_id IN (?, ?)", (keep, drop))
        # Keep the more advanced automatic assessment for each stage.
        self.db.execute("INSERT OR IGNORE INTO assessments SELECT ?, stage, label, score, detail, created_at "
                        "FROM assessments WHERE paper_id=?", (keep, drop))
        self.db.execute("DELETE FROM assessments WHERE paper_id=?", (drop,))
        self.db.execute("DELETE FROM embeddings WHERE kind='paper' AND ref IN (?, ?)", (str(keep), str(drop)))
        self.db.execute("DELETE FROM papers WHERE id=?", (drop,))
        self.merges += 1

    # -- resolution ------------------------------------------------------------------------

    def pending_items(self, statuses=("pending",), limit: int | None = None) -> list[dict]:
        q = f"SELECT * FROM papers WHERE status IN ({','.join('?' * len(statuses))}) ORDER BY id"
        if limit:
            q += f" LIMIT {int(limit)}"
        items = []
        for p in self.db.execute(q, tuple(statuses)).fetchall():
            ids: dict[str, str] = {}
            for r in self.db.execute("SELECT scheme, value FROM identifiers WHERE paper_id=?", (p["id"],)):
                ids.setdefault(r["scheme"], r["value"])
            if p["doi"]:
                ids["doi"] = p["doi"]
            sights = self.db.execute(
                "SELECT raw_url, raw_authors, raw_title FROM sightings WHERE paper_id=?", (p["id"],)
            ).fetchall()
            urls, surn = [], []
            for s in sights:
                if s["raw_url"] and s["raw_url"] not in urls:
                    urls.append(s["raw_url"])
                for n in surnames_from_list(s["raw_authors"]):
                    if n not in surn:
                        surn.append(n)
            items.append({
                "id": p["id"], "title": p["title"], "title_truncated": bool(p["title_truncated"]),
                "year": p["year"], "ids": ids, "urls": urls, "surnames": surn,
            })
        return items

    def apply_resolution(self, res: Resolution) -> None:
        pid = res.paper_id
        if self._paper(pid) is None:  # merged away while its lookup was in flight
            return
        m = res.meta
        if res.status == "error" and not m.title:
            self._update(pid, {"status": "error", "status_detail": res.detail, "resolved_at": utcnow()})
            return
        values = {"status": res.status, "status_detail": res.detail, "resolved_at": utcnow()}
        if m.title:
            values.update(title=m.title, title_key=title_key(m.title), title_truncated=0)
        for col, val in (
            ("authors", json.dumps(m.authors) if m.authors else None),
            ("year", m.year), ("pub_date", m.pub_date), ("venue", m.venue), ("doi", m.doi),
            ("published_doi", m.published_doi),
            ("is_preprint", int(m.is_preprint) if m.is_preprint is not None else None),
            ("abstract", m.abstract), ("abstract_source", m.abstract_source),
            ("oa_pdf_url", m.oa_pdf_url), ("landing_url", m.landing_url),
            ("work_type", m.work_type), ("meta_sources", json.dumps(m.sources) if m.sources else None),
            ("article_type", m.article_type), ("article_type_source", m.article_type_source),
            ("related", json.dumps(m.related) if m.related else None),
        ):
            if val is not None:
                values[col] = val
        self._update(pid, values)
        for scheme, value in m.identifiers():
            self._claim(pid, scheme, value)

    def set_article_type(self, pid: int, article_type: str, source: str, related: list | None) -> None:
        self.db.execute("UPDATE papers SET article_type=?, article_type_source=?, "
                        "related=coalesce(?, related) WHERE id=?",
                        (article_type, source, json.dumps(related) if related else None, pid))

    # -- housekeeping ----------------------------------------------------------------------

    def dedupe_titles(self) -> int:
        """Merge papers whose normalized titles match (incl. truncated prefixes), unless their DOIs conflict."""
        before = self.merges
        groups = self.db.execute(
            "SELECT title_key FROM papers WHERE length(title_key)>=? GROUP BY title_key HAVING count(*)>1",
            (MIN_TITLE_KEY,)).fetchall()
        for g in groups:
            rows = self.db.execute("SELECT id, doi, status FROM papers WHERE title_key=?", (g[0],)).fetchall()
            dois = {r["doi"] for r in rows if r["doi"]}
            published = {d for d in dois if not is_preprint_doi(d)}
            if len(dois) > 1:
                # Different DOIs with an identical (long) title: merge only a preprint with its
                # published version, never two distinct published works.
                if len(published) > 1 or len(g[0]) < 30:
                    continue
            rows = sorted(rows, key=lambda r: (-STATUS_RANK.get(r["status"], 0),
                                               r["doi"] is None or is_preprint_doi(r["doi"]), r["id"]))
            keep = rows[0]["id"]
            for r in rows[1:]:
                self.merge(keep, r["id"])
            if published and len(dois) > 1:
                pub = next(iter(published))
                self.db.execute("UPDATE papers SET published_doi=? WHERE id=? AND doi<>?", (pub, keep, pub))

        for t in self.db.execute(
                "SELECT id, title_key, doi FROM papers WHERE title_truncated=1 AND length(title_key)>=?",
                (MIN_TITLE_KEY,)).fetchall():
            cands = self.db.execute(
                "SELECT id, doi FROM papers WHERE id<>? AND title_truncated=0 AND title_key LIKE ?",
                (t["id"], t["title_key"] + "%")).fetchall()
            if len(cands) == 1 and (t["doi"] is None or cands[0]["doi"] in (None, t["doi"])):
                self.merge(cands[0]["id"], t["id"])
        return self.merges - before

    def delete_orphans(self) -> int:
        orphan_ids = [r[0] for r in self.db.execute(
            "SELECT id FROM papers WHERE id NOT IN (SELECT DISTINCT paper_id FROM sightings)")]
        for pid in orphan_ids:
            for table in ("identifiers", "assessments", "judgments", "reading", "briefing_items", "fulltexts",
                          "paper_notes", "writeups", "zotero_sent"):
                self.db.execute(f"DELETE FROM {table} WHERE paper_id=?", (pid,))
            self.db.execute("DELETE FROM embeddings WHERE kind='paper' AND ref=?", (str(pid),))
            self.db.execute("DELETE FROM papers WHERE id=?", (pid,))
        return len(orphan_ids)

    # -- reporting -------------------------------------------------------------------------

    def stats(self) -> dict:
        q = lambda sql, *a: self.db.execute(sql, a).fetchall()
        return {
            "emails_by_source": {r[0]: r[1] for r in q("SELECT source, count(*) FROM emails GROUP BY source")},
            "emails_with_no_papers": q("SELECT count(*) FROM emails WHERE n_sightings=0")[0][0],
            "emails_by_parser": {r[0]: r[1] for r in q("SELECT parser, count(*) FROM emails GROUP BY parser")},
            "reported_but_not_shown": q(
                "SELECT coalesce(sum(n_reported - n_sightings), 0) FROM emails WHERE n_reported > n_sightings")[0][0],
            "sightings": q("SELECT count(*) FROM sightings")[0][0],
            "papers": q("SELECT count(*) FROM papers")[0][0],
            "papers_by_status": {r[0]: r[1] for r in q("SELECT status, count(*) FROM papers GROUP BY status")},
            "papers_by_source_combo": {r[0]: r[1] for r in q(
                "SELECT combo, count(*) FROM (SELECT paper_id, group_concat(DISTINCT source) combo FROM "
                "(SELECT paper_id, source FROM sightings ORDER BY source) GROUP BY paper_id) GROUP BY combo "
                "ORDER BY count(*) DESC")},
            "papers_seen_in_multiple_alerts": q(
                "SELECT count(*) FROM (SELECT paper_id FROM sightings GROUP BY paper_id "
                "HAVING count(DISTINCT message_id)>1)")[0][0],
            "article_types": {(r[0] or "not yet determined"): r[1] for r in q(
                "SELECT article_type, count(*) FROM papers GROUP BY article_type ORDER BY count(*) DESC")},
            "abstract_sources": {r[0]: r[1] for r in q(
                "SELECT abstract_source, count(*) FROM papers WHERE abstract_source IS NOT NULL "
                "GROUP BY abstract_source")},
            "top_alerts": [(r[0], r[1], r[2]) for r in q(
                "SELECT source, alert_name, count(DISTINCT paper_id) n FROM sightings "
                "GROUP BY source, alert_name ORDER BY n DESC LIMIT 15")],
        }

    def export_rows(self, statuses: list[str] | None = None, since: str | None = None,
                    added_since: str | None = None):
        q, args = "SELECT * FROM papers WHERE 1=1", []
        if statuses:
            q += f" AND status IN ({','.join('?' * len(statuses))})"
            args += statuses
        if since:
            q += " AND last_seen >= ?"
            args.append(since)
        if added_since:
            q += " AND added_at >= ?"
            args.append(added_since)
        for p in self.db.execute(q + " ORDER BY last_seen DESC", args).fetchall():
            rec = dict(p)
            rec.pop("title_key", None)
            rec.pop("title_truncated", None)
            for col in ("authors", "meta_sources", "related"):
                rec[col] = json.loads(rec[col]) if rec.get(col) else None
            rec["identifiers"] = {}
            for r in self.db.execute("SELECT scheme, value FROM identifiers WHERE paper_id=?", (p["id"],)):
                rec["identifiers"].setdefault(r["scheme"], []).append(r["value"])
            rec["sightings"] = [dict(r) for r in self.db.execute(
                "SELECT source, alert_name, alert_kind, seen_at, raw_title, raw_authors, raw_venue, snippet FROM sightings "
                "WHERE paper_id=? ORDER BY seen_at", (p["id"],))]
            yield rec
