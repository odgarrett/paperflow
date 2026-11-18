"""Full texts for briefings: open-access retrieval plus a drop folder for PDFs you have.

Sources, in order:
  1. your PDFs in the drop folder (default fulltext/ next to config.yaml), matched to papers
     by the DOI printed in them (or failing that, by title)
  2. for preprints since published: the published version's full text in Europe PMC
  3. Europe PMC full-text XML (PubMed Central open-access articles, and many preprints)
  4. bioRxiv / medRxiv's own full-text XML
  5. open-access PDFs located by Unpaywall and OpenAlex (published version first)

bioRxiv's website rate-limits by IP address (blocking your browser too), so requests to it are
spaced out, and if it answers "too many requests" it's left alone for the rest of the run and
the papers that needed it are retried next time.

XML is preferred over PDF because its sections come out cleanly. A downloaded PDF is only
kept if it really is a PDF and contains the paper's title, so a login page or a different
paper is never stored as the full text. Texts are saved in data/fulltext/<paper id>.txt.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from .http import Http, HttpError
from .links import biorxiv_id, clean_doi
from .store import utcnow
from .text import squash, title_key

log = logging.getLogger(__name__)

MIN_CHARS = 5000          # less than this is an abstract page or a scan, not a full text
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


@dataclass
class FullText:
    status: str            # ok | unavailable (no open copy found) | error (lookup failed; retried next run)
    source: str | None = None
    url: str | None = None
    text: str | None = None
    detail: str | None = None


# -- converting ----------------------------------------------------------------------------

def jats_to_text(xml: bytes) -> str:
    """JATS XML (PMC, Europe PMC, bioRxiv) to plain text with section headings."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(xml, "lxml-xml")
    for t in soup.find_all(["ref-list", "back", "xref", "table", "inline-formula", "disp-formula"]):
        t.decompose()
    out = []
    title = soup.find("article-title")
    if title:
        out.append("# " + squash(title.get_text(" ")))
    abstract = soup.find("abstract")
    if abstract:
        out.append("## Abstract\n" + squash(abstract.get_text(" ")))
    body = soup.find("body")
    if body:
        for sec in body.find_all("sec"):  # nested sections each contribute their own paragraphs
            head = sec.find("title", recursive=False)
            paras = [squash(p.get_text(" ")) for p in sec.find_all("p", recursive=False)]
            if head or paras:
                level = "###" if sec.find_parent("sec") else "##"
                block = (f"{level} {squash(head.get_text(' '))}\n" if head else "") + "\n\n".join(x for x in paras if x)
                out.append(block.strip())
        if not body.find("sec"):
            out.append("\n\n".join(squash(p.get_text(" ")) for p in body.find_all("p")))
    captions = []
    for fig in soup.find_all("fig"):
        label = fig.find("label")
        cap = fig.find("caption")
        if cap:
            captions.append(f"{squash(label.get_text(' ')) + ': ' if label else ''}{squash(cap.get_text(' '))}")
    if captions:
        out.append("## Figure legends\n" + "\n\n".join(captions))
    return "\n\n".join(x for x in out if x).strip()


_REFS_HEADING = re.compile(r"^\s*(?:references|literature cited|bibliography|references and notes)\s*$", re.I | re.M)


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    # Drop the reference list: the last "References" heading in the second half of the text.
    cut = None
    for m in _REFS_HEADING.finditer(text):
        if m.start() > len(text) * 0.5:
            cut = m.start()
    if cut:
        text = text[:cut]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _mentions_title(text: str, title: str | None) -> bool:
    if not title:
        return True
    tk, head = title_key(title), title_key(text[:6000])
    return fuzz.partial_ratio(tk, head) >= 85 if tk else True


# -- the drop folder -----------------------------------------------------------------------

_ID_NAME = re.compile(r"^id[ _-]?(\d+)(?:\b|_)|^(\d+)\.pdf$", re.I)


def _pdf_head(path: Path) -> tuple[str, list[str], list[str]]:
    """(first two pages' text, DOIs from the PDF's metadata, DOIs on those pages in order)."""
    from pypdf import PdfReader
    from .links import DOI_RE
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:2]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    head = "\n".join(pages)
    meta_dois = []
    meta = reader.metadata or {}
    for key in ("/doi", "/DOI", "/prism:doi", "/WPS-ARTICLEDOI"):
        if meta.get(key):
            m = DOI_RE.search(str(meta[key]))
            if m:
                meta_dois.append(clean_doi(m.group(1)))
    page_dois = []
    for m in DOI_RE.finditer(head):
        d = clean_doi(m.group(1))
        if d not in page_dois:
            page_dois.append(d)
    return head, meta_dois, page_dois


def _title_score(title: str | None, text: str) -> float:
    tk = title_key(title or "")
    return fuzz.partial_ratio(tk, title_key(text[:6000])) if len(tk) >= 10 else 0.0


def _match_pdf(store, path: Path) -> tuple[int | None, str]:
    """Which paper a dropped PDF belongs to, and how (or why not)."""
    m = _ID_NAME.search(path.name)
    if m:
        pid = int(m.group(1) or m.group(2))
        if store.db.execute("SELECT 1 FROM papers WHERE id=?", (pid,)).fetchone():
            return pid, "file name"
        return None, f"its name says paper {pid}, but there's no paper with that id"
    try:
        head, meta_dois, page_dois = _pdf_head(path)
    except Exception as e:
        return None, f"unreadable ({e.__class__.__name__})"
    dois = meta_dois + [d for d in page_dois if d not in meta_dois]
    # Every DOI on the first pages that belongs to a paper in the database.
    candidates = []
    for d in dois:
        keys = [("doi", d)] + ([("biorxiv", biorxiv_id(d))] if biorxiv_id(d) else [])
        for scheme, value in keys:
            r = store.db.execute("SELECT paper_id FROM identifiers WHERE scheme=? AND value=?", (scheme, value)).fetchone()
            if r and r[0] not in [c[0] for c in candidates]:
                candidates.append((r[0], d))
    if candidates:
        scored = []
        for i, (pid, d) in enumerate(candidates):
            title = store.db.execute("SELECT title FROM papers WHERE id=?", (pid,)).fetchone()[0]
            scored.append((_title_score(title, head), -i, pid, d))
        best = max(scored)
        if best[0] >= 60:
            return best[2], "DOI and title"
        # The publisher's embedded metadata names the paper's own DOI: trust it even if the title
        # didn't extract well. A DOI that's merely printed on the page could be a citation.
        for pid, d in candidates:
            if d in meta_dois:
                return pid, "DOI in the PDF's metadata"
    # The title on the first page, among papers still without a full text.
    norm = title_key(head[:6000])
    if len(norm) > 50:
        rows = store.db.execute(
            "SELECT id, title FROM papers WHERE title IS NOT NULL AND length(title) >= 25 "
            "AND id NOT IN (SELECT paper_id FROM fulltexts WHERE status='ok')").fetchall()
        scores = sorted(((fuzz.partial_ratio(title_key(t), norm), pid) for pid, t in rows), reverse=True)[:2]
        if scores and scores[0][0] >= 92 and (len(scores) < 2 or scores[0][0] - scores[1][0] >= 4):
            return scores[0][1], "title"
    if candidates:
        return None, "the DOIs on its first pages belong to other papers in the database (cited ones?)"
    return None, "no DOI found and its title didn't match a paper"


def scan_drop_folder(store, folder: Path) -> tuple[int, list[str]]:
    """Read PDFs you put in the drop folder. Returns (number added, problems).

    A PDF is matched to its paper by its file name (id1834.pdf), else by the DOIs on its
    first pages (checked against the paper's title), else by its title."""
    if not folder.exists():
        return 0, []
    added, problems = 0, []
    known = {r[0] for r in store.db.execute("SELECT url FROM fulltexts WHERE source='your pdf'")}
    for path in sorted(folder.rglob("*.pdf")):
        if str(path) in known:
            continue
        pid, how = _match_pdf(store, path)
        if pid is None:
            problems.append(f"{path.name}: {how}. Rename it id<number>.pdf with the paper's id to match it.")
            continue
        try:
            text = pdf_to_text(path.read_bytes())
        except Exception as e:
            problems.append(f"{path.name}: unreadable ({e.__class__.__name__})")
            continue
        if len(text) < MIN_CHARS:
            problems.append(f"{path.name}: almost no text in it (a scanned PDF?)")
            continue
        save(store, pid, FullText("ok", "your pdf", str(path), text, detail=f"matched by {how}"))
        added += 1
    return added, problems


# -- open-access retrieval -----------------------------------------------------------------

class Fetcher:
    def __init__(self, http: Http, contact_email: str):
        self.http, self.email = http, contact_email

    def _epmc_record(self, doi: str | None, pmid: str | None) -> dict | None:
        q = f"EXT_ID:{pmid} AND SRC:MED" if pmid else f'DOI:"{doi}"'
        j = self.http.get_json(f"{EPMC}/search", {"query": q, "resultType": "core", "format": "json", "pageSize": 3})
        results = ((j or {}).get("resultList") or {}).get("result") or []
        # prefer a record that has full text
        results.sort(key=lambda r: (r.get("inEPMC") != "Y", r.get("source") != "PPR"))
        return results[0] if results else None

    def _xml(self, url: str) -> bytes | None:
        got = self.http.get_bytes(url)
        if not got:
            return None
        data, ctype, _ = got
        return data if b"<" in data[:200] and (b"<article" in data[:5000] or b"<!DOCTYPE" in data[:300]) else None

    def _pdf(self, url: str, title: str | None) -> tuple[str | None, str]:
        got = self.http.get_bytes(url)
        if not got:
            return None, "download failed"
        data, ctype, _ = got
        if not data.startswith(b"%PDF"):
            return None, "not a PDF (probably a login or bot-check page)"
        try:
            text = pdf_to_text(data)
        except Exception as e:
            return None, f"unreadable PDF ({e.__class__.__name__})"
        if len(text) < MIN_CHARS:
            return None, "PDF has almost no text"
        if not _mentions_title(text, title):
            return None, "PDF doesn't contain the paper's title (wrong file?)"
        return text, "ok"

    def _europepmc(self, doi: str | None, pmid: str | None) -> FullText | None:
        rec = self._epmc_record(doi, pmid)
        if not rec:
            return None
        ids = []
        if rec.get("pmcid"):
            ids.append(rec["pmcid"])
        if rec.get("source") == "PPR" and rec.get("id"):
            ids.append(rec["id"])
        for rid in ids:
            xml = self._xml(f"{EPMC}/{rid}/fullTextXML")
            if xml:
                text = jats_to_text(xml)
                if len(text) >= MIN_CHARS:
                    return FullText("ok", "europepmc", f"https://europepmc.org/article/{rid}", text)
        return None

    def fetch(self, paper: dict) -> FullText:
        doi, pmid, title = paper.get("doi"), paper.get("pmid"), paper.get("title")
        published = paper.get("published_doi")
        tried: list[str] = []
        hosts: set[str] = set()
        try:
            # 1. A preprint's published version, if open in Europe PMC: usually the better text,
            #    and fetching it puts no load on the preprint server.
            if published and published != doi:
                got = self._europepmc(published, None)
                if got:
                    got.source = "europepmc (published version)"
                    return got
            # 2. Europe PMC
            if doi or pmid:
                got = self._europepmc(doi, pmid)
                if got:
                    return got
                tried.append("Europe PMC: no open full text")
            # 3. bioRxiv / medRxiv XML
            bid = biorxiv_id(doi)
            if bid:
                for server in ("biorxiv", "medrxiv"):
                    j = self.http.get_json(f"https://api.biorxiv.org/details/{server}/{doi}")
                    coll = (j or {}).get("collection") or []
                    if coll:
                        latest = max(coll, key=lambda r: int(r.get("version") or 0))
                        if latest.get("jatsxml"):
                            hosts.add(latest["jatsxml"].split("/")[2])
                            xml = self._xml(latest["jatsxml"])
                            if xml:
                                text = jats_to_text(xml)
                                if len(text) >= MIN_CHARS:
                                    return FullText("ok", server, latest["jatsxml"], text)
                        tried.append(f"{server}: full text not downloadable")
                        break
            # 4. Open-access PDFs (published version first)
            urls: list[str] = []
            for d in [x for x in (published, doi) if x]:
                j = self.http.get_json(f"https://api.unpaywall.org/v2/{d}", {"email": self.email})
                for loc in ([(j or {}).get("best_oa_location")] + ((j or {}).get("oa_locations") or [])):
                    u = (loc or {}).get("url_for_pdf")
                    if u and u.split("?")[0] not in {x.split("?")[0] for x in urls}:
                        urls.append(u)
            u = paper.get("oa_pdf_url")
            if u and u.split("?")[0] not in {x.split("?")[0] for x in urls}:
                urls.append(u)
            for url in urls[:4]:
                hosts.add(url.split("/")[2])
                text, why = self._pdf(url, title)
                if text:
                    return FullText("ok", "open-access pdf", url, text)
                tried.append(f"{url.split('/')[2]}: {why}")
            if not urls and (doi or published):
                tried.append("no open-access copy known")
        except HttpError as e:
            return FullText("error", detail=f"lookup failed: {e}")
        blocked = sorted(h for h in hosts if h in getattr(self.http, "skipped_hosts", {}))
        if blocked:
            # Not a verdict on the paper: a site we needed stopped answering. Try again next run.
            return FullText("error", detail=f"{', '.join(blocked)} rate limiting or not responding; "
                                            "retried on the next run")
        return FullText("unavailable", detail="; ".join(tried) or "no identifiers to look up")


# -- storage -------------------------------------------------------------------------------

def save(store, pid: int, ft: FullText) -> None:
    if ft.status == "ok":
        store.fulltext_dir.mkdir(parents=True, exist_ok=True)
        (store.fulltext_dir / f"{pid}.txt").write_text(ft.text, encoding="utf-8")
    store.db.execute("INSERT OR REPLACE INTO fulltexts VALUES (?,?,?,?,?,?,?)",
                     (pid, ft.status, ft.source, ft.url, len(ft.text) if ft.text else None, ft.detail, utcnow()))


def load(store, pid: int) -> str | None:
    p = store.fulltext_dir / f"{pid}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None
