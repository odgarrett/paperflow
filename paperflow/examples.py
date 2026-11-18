"""Read your example papers from examples/{mandatory,interest,skip}/.

Anything you drop in a label folder (subfolders are fine, e.g. mirroring Zotero
sub-collections) is read according to its type:

  .bib            BibTeX, e.g. Zotero "Export Collection…" → BibTeX (includes abstracts)
  .ris            RIS export
  .json           CSL JSON export
  .pdf            the paper itself: DOI and title are read from the PDF, the rest is looked up
  .txt            one DOI or title per line (lines starting with # are ignored)
  .md             notes (e.g. Obsidian literature notes): every DOI found in the file

Examples with a DOI but no abstract get their metadata looked up (cached), the same way
alert papers do.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .links import DOI_RE, clean_doi
from .text import clean_abstract, squash, strip_tags, title_key

log = logging.getLogger(__name__)

LABELS = ("mandatory", "interest", "skip")
_FOLDER_ALIASES = {
    "mandatory": "mandatory", "must-read": "mandatory", "must_read": "mandatory",
    "interest": "interest", "of_interest": "interest", "of-interest": "interest", "of interest": "interest",
    "interesting": "interest", "skip": "skip", "not_interested": "skip", "not interested": "skip",
}
SUPPORTED = {".bib", ".ris", ".json", ".pdf", ".txt", ".md"}


@dataclass
class Example:
    label: str
    source: str
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None

    @property
    def key(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        return "title:" + hashlib.sha1(title_key(self.title).encode()).hexdigest()[:16]


# -- BibTeX ----------------------------------------------------------------------------------

def _bib_value(s: str, i: int) -> tuple[str, int]:
    """Read a field value starting at s[i] ({…}, "…" or bare); return (value, next index)."""
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{" and s[j - 1] != "\\":
                depth += 1
            elif s[j] == "}" and s[j - 1] != "\\":
                depth -= 1
                if depth == 0:
                    return s[i + 1: j], j + 1
            j += 1
        return s[i + 1:], len(s)
    if s[i] == '"':
        j = i + 1
        while j < len(s) and not (s[j] == '"' and s[j - 1] != "\\"):
            j += 1
        return s[i + 1: j], j + 1
    m = re.match(r"[^,}\s]+", s[i:])
    return (m.group(0), i + m.end()) if m else ("", i)


def _delatex(s: str) -> str:
    s = re.sub(r"\\(?:textit|textbf|emph|mathrm|textsuperscript|textsubscript)\s*", "", s)
    s = re.sub(r"\\[`'^\"~=.uvHcdb]\{?([A-Za-z])\}?", r"\1", s)   # accents: \'{e} -> e
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("--", "–")
    return squash(s.replace("{", "").replace("}", ""))


def parse_bibtex(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{", text):
        if m.group(1).lower() in ("comment", "preamble", "string"):
            continue
        body, _ = _bib_value(text, m.end() - 1)
        fields = {}
        i = body.find(",") + 1  # skip citation key
        while i and i < len(body):
            fm = re.match(r"\s*([\w-]+)\s*=", body[i:])
            if not fm:
                break
            val, j = _bib_value(body, i + fm.end())
            fields[fm.group(1).lower()] = _delatex(val)
            k = body.find(",", j)
            i = k + 1 if k != -1 else len(body)
        entries.append(fields)
    return entries


# -- other formats -----------------------------------------------------------------------------

def parse_ris(text: str) -> list[dict]:
    entries, cur = [], {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "ER":
            entries.append(cur)
            cur = {}
        elif tag in ("TI", "T1") and "title" not in cur:
            cur["title"] = val
        elif tag in ("AB", "N2") and "abstract" not in cur:
            cur["abstract"] = val
        elif tag == "DO":
            cur["doi"] = val
    if cur:
        entries.append(cur)
    return entries


def parse_csl_json(text: str) -> list[dict]:
    data = json.loads(text)
    items = data if isinstance(data, list) else data.get("items", [data])
    return [{"title": it.get("title"), "abstract": it.get("abstract"), "doi": it.get("DOI") or it.get("doi")}
            for it in items if isinstance(it, dict)]


def parse_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        log.warning("Can't read PDF %s: %s", path.name, e)
        return []
    meta = reader.metadata or {}
    title = str(meta.get("/Title") or "").strip()
    if len(title) < 15 or title.lower().startswith(("untitled", "microsoft word")) or title.lower().endswith(".pdf"):
        title = ""
    doi = None
    for key in ("/doi", "/DOI", "/prism:doi"):
        if meta.get(key):
            doi = str(meta[key])
    first_pages = ""
    for page in reader.pages[:2]:
        try:
            first_pages += (page.extract_text() or "") + "\n"
        except Exception:
            pass
    if not doi:
        m = DOI_RE.search(first_pages)  # a paper's own DOI is printed on its first page
        doi = m.group(1) if m else None
    abstract = None
    m = re.search(r"\b(?:Abstract|ABSTRACT|Summary|SUMMARY)\b[:.\s]*(.{200,2500}?)(?:\n\s*\n|\b(?:Introduction|INTRODUCTION|Keywords|KEYWORDS)\b)",
                  first_pages, re.S)
    if m:
        abstract = squash(m.group(1))
    if not title:
        # Fall back to the first substantial line of page 1 (usually the title); a DOI
        # lookup later replaces it with the real one.
        for line in first_pages.splitlines():
            line = squash(line)
            if len(line.split()) >= 4 and not re.search(r"doi|https?://|©|journal|volume|received", line, re.I):
                title = line
                break
    if not (title or doi):
        title = path.stem.replace("_", " ")
    return [{"title": title or None, "doi": doi, "abstract": abstract}]


def parse_lines(text: str, titles_allowed: bool) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dois = DOI_RE.findall(line)
        if dois:
            out += [{"doi": d} for d in dois]
        elif titles_allowed and len(line.split()) >= 4:
            out.append({"title": line})
    return out


def _read_file(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if ext == ".bib":
        return parse_bibtex(text)
    if ext == ".ris":
        return parse_ris(text)
    if ext == ".json":
        return parse_csl_json(text)
    if ext == ".txt":
        return parse_lines(text, titles_allowed=True)
    if ext == ".md":
        return parse_lines(text, titles_allowed=False)
    return []


def label_of_folder(name: str) -> str | None:
    return _FOLDER_ALIASES.get(name.strip().lower())


def scan(examples_dir: Path) -> tuple[list[Example], list[str]]:
    """Read every example file. Returns (examples, problems)."""
    examples: dict[str, Example] = {}
    problems: list[str] = []
    if not examples_dir.exists():
        return [], [f"{examples_dir} doesn't exist (run `paperflow init-triage`)"]
    for folder in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        label = label_of_folder(folder.name)
        if label is None:
            problems.append(f"Ignoring folder {folder.name!r} (expected mandatory, interest or skip)")
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in SUPPORTED:
                problems.append(f"Unsupported file type: {path.relative_to(examples_dir)}")
                continue
            rel = str(path.relative_to(examples_dir))
            try:
                records = _read_file(path)
            except Exception as e:
                problems.append(f"Couldn't read {rel}: {e}")
                continue
            if not records:
                problems.append(f"No papers found in {rel}")
            for r in records:
                doi = clean_doi(r["doi"]) if r.get("doi") else None
                title = strip_tags(r.get("title"))
                if not (doi or (title and len(title_key(title)) >= 10)):
                    continue
                ex = Example(label=label, source=rel, doi=doi, title=title,
                             abstract=clean_abstract(r.get("abstract")) if r.get("abstract") else None)
                prev = examples.get(ex.key)
                if prev and prev.label != label:
                    problems.append(f"{ex.title or ex.doi!r} is in both {prev.label} and {label}; using {label}")
                examples[ex.key] = ex
    return list(examples.values()), problems
