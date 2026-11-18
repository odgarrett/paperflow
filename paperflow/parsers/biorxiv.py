"""bioRxiv / medRxiv alert emails (openRxiv-mailer@alerts.highwire.org), two layouts.

Subject Collection alerts — sections per collection, items:
    Title
    Author A, Author B, and Author C
    bioRxiv posted 6 January 2026 doi:10.64898/2026.01.05.697760
    [Abstract] <http://www.biorxiv.org/cgi/content/abstract/2026.01.05.697760v1??collection>  [PDF] <…>

CiteTrack keyword/author alerts ("Your CiteTrack Alert found 22 articles … results 1 to 10"):
    Title
    Author A, Author B and Author C
    bioRxiv. posted 3 September 2026, 10.64898/2026.09.02.748808
    http://biorxiv.org/content/early/2026/10/09/2026.09.02.748808 <…?ct>

Both are anchored on the "posted <date>" line, which every item has.
"""
from __future__ import annotations

import re
from datetime import datetime

from ..links import DOI_RE, biorxiv_id, clean_doi, extract_ids
from ..models import Sighting
from ..text import clean_title, normalize_newlines, squash

_POSTED = re.compile(
    r"^\s*(bioRxiv|medRxiv)\.?\s+posted\s+(\d{1,2}\s+\w+\s+\d{4})\s*,?\s*(?:doi:\s*)?(10\.\d{4,9}/\S+)?",
    re.I,
)
_SECTION = re.compile(r"^\s*([A-Z][A-Za-z ,&-]+?)\s*<https?://(?:www\.)?(?:bio|med)rxiv\.org/cgi/collection/\w+>\s*$")
_RULE = re.compile(r"^\s*_{5,}\s*$")
_LINK = re.compile(r"<(https?://[^>\s]+)>")


def _iso(d: str) -> str | None:
    try:
        return datetime.strptime(squash(d), "%d %B %Y").date().isoformat()
    except ValueError:
        return None


def _authors_list(raw: str) -> str:
    """'A, B, and C' / 'A, B and C' -> 'A; B; C' (semicolons avoid ambiguity with 'Smith, J.')."""
    raw = re.sub(r",?\s+and\s+", ", ", squash(raw))
    return "; ".join(a.strip() for a in raw.split(",") if a.strip())


def parse(text: str, subject: str):
    """Return (alert_name, alert_kind, sightings, n_reported) or None."""
    text = normalize_newlines(text)
    lines = text.split("\n")
    collection = "Subject Collection" in subject or "online collections" in text[:400]

    m = re.search(r'Alert Results for "(.+?)" in (?:bio|med)Rxiv', text)
    alert_name = m.group(1) if m else re.sub(r"^(?:bio|med)Rxiv:\s*", "", subject).strip()
    if collection:
        alert_name = "bioRxiv subject collections"
        kind = "collection"
    elif re.search(r"^\s*Author:", text, re.M):
        kind = "author"
    else:
        kind = "keyword"

    n_reported = None
    m = re.search(r"found (\d+|one) articles?", text)
    if m:
        n_reported = 1 if m.group(1) == "one" else int(m.group(1))

    sightings = []
    section = None
    for i, line in enumerate(lines):
        sm = _SECTION.match(line)
        if sm and i + 1 < len(lines) and _RULE.match(lines[i + 1]):
            section = sm.group(1).strip()
            continue
        pm = _POSTED.match(line)
        if not pm:
            continue

        # Walk back to the start of this item's block.
        j = i - 1
        block: list[str] = []
        while j >= 0 and lines[j].strip() and not _RULE.match(lines[j]) \
                and not lines[j].strip().startswith("Results:"):
            block.insert(0, lines[j].strip())
            j -= 1
        if len(block) < 2:
            continue
        authors = block[-1]
        title = " ".join(block[:-1])

        # Links follow the posted line until the next blank line.
        urls = []
        k = i + 1
        while k < len(lines) and lines[k].strip():
            urls += _LINK.findall(lines[k])
            k += 1

        ids: dict[str, str] = {}
        for u in urls:
            for key, val in extract_ids(u).items():
                ids.setdefault(key, val)
        doi_text = pm.group(3)
        if doi_text:
            dm = DOI_RE.search(doi_text)
            if dm:
                ids["doi"] = clean_doi(dm.group(1))
        bid = biorxiv_id(ids.get("doi"))
        if bid:
            ids["biorxiv"] = bid

        posted = _iso(pm.group(2))
        sightings.append(Sighting(
            source="biorxiv",
            title=clean_title(title),
            authors_raw=_authors_list(authors),
            venue_raw=pm.group(1),
            year=int(posted[:4]) if posted else None,
            url=urls[0] if urls else None,
            ids=ids,
            alert=f"bioRxiv collection: {section}" if collection and section else None,
            posted=posted,
        ))
    if not sightings:
        return None
    return alert_name, kind, sightings, n_reported
