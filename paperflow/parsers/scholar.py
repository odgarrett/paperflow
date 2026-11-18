"""Google Scholar alert emails.

Each result looks like:
  <h3><span>[PDF]</span> <a class="gse_alrt_title" href="https://scholar.google.com/scholar_url?url=...&folt=kw">Title…</a></h3>
  <div style="color:#006621">A Smith, B Jones… - Journal name, 2024</div>
  <div class="gse_alrt_sni">Snippet…</div>
"""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from ..links import extract_ids, unwrap_url
from ..models import Sighting
from ..text import clean_title, is_truncated, squash

FOLT_KIND = {"kw": "keyword", "art": "author", "cit": "citation", "rel": "related"}

_SUBJECT_SUFFIX = re.compile(
    r"\s+-\s+new (?:results|articles|related research|citations?).*$", re.I
)


def split_byline(line: str | None) -> tuple[str | None, str | None, int | None]:
    """'A Smith, B Jones - Nature, 2024' -> ('A Smith, B Jones', 'Nature', 2024)."""
    if not line:
        return None, None, None
    line = squash(line.replace("\xa0", " "))
    parts = re.split(r"\s+-\s+", line)
    authors = parts[0].strip() or None
    rest = parts[1] if len(parts) > 1 else ""
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", rest)
    year = int(years[-1]) if years else None
    venue = re.sub(r",?\s*(?:19|20)\d{2}\s*$", "", rest).strip(" ,") or None
    return authors, venue, year


def alert_name_from_subject(subject: str) -> str:
    return _SUBJECT_SUFFIX.sub("", squash(subject)) or squash(subject)


def parse(html: str, subject: str):
    """Return (alert_name, alert_kind, sightings), or None if this isn't Scholar's layout."""
    soup = BeautifulSoup(html, "lxml")
    anchors = soup.select("a.gse_alrt_title")
    if not anchors:
        anchors = [a for h3 in soup.find_all("h3") for a in h3.find_all("a", href=True)
                   if "scholar_url" in a["href"]]
    if not anchors:
        return None

    kinds: Counter = Counter()
    sightings = []
    for a in anchors:
        href = a.get("href", "")
        folt = parse_qs(urlparse(href).query).get("folt")
        if folt:
            kinds[folt[0].split("-")[0]] += 1  # e.g. "kw-top" -> "kw"
        url = unwrap_url(href)
        raw_title = a.get_text(" ", strip=True)

        h3 = a.find_parent("h3") or a.parent
        byline = snippet = None
        for sib in h3.find_next_siblings():
            if sib.name == "h3":
                break
            if sib.name != "div":
                continue
            if "gse_alrt_sni" in (sib.get("class") or []):
                snippet = squash(sib.get_text(" "))
                break
            if byline is None:
                byline = sib.get_text(" ", strip=True)

        authors, venue, year = split_byline(byline)
        ids = extract_ids(url)
        if "arxiv" not in ids and venue:
            m = re.search(r"arXiv:(\d{4}\.\d{4,5})", venue)
            if m:
                ids["arxiv"] = m.group(1)

        sightings.append(Sighting(
            source="scholar",
            title=clean_title(raw_title),
            title_truncated=is_truncated(raw_title),
            authors_raw=authors,
            venue_raw=venue,
            year=year,
            url=url,
            snippet=snippet,
            ids=ids,
        ))

    kind = FOLT_KIND.get(kinds.most_common(1)[0][0]) if kinds else None
    if re.search(r"new related research", subject, re.I):
        kind = "related"
    elif kind is None and re.search(r"citation", subject, re.I):
        kind = "citation"
    return alert_name_from_subject(subject), kind, sightings
