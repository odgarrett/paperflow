"""Stork alerts (support@storkapp.me), plain text / Outlook rendering.

Keyword digests ("Stork has brought you N new publications …"). Each item has a
"PMID … doi …" line followed by a share link whose mailto subject/body carry the clean
title, authors, year and PMID, so items are anchored on those share links. Two layouts
have been used:
    old:  Title. <link> by A, B, …, Z (2026) Journal (impact factor: 13.1)
          PMID: 41495893 doi: 10.1093/nar/gkaf1448
    new:  Title. <link>
          A, B, …, Z · 2026
          JournalIF 8.9
          PMID 42886876 · doi 10.1212/WNL.0000000000218692
Keyword sections start with a "<keyword>  Big Analysis" line.

Monthly "Hot papers in your field" digests: numbered items without identifiers:
    1<TAB>Title. <link>
    A, B, …, Z · 2026                                         (new)
    JournalIF 45.1
or  A, B, …, Z et al. Journal (impact factor: 16) (2026)      (old)
"""
from __future__ import annotations

import re
from urllib.parse import unquote_plus

from ..links import clean_doi
from ..models import Sighting
from ..text import clean_title, normalize_newlines, squash

_MAILTO = re.compile(r"<mailto:\?subject=([^&>]*)&(?:amp;)?body=([^>]*)>")
_PMID_DOI = re.compile(r"PMID:?\s*(\d{5,9})(?:\s*·?\s*doi:?\s*(10\.\d{4,9}/[^\s<>]+))?", re.I)
_SECTION = re.compile(r"^\s*(.+?)\s*(?:\t|\s{2,})\s*Big Analysis", re.M)
_VENUE_NEW = re.compile(r"^\s*(.+?)\s*IF\s*[\d.]+\s*$", re.M)
_VENUE_OLD = re.compile(r"\(\d{4}\)\s+(.+?)\s*\(impact factor", re.I)
_HOT_OLD = re.compile(r"^\s*(.+?)\s+et al\.\s+(.+?)\s*(?:\(impact factor:[^)]*\))?\s*\((\d{4})\)\s*$")
_HOT_ITEM = re.compile(r"^\s*(\d+)\t(.+?)\s*<(https?://[^>\s]+)>")


def _parse_digest(text: str) -> list[Sighting]:
    sections = [(m.start(), squash(m.group(1))) for m in _SECTION.finditer(text)]
    sightings = []
    prev_end = 0
    for m in _MAILTO.finditer(text):
        subj = unquote_plus(m.group(1))
        body = unquote_plus(m.group(2)).replace("\r", "\n")
        pm = re.search(r"pubpaper/(\d+)", body)
        title = re.sub(r"^Stork:\s*", "", subj).strip()
        ab = re.search(r"\nby (.+?) \((\d{4})\)", body)
        authors = ab.group(1).replace("…", "").strip(" ,") if ab else None
        year = int(ab.group(2)) if ab else None

        segment = text[prev_end: m.start()]
        prev_end = m.end()
        ids = {}
        ids_m = list(_PMID_DOI.finditer(segment))
        if ids_m:
            last = ids_m[-1]
            ids["pmid"] = last.group(1)
            if last.group(2):
                ids["doi"] = clean_doi(last.group(2))
        if pm:
            ids.setdefault("pmid", pm.group(1))
        venue = None
        vm = _VENUE_OLD.search(segment) or _VENUE_NEW.search(segment)
        if vm:
            venue = squash(vm.group(1))
        section = None
        for pos, name in sections:
            if pos < m.start():
                section = name
        sightings.append(Sighting(
            source="stork",
            title=clean_title(title),
            authors_raw=authors,
            venue_raw=venue,
            year=year,
            url=f"https://www.storkapp.me/pubpaper/{ids['pmid']}" if ids.get("pmid") else None,
            ids=ids,
            alert=f"Stork: {section}" if section else None,
        ))
    return sightings


def _parse_hot(text: str) -> list[Sighting]:
    lines = text.split("\n")
    sightings = []
    for i, line in enumerate(lines):
        m = _HOT_ITEM.match(line)
        if not m:
            continue
        authors = year = venue = None
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        am = re.match(r"^\s*(.+?)\s*·\s*(\d{4})\s*$", nxt)
        om = _HOT_OLD.match(nxt)
        if am:  # new layout: "A, B, …, Z · 2026" then "JournalIF 45.1"
            authors, year = am.group(1).replace("…", "").strip(" ,"), int(am.group(2))
            vm = _VENUE_NEW.match(lines[i + 2]) if i + 2 < len(lines) else None
            if vm:
                venue = squash(vm.group(1))
        elif om:  # old layout: "A, B, …, Z et al. Journal (impact factor: 16) (2026)"
            authors = om.group(1).replace("…", "").strip(" ,")
            venue, year = squash(om.group(2)), int(om.group(3))
        sightings.append(Sighting(
            source="stork", title=clean_title(m.group(2)), authors_raw=authors, venue_raw=venue,
            year=year, url=m.group(3), ids={}, alert="Stork: hot papers", alert_kind="hot",
        ))
    return sightings


def parse(text: str, subject: str):
    """Return (alert_name, alert_kind, sightings, n_reported) or None."""
    text = normalize_newlines(text)
    if re.search(r"hot papers", subject, re.I):
        s = _parse_hot(text)
        return ("Stork hot papers", "hot", s, None) if s else None
    s = _parse_digest(text)
    if not s:
        return None
    m = re.search(r"brought you (\d+) new publication", subject + " " + text[:500])
    name = re.sub(r"^Stork has brought you \d+ new publications? for\s*", "", subject).strip()
    name = re.sub(r"\s+etc$", "", name)
    return f"Stork: {name}", "keyword", s, int(m.group(1)) if m else None
