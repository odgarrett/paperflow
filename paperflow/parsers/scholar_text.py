"""Google Scholar alerts as plain text (Outlook export / text view). Each result:

    [HTML] Title of the paper … <https://scholar.google.com/scholar_url?url=REAL_URL&...&folt=kw-top>

    A Author, B Author… - Journal, 2025
    … snippet lines …
     <https://scholar.google.com/citations?...> <share links…>
"""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import parse_qs, urlparse

from ..links import extract_ids, is_junk_url, unwrap_url
from ..models import Sighting
from ..text import clean_title, is_truncated, normalize_newlines, squash
from .scholar import FOLT_KIND, alert_name_from_subject, split_byline

_RESULT_LINK = re.compile(r"<(https?://scholar\.google\.[a-z.]+/scholar_url\?[^>\s]+)>")
_SCHOLAR_UI = re.compile(r"<https?://scholar\.google\.[a-z.]+/(?:citations|scholar_share)")


def parse(text: str, subject: str):
    text = normalize_newlines(text)
    matches = [m for m in _RESULT_LINK.finditer(text) if not is_junk_url(unwrap_url(m.group(1)))]
    if not matches:
        return None

    kinds: Counter = Counter()
    sightings = []
    for i, m in enumerate(matches):
        href = m.group(1)
        folt = (parse_qs(urlparse(href).query).get("folt") or [""])[0].split("-")[0]
        if folt:
            kinds[folt] += 1
        url = unwrap_url(href)

        before = text[: m.start()]
        cut = max(before.rfind("\n\n"), before.rfind(">"))
        raw_title = squash(before[cut + 1:])

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        after = text[m.end(): end]
        ui = _SCHOLAR_UI.search(after)
        if ui:
            after = after[: ui.start()]
        else:
            # last item: stop before the next item's title paragraph
            after = after.split("\n\n\n")[0]
        lines = [l.strip() for l in after.split("\n") if l.strip()]
        byline = lines[0] if lines else None
        snippet = squash(" ".join(lines[1:])) or None

        authors, venue, year = split_byline(byline)
        ids = extract_ids(url)
        if "arxiv" not in ids and venue:
            am = re.search(r"arXiv:(\d{4}\.\d{4,5})", venue)
            if am:
                ids["arxiv"] = am.group(1)
        sightings.append(Sighting(
            source="scholar", title=clean_title(raw_title), title_truncated=is_truncated(raw_title),
            authors_raw=authors, venue_raw=venue, year=year, url=url, snippet=snippet, ids=ids,
        ))

    kind = FOLT_KIND.get(kinds.most_common(1)[0][0]) if kinds else None
    if re.search(r"new related research", subject, re.I):
        kind = "related"
    elif kind is None and re.search(r"citation", subject, re.I):
        kind = "citation"
    return alert_name_from_subject(subject), kind, sightings
