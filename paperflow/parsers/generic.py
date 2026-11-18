"""Format-agnostic parser: finds paper links in any alert email.

Strategy: walk the links in document order. A link whose text looks like a title starts a
new paper; links that carry an identifier but no title (e.g. "PDF", "Full text") attach to
the paper just above them. Items that share an identifier are merged. This handles most
journal/preprint/Stork-style alerts reasonably well; a dedicated parser per source can be
added later once real sample emails are available.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ..links import extract_ids, is_junk_url, unwrap_url
from ..models import Sighting
from ..text import clean_title, is_truncated, squash

_BLOCK_TAGS = ("td", "li", "p", "div", "tr")
_JUNK_TEXT = re.compile(
    r"^(unsubscribe|manage|view|privacy|preferences|settings|click|read more|more|full[ -]?text|pdf|html"
    r"|abstract|download|see all|cancel|edit|terms|contact|help|sign in|log ?in|forward|share|save"
    r"|twitter|facebook|linkedin|email|follow|update|here|link)\b",
    re.I,
)
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def looks_like_title(text: str | None) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 4 and len(text) < 25:
        return False
    if len(text) > 400:
        return False
    if text.lower().startswith(("http://", "https://", "www.")):
        return False
    if _JUNK_TEXT.match(text) and len(words) <= 4:
        return False
    return True


def _compatible(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in a.keys() & b.keys())


def _shares_id(a: dict, b: dict) -> bool:
    return any(a.get(k) == v for k, v in b.items())


def _new_item(url, ids, title, context=None):
    return {"urls": [url], "ids": dict(ids), "titles": [title] if title else [], "context": context}


def _merge_items(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for it in items:
        target = next((o for o in out if it["ids"] and _shares_id(o["ids"], it["ids"])
                       and _compatible(o["ids"], it["ids"])), None)
        if target is None:
            out.append(it)
            continue
        target["ids"].update(it["ids"])
        target["urls"] += [u for u in it["urls"] if u not in target["urls"]]
        target["titles"] += it["titles"]
        target["context"] = target["context"] or it["context"]
    return out


def _context(a, title: str) -> str | None:
    """Text of the smallest enclosing block that contains more than the title (authors, journal…)."""
    node = a
    for _ in range(6):
        node = node.parent
        if node is None or node.name in ("body", "html", "[document]"):
            break
        if node.name in _BLOCK_TAGS:
            t = squash(node.get_text(" "))
            if len(t) > len(title) + 20:
                if len(t) <= 1500:
                    return squash(t.replace(title, " ", 1))[:800] or None
                break
    return None


def _to_sighting(item: dict, source: str) -> Sighting:
    raw_title = max(item["titles"], key=len) if item["titles"] else None
    return Sighting(
        source=source,
        title=clean_title(raw_title),
        title_truncated=is_truncated(raw_title),
        url=item["urls"][0],
        snippet=item["context"],
        ids=item["ids"],
    )


def _parse_html(html: str, source: str) -> list[Sighting]:
    soup = BeautifulSoup(html, "lxml")
    items: list[dict] = []
    current = None
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().startswith(("http://", "https://")):
            continue
        url = unwrap_url(href)
        if is_junk_url(url):
            continue
        text = squash(a.get_text(" "))
        ids = extract_ids(url)
        titleish = looks_like_title(text)
        if not ids and not titleish:
            continue
        if titleish:
            if current and current["titles"] and current["titles"][0] == text:
                current["urls"].append(url)
                current["ids"].update({k: v for k, v in ids.items() if k not in current["ids"]})
            else:
                current = _new_item(url, ids, text, _context(a, text))
                items.append(current)
        elif current is not None and _compatible(current["ids"], ids):
            current["urls"].append(url)
            current["ids"].update(ids)
        else:
            current = _new_item(url, ids, None)
            items.append(current)
    return [_to_sighting(it, source) for it in _merge_items(items)]


def _parse_text(text: str, source: str) -> list[Sighting]:
    lines = [squash(l) for l in text.splitlines()]
    items = []
    for i, line in enumerate(lines):
        for m in _URL_RE.finditer(line):
            url = unwrap_url(m.group(0).rstrip(".,;"))
            if is_junk_url(url):
                continue
            ids = extract_ids(url)
            before = squash(line[: m.start()]).rstrip(":-–")
            title = before if looks_like_title(before) else None
            j = i - 1
            while title is None and j >= 0 and i - j <= 3:
                if looks_like_title(lines[j]) and not _URL_RE.search(lines[j]):
                    title = lines[j]
                j -= 1
            if ids or title:
                items.append(_new_item(url, ids, title))
    return [_to_sighting(it, source) for it in _merge_items(items)]


def parse(html: str | None, text: str | None, source: str) -> list[Sighting]:
    if html:
        return _parse_html(html, source)
    if text:
        return _parse_text(text, source)
    return []
