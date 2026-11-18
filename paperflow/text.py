"""Small text helpers: title normalization, markup stripping, author surnames."""
from __future__ import annotations

import html
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
ELLIPSES = ("…", "...")


def squash(s: str | None) -> str:
    """Collapse all whitespace runs to single spaces."""
    return _WS.sub(" ", s or "").strip()


def is_truncated(title: str | None) -> bool:
    return squash(title).endswith(ELLIPSES)


def clean_title(title: str | None) -> str | None:
    t = squash(html.unescape(title or ""))
    t = re.sub(r"^\[(?:PDF|HTML|BOOK|B|CITATION|C)\]\s*", "", t, flags=re.I)
    t = t.rstrip("…").rstrip(".").strip()
    return t or None


def strip_tags(s: str | None) -> str | None:
    """Remove inline markup (e.g. <i>Drosophila</i>) from a title."""
    if not s:
        return None
    return squash(html.unescape(_TAG.sub(" ", s))) or None


def clean_abstract(s: str | None) -> str | None:
    """Turn JATS/HTML abstracts into plain text."""
    if not s:
        return None
    txt = squash(BeautifulSoup(s, "lxml").get_text(" "))
    txt = re.sub(r"^(?:abstract|summary)\s*[:.]?\s+", "", txt, flags=re.I)
    return txt or None


def title_key(title: str | None) -> str:
    """Lowercase ASCII alphanumerics separated by single spaces; used for matching."""
    t = unicodedata.normalize("NFKD", title or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = _TAG.sub(" ", t.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


def surname(name: str | None) -> str | None:
    """Best-effort surname: 'J Smith' -> 'smith', 'Smith, J.' -> 'smith'."""
    name = squash((name or "").replace("…", "")).strip(" ,.")
    if not name:
        return None
    last = name.split(",")[0] if "," in name else name.split(" ")[-1]
    return title_key(last) or None


def surnames_from_list(raw: str | None) -> list[str]:
    """Surnames from a comma- or semicolon-separated author string."""
    if not raw:
        return []
    raw = raw.replace("…", "")
    parts = raw.split(";") if ";" in raw else raw.split(",")
    out = []
    for p in parts:
        s = surname(p)
        if s and len(s) > 1 and s not in out:
            out.append(s)
    return out


_BLOCK = {"p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "section"}


def html_to_linktext(html: str) -> str:
    """Render HTML the way Outlook's plain-text view does: links become 'text <url>',
    block elements become line breaks. Lets one text parser handle both forms."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    for a in soup.find_all("a", href=True):
        a.replace_with(f"{a.get_text(' ', strip=True)} <{a['href'].strip()}> ")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for td in soup.find_all(["td", "th"]):
        td.insert_after("\t")
    for el in soup.find_all(_BLOCK):
        el.insert_before("\n")
        el.insert_after("\n")
    text = soup.get_text("")
    lines = [re.sub(r"[ \xa0]+", " ", l).rstrip() for l in text.splitlines()]
    out, blank = [], 0
    for l in lines:  # collapse runs of blank lines to one
        blank = blank + 1 if not l.strip() else 0
        if blank <= 1:
            out.append(l)
    return "\n".join(out).strip()


def normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")
