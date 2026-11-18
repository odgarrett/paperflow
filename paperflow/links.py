"""Turn links found in alert emails into paper identifiers (DOI, PMID, PMCID, arXiv)."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

# Query parameters that redirect wrappers (Scholar, Google, Outlook Safe Links, trackers)
# use to carry the real destination.
_WRAP_PARAMS = ("url", "q", "u", "target", "dest", "destination", "redirect", "redirect_url", "link")

# bioRxiv/medRxiv preprint ids look like 2026.01.05.697760. The DOI prefix changed from
# 10.1101 to 10.64898 (openRxiv) in late 2025, so the bare id is the stable key.
BIORXIV_RE = re.compile(r"(?:bio|med)rxiv\.org/.*?(\d{4}\.\d{2}\.\d{2}\.\d{6,})", re.I)
BIORXIV_DOI_PREFIXES = ("10.64898", "10.1101")
_BIORXIV_SUFFIX = re.compile(r"^(\d{4}\.\d{2}\.\d{2}\.\d{6,})(?:v\d+)?$", re.I)
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>#?&]+)", re.I)
PMID_RE = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/|europepmc\.org/(?:abstract|article)/MED/)(\d{5,9})",
    re.I,
)
PMCID_RE = re.compile(r"\b(PMC\d{4,9})\b", re.I)
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
NATURE_RE = re.compile(r"nature\.com/articles/([a-z0-9][a-z0-9.\-]*?)(?:\.pdf)?(?:[/?#]|$)", re.I)
ELIFE_RE = re.compile(r"elifesciences\.org/(?:articles|reviewed-preprints)/(\d{4,6})", re.I)

_DOI_SUFFIX = re.compile(
    r"[/.](?:full|abstract|short|pdf|epdf|fulltext|meta|summary|html?|article-info|article-metrics"
    r"|supplementary-material|figures-only|references|reader)$",
    re.I,
)

_JUNK_HOSTS = (
    "facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com",
    "youtube.com", "bsky.app", "reddit.com", "weibo.com",
)
_JUNK_PATH = re.compile(
    r"(unsubscribe|optout|opt-out|preferences|/manage|settings|privacy|/account|/login|/signin"
    r"|/alerts?/(?:edit|delete|cancel|manage)|scholar_alerts|scholar_settings|/help)",
    re.I,
)


def unwrap_url(href: str) -> str:
    """Peel off redirect wrappers until we reach the real destination."""
    for _ in range(4):
        p = urlparse(href)
        host = p.netloc.lower()
        if "urldefense" in host:
            m = re.search(r"__(.+?)__;", href)
            if m:
                href = m.group(1)
                continue
        qs = parse_qs(p.query)
        nxt = None
        for k in _WRAP_PARAMS:
            vals = qs.get(k)
            if vals and vals[0].startswith(("http://", "https://")):
                nxt = vals[0]
                break
        if nxt and nxt != href:
            href = nxt
            continue
        break
    return href


def is_junk_url(url: str) -> bool:
    """Links that are never papers: social media, alert management, Scholar UI."""
    if url.lower().startswith("mailto:"):
        return True
    p = urlparse(url)
    host = p.netloc.lower().split(":")[0]
    if any(host == h or host.endswith("." + h) for h in _JUNK_HOSTS):
        return True
    if re.match(r"(?:.*\.)?scholar\.google\.", host):
        return True
    return bool(_JUNK_PATH.search(p.path))


# Silverchair-hosted journals (OUP, ACS, Company of Biologists, ASH, CSIRO…) append a
# numeric article id after the DOI: /article-pdf/doi/10.1093/bioadv/vbag270/71046481/vbag270.pdf
_SILVERCHAIR_PATH = re.compile(r"/(?:advance-)?article(?:-pdf|-abstract|-lookup)?/doi/", re.I)
# DOI prefixes of preprint servers (bioRxiv/medRxiv, arXiv, ChemRxiv, Research Square,
# Preprints.org, SSRN, OSF) — used to pair preprints with their published versions.
PREPRINT_DOI_PREFIXES = BIORXIV_DOI_PREFIXES + ("10.48550", "10.26434", "10.21203", "10.20944",
                                                "10.2139", "10.31219", "10.31234")


def is_preprint_doi(doi: str | None) -> bool:
    return bool(doi) and doi.split("/")[0] in PREPRINT_DOI_PREFIXES


def clean_doi(d: str) -> str:
    d = unquote(unquote(d)).strip()
    d = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", d, flags=re.I)
    d = d.rstrip(".,;:]}>\"'")
    while d.endswith(")") and d.count("(") < d.count(")"):
        d = d[:-1]
    prev = None
    while prev != d:
        prev = d
        d = _DOI_SUFFIX.sub("", d)
        d = re.sub(r"_references?$", "", d, flags=re.I)  # Springer PDF links
    prefix, _, suffix = d.partition("/")
    if prefix in BIORXIV_DOI_PREFIXES:
        d = re.sub(r"v\d+$", "", d, flags=re.I)  # bioRxiv version suffix
    return d.lower()


def biorxiv_id(doi: str | None) -> str | None:
    """'10.64898/2026.01.05.697760' -> '2026.01.05.697760' (None for non-bioRxiv DOIs)."""
    if not doi:
        return None
    prefix, _, suffix = doi.partition("/")
    m = _BIORXIV_SUFFIX.match(suffix)
    if prefix in BIORXIV_DOI_PREFIXES and m:
        return m.group(1)
    return None


def extract_ids(url: str | None) -> dict[str, str]:
    """Identifiers recoverable from a URL alone (no network)."""
    if not url:
        return {}
    u = unquote(unquote(url))
    ids: dict[str, str] = {}

    m = BIORXIV_RE.search(u)
    if m:
        ids["biorxiv"] = m.group(1)
    m = DOI_RE.search(u)
    if m:
        doi = clean_doi(m.group(1))
        if _SILVERCHAIR_PATH.search(urlparse(u).path):
            doi = re.sub(r"/\d{4,}(?:/.*)?$", "", doi)
        ids["doi"] = doi
        bid = biorxiv_id(ids["doi"])
        if bid:
            ids["biorxiv"] = bid
    elif "biorxiv" not in ids:
        m = NATURE_RE.search(u)
        if m:
            ids["doi"] = "10.1038/" + m.group(1).lower()
        m = ELIFE_RE.search(u)
        if m:
            ids["doi"] = "10.7554/elife." + m.group(1)

    m = PMID_RE.search(u)
    if m:
        ids["pmid"] = m.group(1)
    m = PMCID_RE.search(u)
    if m and ("ncbi.nlm.nih.gov" in u.lower() or "europepmc" in u.lower()):
        ids["pmcid"] = m.group(1).upper()
    m = ARXIV_RE.search(u)
    if m:
        ids["arxiv"] = m.group(1)
    return ids
