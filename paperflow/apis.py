"""Clients for the scholarly metadata APIs. Each returns Meta objects (or None).

Coverage notes:
  bioRxiv API    - authoritative for bioRxiv/medRxiv preprints, always has abstracts,
                   and reports the DOI of the published version when there is one.
  Europe PMC     - best abstract coverage for life sciences (PubMed + preprints).
  OpenAlex       - broadest coverage, open-access PDF locations; abstracts for many works.
  Crossref       - authoritative DOI metadata; abstracts only when publishers deposit them.
  Semantic Scholar - last-resort abstract source (heavily rate-limited without a key).
"""
from __future__ import annotations

import re
from urllib.parse import quote

from bs4 import BeautifulSoup

from .http import Http
from .links import BIORXIV_DOI_PREFIXES, DOI_RE, biorxiv_id, clean_doi
from .models import Meta
from .text import clean_abstract, squash, strip_tags


def _doi(u: str | None) -> str | None:
    if not u:
        return None
    return clean_doi(re.sub(r"^https?://(?:dx\.)?doi\.org/", "", u, flags=re.I))


def _reconstruct(inv: dict | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not inv:
        return None
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return squash(" ".join(w for _, w in pos)) or None


class OpenAlex:
    BASE = "https://api.openalex.org"

    def __init__(self, http: Http, email: str, api_key: str | None = None):
        self.http, self.email, self.api_key = http, email, api_key

    def _params(self, extra: dict | None = None) -> dict:
        p = {"mailto": self.email}
        if self.api_key:
            p["api_key"] = self.api_key
        p.update(extra or {})
        return p

    def by_doi(self, doi: str) -> Meta | None:
        j = self.http.get_json(f"{self.BASE}/works/doi:{quote(doi, safe='/()')}", self._params())
        return self._meta(j) if j else None

    def search_title(self, key: str, n: int = 5) -> list[Meta]:
        # `key` is already reduced to [a-z0-9 ] so it can't break the filter syntax.
        j = self.http.get_json(f"{self.BASE}/works",
                               self._params({"filter": f"title.search:{key}", "per-page": n}))
        return [self._meta(w) for w in (j or {}).get("results", [])]

    @staticmethod
    def _meta(w: dict) -> Meta:
        ids = w.get("ids") or {}
        pmid = re.sub(r"\D", "", ids["pmid"]) if ids.get("pmid") else None
        m = re.search(r"PMC\d+", ids.get("pmcid") or "")
        loc = w.get("primary_location") or {}
        best = w.get("best_oa_location") or {}
        abstract = _reconstruct(w.get("abstract_inverted_index"))
        return Meta(
            title=strip_tags(w.get("display_name") or w.get("title")),
            authors=[a["author"]["display_name"] for a in w.get("authorships") or []
                     if (a.get("author") or {}).get("display_name")],
            year=w.get("publication_year"),
            pub_date=w.get("publication_date"),
            venue=(loc.get("source") or {}).get("display_name"),
            doi=_doi(w.get("doi")),
            abstract=abstract,
            abstract_source="openalex" if abstract else None,
            oa_pdf_url=best.get("pdf_url") or (w.get("open_access") or {}).get("oa_url"),
            landing_url=loc.get("landing_page_url"),
            work_type=w.get("type"),
            is_preprint=True if w.get("type") == "preprint" else None,
            pmid=pmid or None,
            pmcid=m.group(0) if m else None,
            sources=["openalex"],
            pub_types=[f"openalex:{w['type']}"] if w.get("type") else [],
        )


class Crossref:
    BASE = "https://api.crossref.org"

    def __init__(self, http: Http, email: str):
        self.http, self.email = http, email

    def by_doi(self, doi: str) -> Meta | None:
        j = self.http.get_json(f"{self.BASE}/works/{quote(doi, safe='/()')}", {"mailto": self.email})
        return self._meta(j["message"]) if j and "message" in j else None

    def search(self, title: str, author: str | None = None, n: int = 5) -> list[Meta]:
        q = title + (f" {author}" if author else "")
        j = self.http.get_json(f"{self.BASE}/works",
                               {"query.bibliographic": q, "rows": n, "mailto": self.email})
        items = ((j or {}).get("message") or {}).get("items") or []
        return [self._meta(m) for m in items]

    @staticmethod
    def _meta(m: dict) -> Meta:
        parts = None
        for k in ("published", "issued", "posted", "created"):
            dp = ((m.get(k) or {}).get("date-parts") or [[None]])[0]
            if dp and dp[0]:
                parts = dp
                break
        pub_date = "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(parts)) if parts else None
        authors = []
        for a in m.get("author") or []:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name")
            if name:
                authors.append(name)
        venue = (m.get("container-title") or [None])[0] or ((m.get("institution") or [{}])[0]).get("name")
        published = None
        for rel in (m.get("relation") or {}).get("is-preprint-of") or []:
            if rel.get("id-type") == "doi":
                published = clean_doi(rel["id"])
        is_pre = True if m.get("type") == "posted-content" and m.get("subtype") == "preprint" else None
        abstract = clean_abstract(m.get("abstract"))
        return Meta(
            title=strip_tags((m.get("title") or [None])[0]),
            authors=authors or None,
            year=parts[0] if parts else None,
            pub_date=pub_date,
            venue=venue,
            doi=_doi(m.get("DOI")),
            published_doi=published,
            abstract=abstract,
            abstract_source="crossref" if abstract else None,
            landing_url=m.get("URL"),
            work_type=m.get("type"),
            is_preprint=is_pre,
            sources=["crossref"],
            pub_types=[f"crossref:{m['type']}"] if m.get("type") else [],
        )


class EuropePMC:
    URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, http: Http):
        self.http = http

    def query(self, q: str, n: int = 1) -> list[Meta]:
        j = self.http.get_json(self.URL, {"query": q, "resultType": "core", "format": "json", "pageSize": n})
        results = ((j or {}).get("resultList") or {}).get("result") or []
        return [self._meta(r) for r in results]

    def _first(self, q: str) -> Meta | None:
        r = self.query(q)
        return r[0] if r else None

    def by_doi(self, doi: str) -> Meta | None:
        return self._first(f'DOI:"{doi}"')

    def by_pmid(self, pmid: str) -> Meta | None:
        return self._first(f"EXT_ID:{pmid} AND SRC:MED")

    def by_pmcid(self, pmcid: str) -> Meta | None:
        return self._first(f"PMCID:{pmcid}")

    def search_title(self, key: str, n: int = 5) -> list[Meta]:
        return self.query(f'TITLE:"{key}"', n)

    @staticmethod
    def _meta(r: dict) -> Meta:
        authors = [a.get("fullName") for a in ((r.get("authorList") or {}).get("author") or [])
                   if a.get("fullName")]
        if not authors and r.get("authorString"):
            authors = [a.strip(" .") for a in r["authorString"].split(",") if a.strip(" .")]
        venue = ((r.get("journalInfo") or {}).get("journal") or {}).get("title")
        is_pre = r.get("source") == "PPR"
        if is_pre and not venue:
            venue = (r.get("bookOrReportDetails") or {}).get("publisher")
        abstract = clean_abstract(r.get("abstractText"))
        year = r.get("pubYear")
        src, rid = r.get("source"), r.get("id")
        # Publication types are only meaningful for PubMed/PMC records, not preprint (PPR) ones.
        pub_types = []
        if src in ("MED", "PMC"):
            types = (r.get("pubTypeList") or {}).get("pubType") or []
            if isinstance(types, str):
                types = [types]
            pub_types = [f"europepmc:{t}" for t in types if isinstance(t, str)] or ["europepmc:Journal Article"]
        related = []
        cc = (r.get("commentCorrectionList") or {}).get("commentCorrection") or []
        for c in cc if isinstance(cc, list) else [cc]:
            if isinstance(c, dict) and c.get("type") and c.get("id"):
                related.append({"type": c["type"], "id": str(c["id"]), "source": c.get("source")})
        return Meta(
            title=strip_tags(r.get("title")),
            authors=authors or None,
            year=int(year) if year and str(year).isdigit() else None,
            pub_date=r.get("firstPublicationDate"),
            venue=venue,
            doi=_doi(r.get("doi")),
            abstract=abstract,
            abstract_source="europepmc" if abstract else None,
            landing_url=f"https://europepmc.org/article/{src}/{rid}" if src and rid else None,
            is_preprint=True if is_pre else None,
            pmid=r.get("pmid"),
            pmcid=r.get("pmcid"),
            sources=["europepmc"],
            pub_types=pub_types,
            related=related,
        )


class BioRxiv:
    def __init__(self, http: Http):
        self.http = http

    def by_doi(self, doi: str) -> Meta | None:
        for server in ("biorxiv", "medrxiv"):
            j = self.http.get_json(f"https://api.biorxiv.org/details/{server}/{doi}")
            coll = (j or {}).get("collection") or []
            if coll:
                return self._meta(coll, server, doi)
        return None

    def by_id(self, bid: str, preferred_doi: str | None = None) -> Meta | None:
        """Look up a preprint by its bare id under either DOI prefix (10.64898 / 10.1101)."""
        dois = [preferred_doi] if preferred_doi else []
        dois += [f"{p}/{bid}" for p in BIORXIV_DOI_PREFIXES if f"{p}/{bid}" not in dois]
        for doi in dois:
            m = self.by_doi(doi)
            if m:
                return m
        return None

    @staticmethod
    def _meta(coll: list, server: str, doi: str) -> Meta:
        coll = sorted(coll, key=lambda r: int(r.get("version") or 0))
        latest, first = coll[-1], coll[0]
        published = (latest.get("published") or "").strip()
        published = clean_doi(published) if published and published.upper() != "NA" else None
        ver = latest.get("version") or "1"
        date = first.get("date")
        abstract = clean_abstract(latest.get("abstract"))
        base = f"https://www.{server}.org/content/{doi}v{ver}"
        return Meta(
            title=strip_tags(latest.get("title")),
            authors=[squash(a) for a in (latest.get("authors") or "").split(";") if a.strip()] or None,
            year=int(date[:4]) if date else None,
            pub_date=date,
            venue="medRxiv" if server == "medrxiv" else "bioRxiv",
            doi=clean_doi(latest.get("doi") or doi),
            published_doi=published,
            abstract=abstract,
            abstract_source=server if abstract else None,
            landing_url=base,
            oa_pdf_url=base + ".full.pdf",
            work_type="preprint",
            is_preprint=True,
            sources=[server],
            pub_types=[f"{server}:{latest.get('type') or 'preprint'}"],
        )


class SemanticScholar:
    BASE = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "title,abstract,authors,year,venue,externalIds,openAccessPdf,publicationDate,publicationTypes"

    def __init__(self, http: Http, api_key: str | None = None):
        self.http, self.api_key = http, api_key

    def by_doi(self, doi: str) -> Meta | None:
        headers = {"x-api-key": self.api_key} if self.api_key else None
        j = self.http.get_json(f"{self.BASE}/DOI:{quote(doi, safe='/()')}", {"fields": self.FIELDS},
                               headers=headers)
        if not j:
            return None
        ext = j.get("externalIds") or {}
        abstract = clean_abstract(j.get("abstract"))
        pmc = ext.get("PubMedCentral")
        return Meta(
            title=strip_tags(j.get("title")),
            authors=[a["name"] for a in j.get("authors") or [] if a.get("name")] or None,
            year=j.get("year"),
            pub_date=j.get("publicationDate"),
            venue=j.get("venue") or None,
            doi=_doi(ext.get("DOI")) or doi,
            abstract=abstract,
            abstract_source="semanticscholar" if abstract else None,
            oa_pdf_url=(j.get("openAccessPdf") or {}).get("url") or None,
            pmid=str(ext["PubMed"]) if ext.get("PubMed") else None,
            pmcid=f"PMC{pmc}" if pmc and not str(pmc).startswith("PMC") else pmc,
            arxiv=ext.get("ArXiv"),
            sources=["semanticscholar"],
            pub_types=[f"semanticscholar:{t}" for t in (j.get("publicationTypes") or []) if isinstance(t, str)],
        )


_DOI_META_NAMES = {"citation_doi", "dc.identifier", "dc.identifier.doi", "prism.doi",
                   "bepress_citation_doi", "doi"}


def doi_from_landing_page(http: Http, url: str) -> str | None:
    """Read the DOI from a publisher page's <meta> tags (opt-in; many sites block scripts)."""
    page = http.get_html(url)
    if not page:
        return None
    soup = BeautifulSoup(page, "lxml")
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").lower()
        if name in _DOI_META_NAMES:
            m = DOI_RE.search(tag.get("content") or "")
            if m:
                return clean_doi(m.group(1))
    return None
