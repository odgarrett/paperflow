"""Resolve a pending paper (identifiers / title / links from emails) into full metadata.

Order of attempts:
  1. Get a DOI: from the links; arXiv id -> DataCite DOI; PMID/PMCID -> Europe PMC;
     following tracking redirects; (optionally) the landing page's <meta> tags;
     finally a title search (OpenAlex -> Crossref -> Europe PMC) with fuzzy matching
     that copes with Scholar's truncated titles and checks author surnames.
  2. With a DOI: bioRxiv API (preprints), OpenAlex, then Europe PMC / Crossref /
     Semantic Scholar until an abstract turns up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

from rapidfuzz import fuzz

from .articletypes import determine
from .apis import BioRxiv, Crossref, EuropePMC, OpenAlex, SemanticScholar, doi_from_landing_page
from .config import Config
from .http import Http, HttpError
from .links import biorxiv_id, extract_ids, is_preprint_doi
from .models import Meta
from .text import surname, title_key

log = logging.getLogger(__name__)

ACCEPT_SCORE = 92


@dataclass
class Resolution:
    paper_id: int
    meta: Meta
    status: str          # resolved | no_abstract | unmatched | error
    detail: str | None = None


def title_similarity(query: str, truncated: bool, candidate: str) -> float:
    qk, ck = title_key(query), title_key(candidate)
    if not qk or not ck:
        return 0.0
    if truncated:
        return fuzz.ratio(qk, ck[: len(qk)])
    return fuzz.ratio(qk, ck)


def score_candidate(item: dict, cand: Meta) -> float:
    if not cand.title:
        return 0.0
    score = title_similarity(item["title"], item["title_truncated"], cand.title)
    mine = set(item.get("surnames") or [])
    theirs = {s for s in (surname(a) for a in cand.authors or []) if s}
    overlap = bool(mine & theirs) if mine and theirs else None
    if overlap is True:
        score += 4
    elif overlap is False:
        score -= 8
    if item.get("year") and cand.year and abs(item["year"] - cand.year) > 1:
        score -= 5
    if item["title_truncated"] and len(title_key(item["title"])) < 40 and overlap is not True:
        score -= 10  # a short truncated title alone is weak evidence
    return score


class _Guarded:
    """Wraps an API client so a failing source is recorded instead of aborting the lookup."""

    def __init__(self, client, failures: list[str]):
        self._client, self._failures = client, failures

    def __getattr__(self, name):
        fn = getattr(self._client, name)

        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except HttpError as e:
                self._failures.append(str(e))
                return [] if name.startswith("search") else None
        return call


def _is_preprint(meta: Meta) -> bool:
    return bool(meta.is_preprint) or is_preprint_doi(meta.doi)


class Resolver:
    def __init__(self, http: Http, cfg: Config):
        self.http, self.cfg = http, cfg
        self.openalex = OpenAlex(http, cfg.contact_email, cfg.openalex_api_key)
        self.crossref = Crossref(http, cfg.contact_email)
        self.epmc = EuropePMC(http)
        self.biorxiv = BioRxiv(http)
        self.s2 = SemanticScholar(http, cfg.semantic_scholar_api_key) if cfg.use_semantic_scholar else None

    def resolve(self, item: dict) -> Resolution:
        try:
            return self._resolve(item)
        except HttpError as e:
            return Resolution(item["id"], Meta(), "error", str(e))
        except Exception as e:  # never let one odd record kill a long run
            log.debug("Unexpected error resolving %s", item["id"], exc_info=True)
            return Resolution(item["id"], Meta(), "error", f"{e.__class__.__name__}: {e}")

    # -- article type ----------------------------------------------------------------------

    def _add_article_type(self, api, meta: Meta) -> None:
        """PubMed's publication types are the best evidence; fetch them if not already here."""
        has_epmc = any(t.startswith("europepmc:") for t in meta.pub_types)
        if not has_epmc and not _is_preprint(meta) and (meta.pmid or meta.doi):
            meta.fill(api.epmc.by_pmid(meta.pmid) if meta.pmid else api.epmc.by_doi(meta.doi))
        meta.article_type, meta.article_type_source = determine(meta.pub_types, meta.title, meta.venue, meta.abstract)

    def lookup_type(self, paper: dict) -> tuple[Meta, list[str]]:
        """Article type for a paper already in the database (for `paperflow types`).

        `paper` has doi, pmid, title, venue, is_preprint. Mostly answered from cached responses.
        Returns (meta with article_type set, failures)."""
        failures: list[str] = []
        api = SimpleNamespace(
            openalex=_Guarded(self.openalex, failures), crossref=_Guarded(self.crossref, failures),
            epmc=_Guarded(self.epmc, failures), biorxiv=_Guarded(self.biorxiv, failures))
        meta = Meta(title=paper.get("title"), venue=paper.get("venue"), doi=paper.get("doi"), abstract=paper.get("abstract"),
                    pmid=paper.get("pmid"), is_preprint=True if paper.get("is_preprint") else None)
        bid = biorxiv_id(meta.doi)
        if bid:
            meta.fill(api.biorxiv.by_id(bid, preferred_doi=meta.doi))
        elif meta.doi or meta.pmid:
            if not _is_preprint(meta):
                meta.fill(api.epmc.by_pmid(meta.pmid) if meta.pmid else api.epmc.by_doi(meta.doi))
            if meta.doi and not any(t.startswith("europepmc:") for t in meta.pub_types):
                meta.fill(api.openalex.by_doi(meta.doi))
                if not meta.pub_types:
                    meta.fill(api.crossref.by_doi(meta.doi))
        meta.article_type, meta.article_type_source = determine(meta.pub_types, meta.title, meta.venue, meta.abstract)
        return meta, failures

    # -- steps -----------------------------------------------------------------------------

    def _via_pubmed(self, api, ids: dict, meta: Meta) -> None:
        m = None
        if ids.get("pmid"):
            m = api.epmc.by_pmid(ids["pmid"])
        elif ids.get("pmcid"):
            m = api.epmc.by_pmcid(ids["pmcid"])
        if m:
            meta.fill(m)
            if m.doi:
                ids["doi"] = m.doi

    def _search_title(self, api, item: dict) -> tuple[Meta | None, float, str | None]:
        key = title_key(item["title"])
        if len(key) < 12:
            return None, 0.0, None
        first_author = (item.get("surnames") or [None])[0]
        searches = (
            ("openalex", lambda: api.openalex.search_title(key)),
            ("crossref", lambda: api.crossref.search(item["title"], first_author)),
            ("europepmc", lambda: api.epmc.search_title(key)),
        )
        best_overall = (None, 0.0, None)
        for name, run in searches:
            scored = [(score_candidate(item, c), c) for c in run()]
            if not scored:
                continue
            score, cand = max(scored, key=lambda x: x[0])
            if score >= ACCEPT_SCORE:
                return cand, score, name
            if score > best_overall[1]:
                best_overall = (None, score, name)
        return best_overall

    def _resolve(self, item: dict) -> Resolution:
        ids = dict(item["ids"])
        meta = Meta()
        notes: list[str] = []
        failures: list[str] = []
        api = SimpleNamespace(
            openalex=_Guarded(self.openalex, failures), crossref=_Guarded(self.crossref, failures),
            epmc=_Guarded(self.epmc, failures), biorxiv=_Guarded(self.biorxiv, failures),
            s2=_Guarded(self.s2, failures) if self.s2 else None,
        )

        if not ids.get("doi") and ids.get("biorxiv"):
            m = api.biorxiv.by_id(ids["biorxiv"])
            if m:
                meta.fill(m)
                ids["doi"] = m.doi
        if not ids.get("doi") and ids.get("arxiv"):
            ids["doi"] = f"10.48550/arxiv.{ids['arxiv']}"
        if not ids.get("doi"):
            self._via_pubmed(api, ids, meta)

        # Title search (fast metadata APIs) before visiting any publisher website.
        if not ids.get("doi") and not meta.title and item.get("title"):
            hit, score, src = self._search_title(api, item)
            if hit:
                meta.fill(hit)
                notes.append(f"title_match:{src}:{min(score, 100):.0f}")
                if hit.doi:
                    ids["doi"] = hit.doi
            elif score:
                notes.append(f"best_title_score:{src}:{score:.0f}")

        # Last resort: follow the link (tracking redirects, publisher pages) looking for an id.
        if not ids.get("doi") and not meta.title and self.cfg.follow_redirects:
            for url in item.get("urls", [])[:2]:
                final = self.http.final_url(url)
                if final and final != url:
                    for k, v in extract_ids(final).items():
                        ids.setdefault(k, v)
                if not ids.get("doi") and self.cfg.scrape_landing_pages:
                    d = doi_from_landing_page(self.http, final or url)
                    if d:
                        ids["doi"] = d
                if ids.get("doi") or ids.get("pmid"):
                    notes.append("via_redirect")
                    break
            if not ids.get("doi"):
                self._via_pubmed(api, ids, meta)

        doi = ids.get("doi")
        if doi and not meta.title:
            # A DOI scraped from an unfamiliar URL layout may carry extra path segments.
            # If it resolves nowhere, retry shorter versions; failing that, search by title.
            found = None
            for cand in [doi] + [doi.rsplit("/", n)[0] for n in range(1, doi.count("/"))]:
                found = api.openalex.by_doi(cand) or api.crossref.by_doi(cand)
                if found:
                    if cand != doi:
                        notes.append(f"doi_trimmed:{doi}")
                    doi = ids["doi"] = cand
                    meta.fill(found)
                    break
            if not found and item.get("title") and not biorxiv_id(doi):
                hit, score, src = self._search_title(api, item)
                if hit:
                    notes.append(f"doi_unresolvable:{doi};title_match:{src}:{min(score, 100):.0f}")
                    meta.fill(hit)
                    doi = ids["doi"] = hit.doi or doi
        if doi:
            meta.doi = meta.doi or doi
            bid = biorxiv_id(doi) or ids.get("biorxiv")
            if bid and "biorxiv" not in meta.sources and "medrxiv" not in meta.sources:
                meta.fill(api.biorxiv.by_id(bid, preferred_doi=doi if biorxiv_id(doi) else None))
            meta.fill(api.openalex.by_doi(doi))
            if not meta.abstract:
                meta.fill(api.epmc.by_doi(doi))
            if not meta.abstract or not meta.title:
                meta.fill(api.crossref.by_doi(doi))
            if not meta.abstract and api.s2:
                meta.fill(api.s2.by_doi(doi))

        for k in ("pmid", "pmcid", "arxiv"):
            if ids.get(k) and not getattr(meta, k):
                setattr(meta, k, ids[k])

        self._add_article_type(api, meta)

        # Sanity check: does what we found match the title printed in the email?
        if item.get("title") and meta.title:
            sim = title_similarity(item["title"], item["title_truncated"], meta.title)
            if sim < 70:
                notes.append(f"title_mismatch:{sim:.0f}")

        if failures and not (meta.title and meta.abstract):
            # Something we might have needed was unreachable: keep it queued for a retry.
            notes.append("failed:" + ",".join(sorted({f.split(":")[0] for f in failures})))
            return Resolution(item["id"], meta, "error", ";".join(notes))
        if meta.title and meta.abstract:
            status = "resolved"
        elif meta.title:
            status = "no_abstract"
        else:
            status = "unmatched"
        return Resolution(item["id"], meta, status, ";".join(notes) or None)
