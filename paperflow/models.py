"""Plain data containers passed between parsers, resolver and store."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sighting:
    """One appearance of a paper in one alert email, as parsed from the email itself."""

    source: str                      # scholar | biorxiv | stork | generic
    title: str | None = None
    title_truncated: bool = False
    authors_raw: str | None = None
    venue_raw: str | None = None
    year: int | None = None
    url: str | None = None
    snippet: str | None = None
    ids: dict = field(default_factory=dict)   # {"doi", "pmid", "pmcid", "arxiv", "biorxiv"}
    alert: str | None = None         # section within the email (bioRxiv collection, Stork keyword)
    alert_kind: str | None = None    # overrides the email-level kind for this item
    posted: str | None = None        # ISO date printed in the email, if any


@dataclass
class ParsedEmail:
    message_id: str
    source: str
    sender: str
    subject: str
    received_at: str | None
    alert_name: str | None
    alert_kind: str | None           # keyword | author | citation | related | None
    parser: str
    sightings: list
    n_reported: int | None = None    # results the alert says it found (may exceed what it shows)


META_FIELDS = (
    "title", "authors", "year", "pub_date", "venue", "doi", "published_doi", "abstract",
    "oa_pdf_url", "landing_url", "work_type", "is_preprint", "pmid", "pmcid", "arxiv",
)


@dataclass
class Meta:
    """Bibliographic metadata assembled from one or more APIs."""

    title: str | None = None
    authors: list | None = None
    year: int | None = None
    pub_date: str | None = None
    venue: str | None = None
    doi: str | None = None
    published_doi: str | None = None   # for preprints that have since been published
    abstract: str | None = None
    abstract_source: str | None = None
    oa_pdf_url: str | None = None
    landing_url: str | None = None
    work_type: str | None = None
    is_preprint: bool | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv: str | None = None
    sources: list = field(default_factory=list)
    pub_types: list = field(default_factory=list)   # "source:Type" from every source consulted
    related: list = field(default_factory=list)     # PubMed links: {"type": "Comment on", "pmid": …}
    article_type: str | None = None                 # decided from pub_types + title/venue
    article_type_source: str | None = None

    def fill(self, other: "Meta | None") -> "Meta":
        """Fill empty fields from another record (earlier sources take priority)."""
        if other is None:
            return self
        for f in META_FIELDS:
            mine, theirs = getattr(self, f), getattr(other, f)
            if mine in (None, "", []) and theirs not in (None, "", []):
                setattr(self, f, theirs)
                if f == "abstract":
                    self.abstract_source = other.abstract_source
        for s in other.sources:
            if s not in self.sources:
                self.sources.append(s)
        for t in other.pub_types:
            if t not in self.pub_types:
                self.pub_types.append(t)
        for r in other.related:
            if r not in self.related:
                self.related.append(r)
        return self

    def identifiers(self) -> list[tuple[str, str]]:
        out = []
        from .links import biorxiv_id
        for attr, scheme in (("doi", "doi"), ("published_doi", "doi"), ("pmid", "pmid"),
                             ("pmcid", "pmcid"), ("arxiv", "arxiv")):
            v = getattr(self, attr)
            if v:
                out.append((scheme, v))
        bid = biorxiv_id(self.doi)
        if bid:
            out.append(("biorxiv", bid))
        return out
