"""Decide what kind of article a paper is, from the publication types reported by the
metadata sources plus title and journal-name patterns.

Types: research, review, commentary (previews, news & views, comments, highlights,
perspectives), editorial, correction, retraction, other (book chapters, datasets, peer
reviews…), unknown. Whether something is a preprint is recorded separately (is_preprint).

Precedence:
  1. Correction and retraction notices, recognised from their (unambiguous) titles.
  2. PubMed's specific publication types (Review, Comment, Erratum…; via Europe PMC), and
     Crossref record types that aren't articles at all (peer review report, book chapter…).
  3. The paper describing itself: "In this review…", "Here we summarize recent advances…",
     "In this perspective…" in the abstract; title patterns ("…: a systematic review",
     "Reply to: …"); review-only journals ("Annual Review of…", "Trends in…").
  4. PubMed lists the paper with no special type: research. (Ranked below 3 because new
     reviews often enter PubMed as plain "Journal Article" until indexing catches up.)
  5. Specific types from Semantic Scholar, then OpenAlex (whose "review" is partly inferred).
  6. Otherwise research, if any source calls it an article or preprint; else unknown.
Two later corrections: the LLM stage reports the type it judges from the abstract and that
replaces weak evidence (generic research, unknown, title/journal patterns), and
`paperflow set-type` records your own, which nothing overrides.
"""
from __future__ import annotations

import re

TYPES = ("research", "review", "commentary", "editorial", "correction", "retraction", "other", "unknown")

# Most specific first: if a record lists several, the earliest in this list wins.
_PRIORITY = ("retraction", "correction", "review", "commentary", "editorial", "other", "research")

_EPMC = {
    "retraction of publication": "retraction",
    "published erratum": "correction", "erratum": "correction", "corrected and republished article": "correction",
    "review": "review", "systematic review": "review", "scoping review": "review", "meta-analysis": "review",
    "comment": "commentary", "news": "commentary", "newspaper article": "commentary", "interview": "commentary",
    "personal narrative": "commentary", "biography": "commentary", "lecture": "commentary",
    "editorial": "editorial", "introductory journal article": "editorial",
    "congress": "other", "dataset": "other", "patient education handout": "other", "guideline": "other",
    "practice guideline": "other", "consensus development conference": "other",
}
_S2 = {"review": "review", "editorial": "editorial", "news": "commentary", "lettersandcomments": "commentary",
       "book": "other", "booksection": "other", "dataset": "other",
       "journalarticle": "research", "conference": "research", "casereport": "research", "study": "research",
       "clinicaltrial": "research", "metaanalysis": "review"}
_OPENALEX = {"review": "review", "editorial": "editorial", "erratum": "correction", "retraction": "retraction",
             "letter": "commentary", "paratext": "other", "book-chapter": "other", "book": "other",
             "dataset": "other", "peer-review": "other", "report": "other", "dissertation": "other",
             "supplementary-materials": "other", "libguides": "other",
             "article": "research", "preprint": "research"}
_CROSSREF = {"peer-review": "other", "book-chapter": "other", "book": "other", "dataset": "other",
             "component": "other", "journal-article": "research", "posted-content": "research",
             "proceedings-article": "research"}

_TITLE_RETRACTION = re.compile(r"^(?:retraction(?: note)?|retracted|notice of retraction|expression of concern)\b", re.I)
_TITLE_CORRECTION = re.compile(r"^(?:correction|erratum|corrigendum|author correction|publisher correction|addendum)\b", re.I)
_TITLE_COMMENTARY = re.compile(r"^(?:(?:reply|response) to|comments? on|re)\s*:|^(?:(?:reply|response) to|comments? on)\s", re.I)
_TITLE_REVIEW = re.compile(
    r"\b(?:a (?:systematic |comprehensive |critical |narrative |brief |short |mini-?)?review|mini-?review|"
    r"review article|systematic review|scoping review|meta-analysis)\b"
    r"|:\s*(?:an? )?(?:overview|review)\b|^(?:review|overview)\b", re.I)
_VENUE_REVIEW = re.compile(
    r"^(?:annual review of|trends in|current opinion in|nature reviews|critical reviews in|"
    r"(?:chemical|physiological|pharmacological|fems microbiology|microbiology and molecular biology) reviews)"
    r"|\breviews?$", re.I)


def _pick(found: list[str]) -> str | None:
    for t in _PRIORITY:
        if t in found:
            return t
    return None


_ABS_REVIEW = re.compile(
    r"\b(?:in )?(?:this|the present|the current) (?:mini-?|short |brief |critical |comprehensive |systematic |"
    r"narrative |focused )?review\b"
    r"|\bhere,? we (?:review|overview|summari[sz]e(?! (?:our|the) (?:findings|results|data))|"
    r"discuss (?:recent|current)|highlight (?:recent|current)|provide an? (?:overview|summary|comprehensive|critical))"
    r"|\bwe (?:review|summari[sz]e) (?:the |recent |current |key |major )(?!(?:findings|results|data)\b)"
    r"|\b(?:this|the present) (?:article|paper|chapter) (?:reviews|summari[sz]es|provides an? (?:overview|summary))"
    r"|\breview(?:s|ing)? (?:the )?(?:recent|current|latest) (?:advances|progress|developments|literature|"
    r"knowledge|findings|understanding|insights)", re.I)
_ABS_COMMENTARY = re.compile(
    r"\b(?:in )?this (?:perspective|commentary|opinion piece|viewpoint|essay)\b"
    r"|\bhere,? we (?:offer|provide|present|share) (?:a |our )?(?:perspective|commentary|opinion)", re.I)


def determine(pub_types: list[str], title: str | None, venue: str | None,
              abstract: str | None = None) -> tuple[str, str]:
    """Return (article_type, source) from 'source:Type' strings plus title, venue and abstract."""
    by_source: dict[str, list[str]] = {}
    for entry in pub_types or []:
        src, _, raw = entry.partition(":")
        by_source.setdefault(src, []).append(raw.strip())
    title = (title or "").strip()
    abstract = abstract or ""

    # Unambiguous notices
    if _TITLE_RETRACTION.search(title):
        return "retraction", "title"
    if _TITLE_CORRECTION.search(title):
        return "correction", "title"

    # PubMed's specific types (Review, Comment, Erratum…), assigned by its indexers
    epmc = by_source.get("europepmc", [])
    epmc_type = _pick([_EPMC[t.lower()] for t in epmc if t.lower() in _EPMC])
    if epmc_type:
        return epmc_type, "europepmc"
    # Crossref's record type, when it says the record isn't an article at all (peer review
    # report, book chapter, dataset…), describes the record itself and is reliable.
    cr = _pick([_CROSSREF[t.lower()] for t in by_source.get("crossref", []) if t.lower() in _CROSSREF])
    if cr == "other":
        return cr, "crossref"

    # The paper describing itself. Beats generic labels: new reviews often enter PubMed as
    # plain "Journal Article" until indexing catches up, and preprint servers don't say.
    if _ABS_REVIEW.search(abstract):
        return "review", "abstract"
    if _ABS_COMMENTARY.search(abstract):
        return "commentary", "abstract"
    if _TITLE_COMMENTARY.search(title):
        return "commentary", "title"
    if _TITLE_REVIEW.search(title):
        return "review", "title"
    if venue and _VENUE_REVIEW.search(venue.strip()):
        return "review", "journal"

    # PubMed has the paper and lists nothing special
    if epmc:
        return "research", "europepmc"

    # Other sources' specific types (OpenAlex's "review" is partly inferred, so it ranks here)
    s2 = _pick([_S2[re.sub(r"\W", "", t).lower()] for t in by_source.get("semanticscholar", [])
                if re.sub(r"\W", "", t).lower() in _S2])
    if s2 and s2 != "research":
        return s2, "semanticscholar"
    oa = _pick([_OPENALEX[t.lower()] for t in by_source.get("openalex", []) if t.lower() in _OPENALEX])
    if oa and oa != "research":
        return oa, "openalex"

    # Generic
    for t, src in ((s2, "semanticscholar"), (oa, "openalex"), (cr, "crossref")):
        if t == "research":
            return "research", src
    if "biorxiv" in by_source or "medrxiv" in by_source:
        return "research", "preprint server"
    return "unknown", "none"


# Evidence weak enough that the LLM, having read the abstract, may override it.
WEAK_SOURCES = ("title", "journal", "none", "llm")


def is_weak(article_type: str | None, source: str | None) -> bool:
    if source == "you":
        return False
    return article_type in (None, "research", "unknown") or source in WEAK_SOURCES


# How each type is described to the LLM and in exports.
DESCRIPTION = {
    "research": "primary research article",
    "review": "review article",
    "commentary": "commentary on other work (e.g. a preview, news & views, comment or perspective)",
    "editorial": "editorial",
    "correction": "correction notice for another article",
    "retraction": "retraction notice",
    "other": "other (e.g. book chapter, dataset or peer review report)",
}
