"""Files created by `paperflow init-triage`."""

PROFILE_TEMPLATE = """# Research profile

<!--
The LLM stage reads this whole file for every paper it judges, so write it for a smart
colleague from a neighbouring field: concrete names beat broad areas. "NLR integrated
domains as engineering scaffolds" says far more than "plant immunity". Markdown is fine;
these comments can be deleted. Keep it to a page or two.
-->

## Who I am

<!-- One or two sentences: field, lab, career stage, what you're aiming at. -->

## Active projects

<!--
One block per project. The project names you use here are what the triage labels papers
with, and what the digests will group by, so keep them short and stable.
-->

### Project: <short name>
- Question:
- Systems / organisms / proteins / genes:
- Methods I use or am developing:
- Groups working on the same problem:
- A paper is MANDATORY for this project if:

### Project: <short name>
- Question:
- Systems / organisms / proteins / genes:
- Methods I use or am developing:
- Groups working on the same problem:
- A paper is MANDATORY for this project if:

## Broader interests

<!-- Things you like to keep up with but aren't working on: worth a summary ("interest"). -->

## Not interested in

<!--
Be specific; this is where most of the noise gets removed. For example: medical or
mammalian applications of protein engineering, metagenomics, LLMs outside biology…
-->

## Article types

<!--
Each paper's type is known (research article, review, commentary such as previews or
news & views, editorial, correction…). Say how each should be treated, e.g.:
- Reviews: interest if on one of my projects; mandatory only if it's the first thorough
  review of a project topic; otherwise skip.
- Commentaries (previews, news & views): skip; I'd rather see the paper they discuss.
- Corrections, retractions, editorials: skip, unless they concern a paper central to a project.
-->

## People and labs I follow

<!-- Names whose new papers are at least "interest" whatever the topic. Note namesakes to avoid. -->
"""

EXAMPLES_README = """Put example papers for this label in this folder (subfolders are fine).

Accepted:
  .bib   Zotero: right-click a collection > Export Collection… > BibTeX (includes abstracts)
  .ris   RIS export
  .json  CSL JSON export
  .pdf   the paper itself (DOI/title are read from it, the rest is looked up)
  .txt   one DOI or title per line
  .md    notes; every DOI in the file is used (e.g. Obsidian literature notes)

Then run `paperflow examples` to check what was found.
"""

LABEL_GUIDANCE = {
    "mandatory": "Papers you'd want to read in full: core to your projects.",
    "interest": "Papers worth a summary: relevant field, methods, or people.",
    "skip": ("Optional, but powerful: papers your alerts bring that you DON'T want, especially "
             "near-misses (right keywords, wrong field). Plain noise needs no examples."),
}
