"""Turning briefing text into a script that sounds right when read aloud.

Two layers:
  1. built-in rules for things text-to-speech handles badly in scientific writing
     ("et al.", units such as µM and kDa, Greek letters, ±, ≥, ~, DOIs and URLs);
  2. your pronunciation list (pronunciations.txt next to config.yaml), for names the voice
     gets wrong. One entry per line:

         ZAR1 => zar one
         NbNRC4 => N B N R C four
         Magnaporthe => /mˈæɡnəpˌɔɹθi/

     The right-hand side is either a respelling, or phonemes between slashes (Kokoro's own
     notation; see its documentation). Matching is case-sensitive and whole-word, and longer
     entries win over shorter ones.
"""
from __future__ import annotations

import re
from pathlib import Path

GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta",
         "θ": "theta", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
         "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
         "Δ": "delta", "Ω": "omega", "Ψ": "psi"}

# Units: only after a standalone number ("24 h", "3.5 µM"), never inside names like "R2h".
UNITS = [("[µμ]M", "micromolar"), ("nM", "nanomolar"), ("mM", "millimolar"), ("[µμ]g", "micrograms"),
         ("[µμ][lL]", "microlitres"), ("[µμ]m", "micrometres"), ("nm", "nanometres"), ("kDa", "kilodaltons"),
         ("Da", "daltons"), ("Å", "angstroms"), ("°C", "degrees Celsius"), ("kb", "kilobases"),
         ("bp", "base pairs"), ("h", "hours"), ("min", "minutes"), ("dpi", "days post-inoculation")]
_NUM = r"(?<![\w.])(\d+(?:\.\d+)?)\s*"

# (pattern, replacement), applied in order
RULES = [
    (r"https?://[^\s)\]]+", ""),
    (r"\b(?:doi:\s*)?10\.\d{4,9}/[^\s)\]]+", ""),
    (r"\bet al\.?", "and colleagues"),
    (r"\be\.g\.,?", "for example,"),
    (r"\bi\.e\.,?", "that is,"),
    (r"\bvs\.?(?=\s)", "versus"),
    (r"\bcf\.", "compare"),
    (r"\bSupplementary\s+Fig(?:ure)?s?\.?\s*", "Supplementary Figure "),
    (r"\bFigs?\.\s*", "Figure "),
    *[(_NUM + unit + r"(?![\w])", r"\1 " + spoken) for unit, spoken in UNITS],
    (r"Å", "angstroms"),
    (r"\b[Pp]\s*<\s*", "p less than "), (r"\b[Pp]\s*=\s*", "p equals "),
    (r"±", " plus or minus "), (r"≥", " at least "), (r"≤", " at most "), (r"≈|~(?=\s?\d)", " roughly "),
    (r"×", " times "), (r"→", " to "), (r"\s<\s", " less than "), (r"\s>\s", " greater than "),
    (r"\[|\]", ""),
]


def load_lexicon(path: Path | None) -> list[tuple[str, str]]:
    if not path or not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        term, spoken = (x.strip() for x in line.split("=>", 1))
        if term and spoken:
            entries.append((term, spoken))
    return sorted(entries, key=lambda e: -len(e[0]))


def apply_lexicon(text: str, lexicon: list[tuple[str, str]]) -> str:
    for term, spoken in lexicon:
        if spoken.startswith("/") and spoken.endswith("/") and len(spoken) > 2:
            spoken = f"[{term}]({spoken})"          # Kokoro's custom-pronunciation notation
        text = re.sub(rf"(?<![\w\[]){re.escape(term)}(?![\w\]])", lambda m: spoken, text)
    return text


def speakable(text: str, lexicon: list[tuple[str, str]] | None = None) -> str:
    """Prepare prose for text-to-speech."""
    text = apply_lexicon(text, lexicon or [])           # first, so your entries see the original words
    protected = re.findall(r"\[[^\]]+\]\(/[^)]+/\)", text)
    for i, chunk in enumerate(protected):                # keep phoneme notation intact
        text = text.replace(chunk, f"\x00{i}\x00")
    for pattern, repl in RULES:
        text = re.sub(pattern, repl, text)
    for ch, name in GREEK.items():
        text = re.sub(rf"(?<=[A-Za-z0-9]){ch}", f" {name}", text)   # TNFα -> TNF alpha
        text = re.sub(rf"{ch}(?=[A-Za-z0-9])", f"{name} ", text)    # βsheet -> beta sheet
        text = text.replace(ch, name)                                # α-helix -> alpha-helix
    for i, chunk in enumerate(protected):
        text = text.replace(f"\x00{i}\x00", chunk)
    text = re.sub(r"\(\s*[;,]?\s*\)", "", text)
    text = re.sub(r"\(\s*[;,]?\s*", "(", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def spoken_authors(authors: list) -> str:
    if not authors:
        return ""
    first = str(authors[0])
    surname = first.split(",")[0] if "," in first else first.split()[-1]
    return f"{surname} and colleagues" if len(authors) > 1 else surname


LEXICON_TEMPLATE = """# Pronunciations for the audio version of briefings.
# One entry per line:  term => how to say it
# The right-hand side is a respelling, or phonemes between slashes (Kokoro's notation).
# Matching is case-sensitive and whole-word. Listen to a pilot briefing, then add the
# names the voice gets wrong. Examples (remove the # to use one):
#
# ZAR1 => zar one
# NbNRC4 => N B N R C four
# Pik-1 => pick one
# RGA5 => R G A five
# Magnaporthe => /mˈæɡnəpˌɔɹθi/
"""
