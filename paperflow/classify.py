"""Stage 2b: Claude reads each paper that passed the embedding filter and decides
mandatory / interest / skip against your research profile.

The prompt has a fixed part (instructions, your research_profile.md, and the titles of your
example papers) that is identical for every paper and is prompt-cached, and a small
per-paper part (title, authors, venue, abstract, which alerts found it, and the nearest
examples from the embedding stage).

Two backends, chosen by `triage.llm_provider`:
  ollama     a local model served by Ollama (free; runs on your GPU)
  anthropic  Claude through the Anthropic API (`pip install -e ".[llm]"`, API key in
             ANTHROPIC_API_KEY or the variable named in config.yaml; billed per token)
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .store import utcnow

log = logging.getLogger(__name__)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["mandatory", "interest", "skip"]},
        "projects": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "article_type": {"type": "string", "enum": ["research", "review", "commentary", "editorial",
                                                     "correction", "retraction", "other"]},
    },
    "required": ["label", "projects", "reason", "article_type"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You triage newly published papers for one researcher, whose research profile follows.
For each paper you are given, decide:

- "mandatory": the researcher should read this paper in full soon. It directly affects one of
  their active projects: it answers or challenges a question they are working on, uses or
  improves a method or system they depend on, or comes from a group working on the same
  problem. This should be rare: a handful of papers per week at most.
- "interest": worth knowing about (a short summary is enough), because it is relevant to their
  field, projects or methods, but doesn't demand a full read now.
- "skip": not relevant enough to mention.

Each paper's article type (research article, review, commentary, …) is given when known. It
comes from publication metadata and is sometimes wrong; recent reviews in particular are
often indexed as research articles. Judge the type yourself from the title and abstract.
Apply the researcher's own definitions and exclusions in the profile when they are given,
including preferences about article types, applied to the type you judge correct;
they take precedence over the general guidance above. When torn between two labels, choose
the lower one unless the paper clearly meets the higher bar. Judge only from the information
given; don't assume content the abstract doesn't state.

Respond with JSON: "label"; "projects": names of the researcher's projects (as written in the
profile) that the paper bears on, empty if none; "reason": one or two sentences naming the
specific connection to the profile (or the lack of one); "article_type": what kind of article
this is as you judge it (research, review, commentary, editorial, correction, retraction or
other), whether or not that matches the type given."""

MAX_EXAMPLE_TITLES = 60


@dataclass
class LlmStats:
    counts: Counter = field(default_factory=Counter)
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    errors: list = field(default_factory=list)
    types_corrected: list = field(default_factory=list)   # (paper id, old type, new type)


def build_system(profile_text: str, examples: dict[str, list[str]]) -> list[dict]:
    parts = [INSTRUCTIONS, "\n\n<research_profile>\n" + profile_text.strip() + "\n</research_profile>"]
    shown = {"mandatory": "Papers they marked as MANDATORY reading",
             "interest": "Papers they marked as OF INTEREST",
             "skip": "Papers they marked as SKIP (not relevant)"}
    for label, heading in shown.items():
        titles = examples.get(label) or []
        if titles:
            lines = "\n".join(f"- {t}" for t in titles[:MAX_EXAMPLE_TITLES])
            parts.append(f"\n\n<{label}_examples>\n{heading}:\n{lines}\n</{label}_examples>")
    return [{"type": "text", "text": "".join(parts), "cache_control": {"type": "ephemeral"}}]


def paper_message(p: dict) -> str:
    lines = [f"Title: {p['title']}"]
    if p.get("authors"):
        a = p["authors"]
        lines.append("Authors: " + (", ".join(a[:12]) + (f", … ({len(a)} authors)" if len(a) > 12 else "")))
    from .articletypes import DESCRIPTION
    if p.get("article_type") in DESCRIPTION:
        note = "" if p.get("article_type_source") == "you" else " (from metadata; may be wrong)"
        lines.append(f"Article type{note}: {DESCRIPTION[p['article_type']]}")
    venue = p.get("venue") or ""
    if p.get("is_preprint"):
        venue = (venue + " (preprint)").strip()
    if venue or p.get("year"):
        lines.append(f"Venue: {venue} {p.get('year') or ''}".strip())
    if p.get("abstract"):
        lines.append(f"Abstract: {p['abstract']}")
    elif p.get("snippet"):
        lines.append(f"(No abstract available; excerpt from the alert email:) {p['snippet']}")
    else:
        lines.append("(No abstract available.)")
    if p.get("alerts"):
        lines.append("Found by these alerts of theirs: " + "; ".join(p["alerts"]))
    if p.get("nearest"):
        near = "; ".join(f"[{n['label']}] {n['title']}" for n in p["nearest"] if n.get("title"))
        lines.append(f"Most similar papers they have labelled: {near}")
    return "\n".join(lines)


def _parse_verdict(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    v = json.loads(m.group(0) if m else text)
    if v.get("label") not in ("mandatory", "interest", "skip"):
        raise ValueError(f"unexpected label {v.get('label')!r}")
    v["projects"] = [str(x) for x in v.get("projects") or []]
    v["reason"] = str(v.get("reason") or "")
    if v.get("article_type") not in ("research", "review", "commentary", "editorial", "correction", "retraction", "other"):
        v["article_type"] = None
    return v


class AnthropicBackend:
    """Claude through the Anthropic API (needs an API key; billed per token)."""

    provider = "anthropic"

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY", client=None):
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise SystemExit('Using Claude needs the Anthropic SDK: pip install -e ".[llm]"')
            key = os.environ.get(api_key_env)
            if not key:
                raise SystemExit(f"Set your Anthropic API key in the environment variable {api_key_env}.")
            client = anthropic.Anthropic(api_key=key, max_retries=5)
        self.client, self.model = client, model
        self.structured = True  # JSON-schema structured output; falls back to plain JSON

    def check(self, prompt_chars: int) -> None:
        pass

    def fits(self, prompt_chars: int, max_tokens: int = 400) -> bool:
        return True

    def complete(self, system: list[dict], user: str, schema: dict | None = VERDICT_SCHEMA,
                 max_tokens: int = 400, temperature: float = 0.0) -> tuple[str, dict]:
        """One request. `schema` constrains the output to JSON; None gives free text."""
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, temperature=temperature,
                      messages=[{"role": "user", "content": user}])
        if self.structured and schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:
            if self.structured and "output_config" in str(e):
                log.warning("Model %s doesn't accept structured output; asking for plain JSON instead", self.model)
                self.structured = False
                return self.complete(system, user, schema, max_tokens, temperature)
            raise
        u = getattr(resp, "usage", None)
        usage = {
            "input": getattr(u, "input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "output": getattr(u, "output_tokens", 0) or 0,
        } if u is not None else {}
        usage["truncated"] = getattr(resp, "stop_reason", None) == "max_tokens"
        return "".join(getattr(b, "text", "") for b in resp.content), usage


class ModelUnavailable(RuntimeError):
    """The model server is down or keeps crashing: no point trying further papers now."""


CRASH_HINT = ("Ollama's model process keeps stopping, usually because the GPU or WSL ran out of memory. "
              "Check `journalctl -u ollama --since '1 hour ago' | grep -i -E 'error|memory|cuda'`, then try "
              "a smaller `ollama_num_ctx` (e.g. 12288) or `llm_model: qwen3:8b` in config.yaml. "
              "Everything done so far is saved; rerun the same command to continue.")


class OllamaBackend:
    """A local model served by Ollama (https://ollama.com). Free, private, runs on your GPU."""

    provider = "ollama"
    retry_wait = 10   # seconds; Ollama needs a moment to restart a crashed model process

    def __init__(self, model: str, url: str = "http://localhost:11434", num_ctx: int = 8192):
        import requests
        self.requests = requests
        self.model, self.url, self.num_ctx = model, url.rstrip("/"), num_ctx
        self.think_param = True  # ask thinking models (e.g. Qwen3) not to think; dropped if unsupported
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=5).json()
        except Exception:
            raise SystemExit(f"Can't reach Ollama at {self.url}. Is it running? Start it with `ollama serve` "
                             "(or check `ollama_url` in config.yaml).")
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            have = ", ".join(sorted(n for n in names if n)) or "none"
            raise SystemExit(f"Ollama doesn't have the model {model!r} yet. Download it with:\n"
                             f"    ollama pull {model}\n(models you have: {have})")

    def check(self, prompt_chars: int) -> None:
        est = prompt_chars // 3 + 500  # rough tokens, with room for the answer
        if est > self.num_ctx:
            raise SystemExit(
                f"The prompt (~{est:,} tokens) is longer than the model's context window ({self.num_ctx:,}), "
                "and Ollama would silently cut it. Raise `ollama_num_ctx` in config.yaml (uses more GPU "
                "memory), or shorten research_profile.md.")

    def fits(self, prompt_chars: int, max_tokens: int = 400) -> bool:
        return prompt_chars // 3 + max_tokens + 100 <= self.num_ctx

    def complete(self, system: list[dict], user: str, schema: dict | None = VERDICT_SCHEMA,
                 max_tokens: int = 400, temperature: float = 0.0) -> tuple[str, dict]:
        """One request. `schema` constrains the output to JSON; None gives free text."""
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": "".join(b["text"] for b in system)},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens},
            "keep_alive": "15m",
        }
        if schema is not None:
            body["format"] = schema            # constrains the output to this JSON shape
        if self.think_param:
            body["think"] = False
        import time
        for attempt in range(3):
            try:
                r = self.requests.post(f"{self.url}/api/chat", json=body, timeout=(10, 900))
            except self.requests.ConnectionError:
                if attempt == 2:
                    raise ModelUnavailable(f"Can't reach Ollama at {self.url} any more. {CRASH_HINT}")
                time.sleep(self.retry_wait * (attempt + 1))
                continue
            crashed = r.status_code >= 500 and any(
                k in r.text.lower() for k in ("runner", "unexpectedly stopped", "out of memory", "cuda"))
            if not crashed:
                break
            if attempt == 2:
                raise ModelUnavailable(f"Ollama: {r.text[:200]}. {CRASH_HINT}")
            log.warning("Ollama's model process stopped; waiting for it to restart (retry %d of 2)", attempt + 1)
            time.sleep(self.retry_wait * (attempt + 1))   # Ollama reloads the model on the next request
        if r.status_code == 400 and self.think_param and "think" in r.text.lower():
            self.think_param = False           # model doesn't know about thinking: retry without
            return self.complete(system, user, schema, max_tokens, temperature)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        text = re.sub(r"<think>.*?</think>", "", (j.get("message") or {}).get("content", ""), flags=re.S)
        return text, {"input": j.get("prompt_eval_count", 0) or 0, "output": j.get("eval_count", 0) or 0,
                      "truncated": j.get("done_reason") == "length"}   # hit the max_tokens cap


def unload(backend) -> None:
    """Free the GPU memory a local model holds (Ollama keeps models loaded for a while)."""
    if getattr(backend, "provider", None) != "ollama":
        return
    try:
        backend.requests.post(f"{backend.url}/api/generate", json={"model": backend.model, "keep_alive": 0},
                              timeout=(5, 60))
    except Exception:
        pass


def make_backend(cfg, model_override: str | None = None):
    model = model_override or cfg.llm_model
    if cfg.llm_provider == "anthropic":
        return AnthropicBackend(model, cfg.anthropic_api_key_env)
    if cfg.llm_provider == "ollama":
        return OllamaBackend(model, cfg.ollama_url, cfg.ollama_num_ctx)
    raise SystemExit(f"Unknown llm_provider {cfg.llm_provider!r} (use ollama or anthropic)")


class Classifier:
    def __init__(self, backend):
        self.backend = backend
        self.model = f"{backend.provider}:{backend.model}"
        self.stats = LlmStats()

    def classify(self, system: list[dict], paper: dict) -> dict:
        text, usage = self.backend.complete(system, paper_message(paper))
        self.stats.input_tokens += usage.get("input", 0)
        self.stats.cache_read_tokens += usage.get("cache_read", 0)
        self.stats.cache_write_tokens += usage.get("cache_write", 0)
        self.stats.output_tokens += usage.get("output", 0)
        return _parse_verdict(text)


# -- running it over the database ----------------------------------------------------------

def candidates(store, redo: bool = False, limit: int | None = None) -> list[dict]:
    """Papers that passed the embedding filter and still need an LLM verdict."""
    db = store.db
    q = ("SELECT p.*, a.detail AS embed_detail FROM papers p JOIN assessments a ON a.paper_id=p.id "
         "AND a.stage='embed' AND a.label='pass' "
         "WHERE p.id NOT IN (SELECT paper_id FROM judgments)")
    if not redo:
        q += " AND p.id NOT IN (SELECT paper_id FROM assessments WHERE stage='llm')"
    q += " ORDER BY a.score DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    out = []
    for r in db.execute(q).fetchall():
        sights = db.execute("SELECT DISTINCT alert_name, snippet FROM sightings WHERE paper_id=?", (r["id"],)).fetchall()
        detail = json.loads(r["embed_detail"] or "{}")
        out.append({
            "id": r["id"], "title": r["title"], "authors": json.loads(r["authors"]) if r["authors"] else None,
            "venue": r["venue"], "year": r["year"], "is_preprint": r["is_preprint"], "abstract": r["abstract"],
            "article_type": r["article_type"], "article_type_source": r["article_type_source"],
            "snippet": max((s["snippet"] for s in sights if s["snippet"]), key=len, default=None),
            "alerts": sorted({s["alert_name"] for s in sights if s["alert_name"]}),
            "nearest": detail.get("nearest"),
        })
    return out


def example_titles(store) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for label, title in store.db.execute("SELECT label, title FROM examples WHERE title IS NOT NULL ORDER BY key"):
        out.setdefault(label, []).append(title)
    for label, title in store.db.execute(
            "SELECT j.label, p.title FROM judgments j JOIN papers p ON p.id=j.paper_id "
            "WHERE j.source='manual' ORDER BY j.id DESC"):
        out.setdefault(label, []).append(title)
    return out


def run_classification(store, classifier: Classifier, profile_path: Path, items: list[dict],
                       workers: int = 4, progress: bool = True) -> LlmStats:
    from tqdm import tqdm
    profile = profile_path.read_text(encoding="utf-8")
    system = build_system(profile, example_titles(store))
    stats = classifier.stats
    if not items:
        return stats
    classifier.backend.check(len(system[0]["text"]) + max(len(paper_message(p)) for p in items))

    # One request first, so the cached prefix exists before the parallel requests start.
    def one(p):
        return p, classifier.classify(system, p)

    from .articletypes import is_weak

    def record(p, v):
        detail = {"projects": v["projects"], "reason": v["reason"], "model": classifier.model,
                  "article_type": v.get("article_type")}
        store.db.execute("INSERT OR REPLACE INTO assessments VALUES (?,?,?,?,?,?)",
                         (p["id"], "llm", v["label"], None, json.dumps(detail), utcnow()))
        stats.counts[v["label"]] += 1
        # The model read the abstract: let its reading replace weak metadata evidence.
        judged = v.get("article_type")
        if (judged and p.get("abstract") and judged != p.get("article_type")
                and is_weak(p.get("article_type"), p.get("article_type_source"))):
            store.set_article_type(p["id"], judged, "llm", None)
            stats.types_corrected.append((p["id"], p.get("article_type"), judged))

    bar = tqdm(total=len(items), unit="paper", desc="classifying", disable=not progress, dynamic_ncols=True)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        first, rest = items[0], items[1:]
        try:
            record(*one(first))
        except Exception as e:
            if "authentication" in str(e).lower() or "api key" in str(e).lower() or "401" in str(e):
                raise SystemExit(f"The Anthropic API rejected the request: {e}")
            stats.errors.append((first["id"], str(e)))
        bar.update(1)
        futures = {pool.submit(one, p): p for p in rest}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                record(*fut.result())
            except Exception as e:
                stats.errors.append((p["id"], str(e)))
            bar.update(1)
            bar.set_postfix(dict(stats.counts), refresh=False)
            if i % 20 == 0:
                store.db.commit()
    except KeyboardInterrupt:
        print("\nStopping: cancelling queued requests and saving the verdicts received so far…")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        bar.close()
        store.db.commit()
    return stats
