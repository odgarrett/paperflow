"""Configuration loading. Secrets come from environment variables, never the YAML file."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_SENDERS = {
    "scholar": ["scholaralerts-noreply@google.com"],
    "biorxiv": ["biorxiv", "medrxiv", "openrxiv", "cshljnls", "highwire"],
    "stork": ["stork"],
}


DEFAULT_MODEL = {"ollama": "qwen3:14b", "anthropic": "claude-sonnet-5-5"}
# A local GPU handles one request at a time efficiently; the API handles several.
DEFAULT_WORKERS = {"ollama": 1, "anthropic": 4}


@dataclass
class Config:
    contact_email: str = "anonymous@example.org"
    database: Path = Path("data/paperflow.sqlite")
    cache: Path = Path("data/http_cache.sqlite")
    imap: dict = field(default_factory=dict)
    senders: dict = field(default_factory=lambda: dict(DEFAULT_SENDERS))
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    use_semantic_scholar: bool = True
    workers: int = 4
    follow_redirects: bool = True
    scrape_landing_pages: bool = False
    cache_days: int = 30
    fulltext_dir: Path = Path("fulltext")   # drop folder for PDFs you have
    outlook_csv: Path = Path("paper_alerts.CSV")   # where you save the Outlook export (for `weekly`)
    # briefings (stage 3)
    briefings_dir: Path = Path("briefings")
    tts: str = "kokoro"                      # kokoro | none
    voice: str = "af_heart"
    speed: float = 1.0
    audio_format: str = "mp3"                # mp3 | m4b (audiobook format; some players show chapters better)
    pronunciations: Path = Path("pronunciations.txt")
    # triage (stage 2)
    examples_dir: Path = Path("examples")
    profile: Path = Path("research_profile.md")
    embedding: str = "specter2"
    keep_recall: float = 0.95
    skip_margin: float = 0.02
    llm_provider: str = "ollama"            # ollama (local) | anthropic (Claude API)
    llm_model: str = "qwen3:14b"
    llm_workers: int = 1
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    ollama_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192


def load_config(path: Path | str, required: bool = True) -> Config:
    path = Path(path)
    if not path.exists():
        if required:
            raise SystemExit(f"Config file not found: {path} (copy config.example.yaml to config.yaml)")
        return Config()

    data = yaml.safe_load(path.read_text()) or {}
    base = path.resolve().parent
    tri = data.get("triage") or {}
    br = data.get("briefing") or {}
    provider = str(tri.get("llm_provider", "ollama")).lower()
    if provider not in DEFAULT_MODEL:
        raise SystemExit(f"triage.llm_provider must be 'ollama' or 'anthropic', not {provider!r}")
    apis = data.get("apis") or {}
    res = data.get("resolve") or {}
    senders = dict(DEFAULT_SENDERS)
    senders.update(data.get("senders") or {})

    cfg = Config(
        contact_email=data.get("contact_email") or "anonymous@example.org",
        database=(base / data.get("database", "data/paperflow.sqlite")).resolve(),
        cache=(base / data.get("cache", "data/http_cache.sqlite")).resolve(),
        imap=data.get("imap") or {},
        senders=senders,
        openalex_api_key=os.environ.get(apis.get("openalex_api_key_env", "OPENALEX_API_KEY")),
        semantic_scholar_api_key=os.environ.get(apis.get("semantic_scholar_api_key_env", "S2_API_KEY")),
        use_semantic_scholar=bool(apis.get("use_semantic_scholar", True)),
        workers=int(res.get("workers", 4)),
        follow_redirects=bool(res.get("follow_redirects", True)),
        scrape_landing_pages=bool(res.get("scrape_landing_pages", False)),
        cache_days=int(res.get("cache_days", 30)),
        fulltext_dir=(base / data.get("fulltext_dir", "fulltext")).resolve(),
        outlook_csv=(base / data.get("outlook_csv", "paper_alerts.CSV")).resolve(),
        briefings_dir=(base / br.get("output_dir", "briefings")).resolve(),
        tts=str(br.get("tts", "kokoro")).lower(),
        voice=str(br.get("voice", "af_heart")),
        speed=float(br.get("speed", 1.0)),
        audio_format="m4b" if str(br.get("audio_format", "mp3")).lower() == "m4b" else "mp3",
        pronunciations=(base / br.get("pronunciations", "pronunciations.txt")).resolve(),
        examples_dir=(base / tri.get("examples_dir", "examples")).resolve(),
        profile=(base / tri.get("profile", "research_profile.md")).resolve(),
        embedding=str(tri.get("embedding", "specter2")),
        keep_recall=float(tri.get("keep_recall", 0.95)),
        skip_margin=float(tri.get("skip_margin", 0.02)),
        llm_provider=provider,
        llm_model=str(tri.get("llm_model") or DEFAULT_MODEL[provider]),
        llm_workers=int(tri.get("llm_workers") or DEFAULT_WORKERS[provider]),
        anthropic_api_key_env=str(tri.get("anthropic_api_key_env", "ANTHROPIC_API_KEY")),
        ollama_url=str(tri.get("ollama_url", "http://localhost:11434")),
        ollama_num_ctx=int(tri.get("ollama_num_ctx", 8192)),
    )
    if cfg.contact_email.endswith("example.org"):
        log.warning("Set contact_email in the config: OpenAlex and Crossref give better "
                    "service to requests that include a real contact address.")
    return cfg
