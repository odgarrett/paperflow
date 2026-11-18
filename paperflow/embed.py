"""Text embeddings for papers and examples.

Backends (config `triage.embedding`):
  specter2   allenai/specter2 with its "proximity" adapter: trained on scientific citation
             links, so papers that cite each other land close together. Needs
             `pip install -e ".[specter2]"` (PyTorch + transformers + adapters; ~1–2 GB)
             and downloads ~450 MB of model weights on first use.
  <name>     any sentence-transformers model, e.g. BAAI/bge-small-en-v1.5
             (`pip install -e ".[st]"`).
  tfidf      word statistics only; no download, no PyTorch. A weak but instant baseline,
             also used by the tests.

Vectors are L2-normalized and cached in the database per model, keyed by a hash of the
text, so a paper is only re-embedded when its text changes (e.g. an abstract arrives).
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

log = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (v / n).astype(np.float32)


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Specter2:
    name = "specter2"
    cacheable = True

    def __init__(self):
        try:
            import torch  # noqa: F401
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                f"SPECTER2 needs extra packages ({e.name} is missing). Install them with\n"
                '    pip install -e ".[specter2]"\n'
                "or set `embedding: tfidf` under `triage:` in config.yaml for a quick baseline.")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        self.device = _device()
        self.model.to(self.device).eval()
        self.sep = self.tokenizer.sep_token

    def format(self, title: str, abstract: str | None) -> str:
        return title + self.sep + (abstract or "")

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[i: i + batch_size], padding=True, truncation=True,
                                       return_tensors="pt", return_token_type_ids=False, max_length=512)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out.append(self.model(**batch).last_hidden_state[:, 0, :].float().cpu().numpy())
        return _normalize(np.vstack(out))


class SentenceTransformer:
    cacheable = True

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer as ST
        except ImportError:
            raise SystemExit('This embedding model needs sentence-transformers: pip install -e ".[st]"')
        self.name = model_name
        self.model = ST(model_name, device=_device())

    def format(self, title: str, abstract: str | None) -> str:
        return f"{title}. {abstract or ''}".strip()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return _normalize(np.asarray(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)))


class Tfidf:
    """Fitted on whatever corpus it's given, so vectors aren't cached between runs."""

    name = "tfidf"
    cacheable = False

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2),
                                          stop_words="english", max_features=200_000)

    def format(self, title: str, abstract: str | None) -> str:
        return f"{title}. {title}. {abstract or ''}"  # title counted twice: it's the densest signal

    def fit_encode(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        x = self.vectorizer.fit_transform(texts)
        k = min(256, x.shape[0] - 1, x.shape[1] - 1)
        if k >= 16:  # latent semantic analysis: lets related wording match
            x = TruncatedSVD(n_components=k, random_state=0).fit_transform(x)
        else:
            x = x.toarray()
        return _normalize(np.asarray(x))


def load_backend(name: str):
    if name == "specter2":
        return Specter2()
    if name == "tfidf":
        return Tfidf()
    return SentenceTransformer(name)


def embed_all(store, backend, items: list[tuple[str, str, str, str | None]], progress: bool = True) -> dict:
    """Embed (kind, ref, title, abstract) items, using and filling the cache.

    Returns {(kind, ref): vector}.
    """
    texts = {(k, r): backend.format(t, a) for k, r, t, a in items}
    if not backend.cacheable:
        keys = list(texts)
        vecs = backend.fit_encode([texts[k] for k in keys])
        return dict(zip(keys, vecs))

    db = store.db
    out, todo = {}, []
    for (kind, ref), text in texts.items():
        row = db.execute("SELECT text_hash, vec FROM embeddings WHERE kind=? AND ref=? AND model=?",
                         (kind, ref, backend.name)).fetchone()
        if row and row[0] == text_hash(text):
            out[(kind, ref)] = np.frombuffer(row[1], dtype=np.float32)
        else:
            todo.append((kind, ref))
    if todo:
        from tqdm import tqdm
        chunk = 256
        bar = tqdm(total=len(todo), unit="paper", desc="embedding", disable=not progress)
        for i in range(0, len(todo), chunk):
            keys = todo[i: i + chunk]
            vecs = backend.encode([texts[k] for k in keys])
            for (kind, ref), v in zip(keys, vecs):
                out[(kind, ref)] = v
                db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                           (kind, ref, backend.name, text_hash(texts[(kind, ref)]), v.tobytes()))
            db.commit()
            bar.update(len(keys))
        bar.close()
    return out
