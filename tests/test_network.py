"""Stall scenarios against local servers (no internet needed): a publisher site that never
answers, an API demanding a very long rate-limit wait, and one asking for a short wait."""
import json, socket, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow import apis, http as phttp
from paperflow.config import Config
from paperflow.http import Http, HttpError
from paperflow.resolver import Resolver


def hanging_site():
    """Accepts connections, never responds (like a slow or bot-blocking journal site)."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(50)
    conns = []
    threading.Thread(target=lambda: [conns.append(srv.accept()) for _ in iter(int, 1)], daemon=True).start()
    return srv.getsockname()[1]


calls = {"slow429": 0, "brief429": 0}

class Api(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send_json(self, obj, code=200, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (headers or {}).items(): self.send_header(k, v)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path.startswith("/limited"):
            calls["slow429"] += 1
            return self.send_json({}, 429, {"Retry-After": "3600"})
        if u.path.startswith("/brief"):
            calls["brief429"] += 1
            if calls["brief429"] == 1:
                return self.send_json({}, 429, {"Retry-After": "1"})
            return self.send_json({"ok": True})
        if u.path == "/oa/works":  # OpenAlex title search: matches titles starting with "Known"
            f = q.get("filter", [""])[0]
            if "known paper" in f:
                n = f.split()[-1]
                return self.send_json({"results": [{"display_name": f"Known paper {n}", "doi": f"https://doi.org/10.9999/known.{n}",
                                       "publication_year": 2026, "authorships": [], "type": "article"}]})
            return self.send_json({"results": []})
        if u.path.startswith("/oa/works/doi:"):
            n = u.path.rsplit(".", 1)[-1]
            return self.send_json({"display_name": f"Known paper {n}", "doi": f"https://doi.org/10.9999/known.{n}",
                                   "publication_year": 2026, "type": "article",
                                   "abstract_inverted_index": {"An": [0], "abstract.": [1]}})
        if u.path.startswith("/cr/"):
            return self.send_json({"message": {"items": []}}) if u.path == "/cr/works" else self.send_json({}, 404)
        if u.path.startswith("/epmc"):
            return self.send_json({"resultList": {"result": []}})
        self.send_json({}, 404)


def main():
    api = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{api.server_address[1]}"
    site = f"http://127.0.0.1:{hanging_site()}"
    api_host = urlparse(base).netloc
    phttp.INTERVALS[api_host] = 0.0
    tmp = Path(tempfile.mkdtemp())
    h = Http(tmp / "cache.sqlite", "me@example.org")

    # 1. very long Retry-After: refuse to wait, skip the host
    t = time.monotonic()
    try:
        h.get_json(base + "/limited/x"); raise SystemExit("expected HttpError")
    except HttpError as e:
        print("long rate limit ->", e)
    assert time.monotonic() - t < 2 and calls["slow429"] == 1
    # (host is now skipped; reset for the remaining checks)
    h.skipped_hosts.clear()

    # 2. short Retry-After: wait and succeed
    t = time.monotonic()
    assert h.get_json(base + "/brief/x") == {"ok": True}
    print(f"brief rate limit -> waited {time.monotonic() - t:.1f}s then succeeded")

    # 3. resolver: 20 papers without DOIs, all linking to the hanging site.
    #    10 are findable by title; 10 aren't, so they fall back to visiting the site.
    apis.OpenAlex.BASE = base + "/oa"; apis.Crossref.BASE = base + "/cr"; apis.EuropePMC.URL = base + "/epmc"
    cfg = Config(follow_redirects=True, use_semantic_scholar=False)
    r = Resolver(h, cfg)
    items = [{"id": i, "title": f"{'Known' if i % 2 else 'Unfindable'} paper {i} about rice blast effectors",
              "title_truncated": False, "year": 2026, "ids": {}, "surnames": [],
              "urls": [f"{site}/jour/article/view/{i}", f"{site}/jour/article/download/{i}"]} for i in range(20)]
    for it in items:  # make findable titles match the mock's "Known paper N" exactly
        if it["id"] % 2: it["title"] = f"Known paper {it['id']}"
    from concurrent.futures import ThreadPoolExecutor
    t = time.monotonic()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(r.resolve, items))
    took = time.monotonic() - t
    status = {}
    for res in results: status[res.status] = status.get(res.status, 0) + 1
    print(f"20 papers in {took:.1f}s: {status}; skipped: {h.skipped_hosts}")
    assert status.get("resolved") == 10, status
    assert took < 40, took   # before the fix: minutes per paper
    print("NETWORK STALL CHECKS PASSED")


if __name__ == "__main__":
    main()
