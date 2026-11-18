"""bioRxiv-style rate limiting: stop at the first 429, retry papers next run, never hammer."""
import sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow import http as phttp
from paperflow.fulltext import Fetcher
from test_library import JATS, make_pdf

hits = []
class Site(BaseHTTPRequestHandler):
    brief_calls = 0
    def log_message(self, *a): pass
    def do_GET(self):
        hits.append(self.path)
        if self.path.startswith("/limited"):
            self.send_response(429); self.end_headers(); return            # no Retry-After: back off entirely
        if self.path.startswith("/brief"):
            Site.brief_calls += 1
            if Site.brief_calls == 1:
                self.send_response(429); self.send_header("Retry-After", "1"); self.end_headers(); return
            body = JATS
        elif self.path.startswith("/ok.pdf"):
            body = make_pdf(Path(tempfile.mkdtemp()) / "x.pdf", "A rice blast effector")
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

class HybridHttp(phttp.Http):
    """Real downloads against the local site; API answers faked."""
    def __init__(self, base, *a, **k):
        super().__init__(*a, **k); self.base = base; self.api = []
    def get_json(self, url, params=None, headers=None):
        self.api.append(url)
        q = (params or {}).get("query", "")
        if url.endswith("/search"):
            if "10.1111/published.1" in q:
                return {"resultList": {"result": [{"pmcid": "PMC777", "source": "MED", "id": "7", "inEPMC": "Y"}]}}
            return {"resultList": {"result": []}}
        if "api.biorxiv.org" in url:
            bid = url.rsplit("/", 1)[1]
            return {"collection": [{"version": "1", "jatsxml": f"{self.base}/limited/{bid}.source.xml"}]}
        if "unpaywall" in url:
            doi = url.split("/v2/")[1]
            if doi.endswith("697761"):
                return {"oa_locations": [{"url_for_pdf": f"{self.base}/ok.pdf?x=1"}]}
            return {"oa_locations": [{"url_for_pdf": f"{self.base}/limited/{doi.rsplit('.', 1)[1]}.full.pdf"}]}
        return None
    def get_bytes(self, url, **kw):
        if url.endswith("PMC777/fullTextXML"):
            return JATS, "application/xml", url
        return super().get_bytes(url, **kw)

def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    host = base.split("/")[2]
    phttp.INTERVALS[host] = 0.0
    tmp = Path(tempfile.mkdtemp())

    h = HybridHttp(base, tmp / "c.sqlite", "me@x.org")
    f = Fetcher(h, "me@x.org")
    r1 = f.fetch({"doi": "10.64898/2026.01.05.697760", "title": "Pik"})
    assert r1.status == "error" and "rate limiting" in r1.detail, r1
    assert host in h.skipped_hosts and len(hits) == 1, (h.skipped_hosts, hits)   # stopped at the first 429
    r2 = f.fetch({"doi": "10.64898/2026.01.05.697762", "title": "Another"})
    assert r2.status == "error" and len(hits) == 1, hits                         # never contacted again
    print("ok: first 429 stops all requests to the site; papers queued for retry, not written off")

    # Published version comes from Europe PMC: no request to the preprint server at all.
    h = HybridHttp(base, tmp / "c2.sqlite", "me@x.org"); f = Fetcher(h, "me@x.org"); n = len(hits)
    r3 = f.fetch({"doi": "10.64898/2026.01.05.697760", "published_doi": "10.1111/published.1", "title": "Pik"})
    assert r3.status == "ok" and r3.source == "europepmc (published version)" and len(hits) == n, (r3, hits)
    print("ok: published version used first, no bioRxiv request")

    # A short Retry-After is honoured once, then the download succeeds.
    h = HybridHttp(base, tmp / "c3.sqlite", "me@x.org")
    got = h.get_bytes(f"{base}/brief/x.xml")
    assert got and got[0] == JATS and host not in h.skipped_hosts and Site.brief_calls == 2
    print("ok: short Retry-After waited out once")

    # Same PDF offered twice (Unpaywall + OpenAlex, different query strings): requested once.
    h = HybridHttp(base, tmp / "c4.sqlite", "me@x.org"); f = Fetcher(h, "me@x.org"); n = len(hits)
    r4 = f.fetch({"doi": "10.5555/697761", "title": "A rice blast effector", "oa_pdf_url": f"{base}/ok.pdf?x=2"})
    assert r4.status == "ok" and sum(p.startswith("/ok.pdf") for p in hits[n:]) == 1, hits[n:]
    print("ok: duplicate links fetched once")

    assert phttp.INTERVALS["www.biorxiv.org"] >= 5
    srv.shutdown()
    print("RATE LIMIT CHECKS PASSED")

if __name__ == "__main__":
    main()
