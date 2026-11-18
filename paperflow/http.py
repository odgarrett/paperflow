"""HTTP access: on-disk cache, retries with backoff, and polite per-host rate limits."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import timedelta
from urllib.parse import urlparse

import requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# Minimum seconds between requests to each host (across all worker threads).
INTERVALS = {
    "api.openalex.org": 0.12,
    "api.crossref.org": 0.12,
    "www.ebi.ac.uk": 0.1,
    "api.biorxiv.org": 0.25,
    "api.semanticscholar.org": 1.1,
    "api.unpaywall.org": 0.1,
    # Preprint servers' websites (not their API) rate-limit by IP address, blocking browsers too.
    "www.biorxiv.org": 5.0, "biorxiv.org": 5.0, "connect.biorxiv.org": 5.0,
    "www.medrxiv.org": 5.0, "medrxiv.org": 5.0, "connect.medrxiv.org": 5.0,
}
DEFAULT_INTERVAL = 0.5


class HttpError(Exception):
    pass


MAX_RETRY_AFTER = 30      # seconds; longer rate-limit waits stop calls to that host for the run
MAX_HOST_FAILURES = 3     # consecutive failures before an API host is skipped for the run
MAX_SITE_FAILURES = 2     # same, for publisher websites visited to follow redirects

_RATE_LIMIT_HINT = (". It limits by IP address, so your browser may be blocked too for a while; "
                    "wait a few hours before running this again, or use another network")
_API_KEY_HINT = {
    "api.openalex.org": " (an OpenAlex API key raises the limit: set OPENALEX_API_KEY)",
    "api.semanticscholar.org": " (a Semantic Scholar API key raises the limit: set S2_API_KEY)",
    **{h: _RATE_LIMIT_HINT for h in ("www.biorxiv.org", "biorxiv.org", "connect.biorxiv.org",
                                     "www.medrxiv.org", "medrxiv.org", "connect.medrxiv.org")},
}


def _retry_after(r) -> float | None:
    value = r.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


class Http:
    def __init__(self, cache_path, contact_email: str, cache_days: int = 30):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests_cache.CachedSession(
            str(cache_path),
            backend="sqlite",
            expire_after=timedelta(days=cache_days),
            allowable_codes=(200, 404),
            stale_if_error=True,
        )
        # Transient server errors are retried here; 429 (rate limiting) is handled in get_json
        # so that very long Retry-After waits can be refused instead of silently sleeping.
        retry = Retry(
            total=3, connect=1, backoff_factor=1.5, status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"), respect_retry_after_header=False, raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
        ua = f"paperflow/0.1 (personal literature triage; mailto:{contact_email})"
        self.session.headers["User-Agent"] = ua
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Publisher websites: no retries, short timeouts.
        self.plain = requests.Session()
        self.plain.headers["User-Agent"] = ua
        self.plain.mount("https://", HTTPAdapter(max_retries=0))
        self.plain.mount("http://", HTTPAdapter(max_retries=0))

        self._host_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._next_ok: dict[str, float] = defaultdict(float)
        self._guard = threading.Lock()
        self._redirects: dict[str, str | None] = {}
        self._failed_urls: set[str] = set()       # downloads that failed this run: not retried
        self._failures: dict[str, int] = defaultdict(int)  # consecutive failures per host
        self.skipped_hosts: dict[str, str] = {}             # host -> reason, for the run summary

    # -- host health -----------------------------------------------------------------------

    def _skip_host(self, host: str, reason: str) -> None:
        if host not in self.skipped_hosts:
            self.skipped_hosts[host] = reason
            log.warning("Skipping %s for the rest of this run: %s%s", host, reason, _API_KEY_HINT.get(host, ""))

    def _failed(self, host: str, reason: str, limit: int = MAX_HOST_FAILURES) -> None:
        self._failures[host] += 1
        if self._failures[host] >= limit:
            self._skip_host(host, reason)

    def _check_host(self, host: str) -> None:
        if host in self.skipped_hosts:
            raise HttpError(f"{host}: skipped ({self.skipped_hosts[host]})")

    def _wait(self, host: str) -> None:
        interval = INTERVALS.get(host, DEFAULT_INTERVAL)
        with self._guard:
            lock = self._host_locks[host]
        with lock:
            now = time.monotonic()
            t = self._next_ok[host]
            if t > now:
                time.sleep(t - now)
                now = time.monotonic()
            self._next_ok[host] = now + interval

    def _is_cached(self, url: str, params) -> bool:
        try:
            req = requests.Request("GET", url, params=params).prepare()
            return self.session.cache.contains(request=req)
        except Exception:
            return False

    # -- requests --------------------------------------------------------------------------

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None,
                 timeout: tuple = (10, 30)):
        """GET and decode JSON. Returns None on 404 or non-JSON; raises HttpError otherwise."""
        host = urlparse(url).netloc
        cached = self._is_cached(url, params)
        for attempt in range(4):
            if not cached:
                self._check_host(host)
                self._wait(host)
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                self._failed(host, f"connection problems ({e.__class__.__name__})")
                raise HttpError(f"{host}: {e.__class__.__name__}") from e
            if r.status_code != 429:
                break
            wait = _retry_after(r)
            if wait is None:
                wait = 2.0 * 2 ** attempt
            if wait > MAX_RETRY_AFTER:
                self._skip_host(host, f"rate limited (asked to wait {wait / 60:.0f} min)")
                raise HttpError(f"{host}: HTTP 429 (rate limited)")
            log.debug("%s rate limited; waiting %.0fs", host, wait)
            time.sleep(wait)
        else:
            self._failed(host, "repeatedly rate limited")
            raise HttpError(f"{host}: HTTP 429 (rate limited)")

        if r.status_code == 404:
            self._failures[host] = 0
            return None
        if r.status_code >= 400:
            self._failed(host, f"HTTP {r.status_code} errors")
            raise HttpError(f"{host}: HTTP {r.status_code}")
        self._failures[host] = 0
        try:
            return r.json()
        except ValueError:
            return None

    def get_html(self, url: str, timeout: tuple = (5, 15)) -> str | None:
        """Fetch a landing page (cached). Many publishers block scripts; failures return None."""
        host = urlparse(url).netloc
        if host in self.skipped_hosts:
            return None
        self._wait(host)
        try:
            r = self.session.get(url, timeout=timeout)
        except requests.RequestException:
            self._failed(host, "not responding", MAX_SITE_FAILURES)
            return None
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text

    def get_bytes(self, url: str, timeout: tuple = (10, 60), max_bytes: int = 60_000_000):
        """Download a file (PDF, XML). Returns (data, content type, final url) or None.

        Uncached and polite: a site that answers "too many requests" (429) is left alone for the
        rest of the run (after one short wait, if it asks for one), and a site that keeps failing
        is skipped too. A URL that failed isn't tried again in the same run."""
        if url in self._failed_urls:
            return None
        result = None
        for attempt in range(2):
            host = urlparse(url).netloc
            if host in self.skipped_hosts:
                break
            self._wait(host)
            try:
                with self.plain.get(url, timeout=timeout, stream=True, allow_redirects=True,
                                    headers={"Accept": "application/pdf, application/xml, text/xml, */*"}) as r:
                    final_host = urlparse(r.url).netloc
                    if r.status_code == 429:
                        wait = _retry_after(r)
                        if wait is not None and wait <= MAX_RETRY_AFTER and attempt == 0:
                            time.sleep(wait)
                            continue
                        self._skip_host(final_host, "too many requests (rate limited)")
                        self._skip_host(host, "too many requests (rate limited)")
                        break
                    if r.status_code != 200:
                        if r.status_code >= 500 or r.status_code == 403:
                            self._failed(final_host, f"HTTP {r.status_code}", MAX_SITE_FAILURES)
                        break
                    chunks, size = [], 0
                    for chunk in r.iter_content(65536):
                        size += len(chunk)
                        if size > max_bytes:
                            chunks = None
                            break
                        chunks.append(chunk)
                    self._failures[host] = 0
                    if chunks is not None:
                        result = (b"".join(chunks), r.headers.get("Content-Type", ""), r.url)
                    break
            except requests.RequestException:
                self._failed(host, "not responding", MAX_SITE_FAILURES)
                break
        if result is None:
            self._failed_urls.add(url)
        return result

    def final_url(self, url: str, timeout: tuple = (5, 10)) -> str | None:
        """Follow redirects (tracking links etc.) and return the destination URL.

        Publisher websites are often slow or block scripts, so: short timeouts, no retries,
        and a site that fails twice is not contacted again during this run.
        """
        if url in self._redirects:
            return self._redirects[url]
        host = urlparse(url).netloc
        if host in self.skipped_hosts:
            return None
        self._wait(host)
        final = None
        try:
            r = self.plain.head(url, allow_redirects=True, timeout=timeout)
            if r.status_code == 405:  # HEAD not allowed
                r = self.plain.get(url, allow_redirects=True, timeout=timeout, stream=True)
                r.close()
            final = r.url
            self._failures[host] = 0
        except requests.RequestException:
            self._failed(host, "not responding", MAX_SITE_FAILURES)
        self._redirects[url] = final
        return final
