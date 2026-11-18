"""Turn a raw email into a ParsedEmail with one Sighting per paper."""
from __future__ import annotations

import email
import email.policy
import email.utils
import hashlib
import logging
from datetime import timezone

from ..models import ParsedEmail
from ..text import html_to_linktext, squash
from . import biorxiv, generic, scholar, scholar_text, stork

log = logging.getLogger(__name__)


def _hdr(msg, name: str) -> str:
    try:
        return squash(str(msg.get(name, "") or ""))
    except Exception:  # malformed header
        return squash(str(msg.get_all(name, [""])[0]))


def message_id_of(msg) -> str:
    mid = _hdr(msg, "Message-ID")
    if mid:
        return mid
    basis = "|".join(_hdr(msg, h) for h in ("From", "Date", "Subject"))
    return "sha1:" + hashlib.sha1(basis.encode("utf-8", "replace")).hexdigest()


def classify_sender(sender: str, senders: dict) -> str | None:
    s = (sender or "").lower()
    for source, patterns in senders.items():
        if any(p.lower() in s for p in patterns):
            return source
    return None


def _bodies(msg) -> tuple[str | None, str | None]:
    html_parts, text_parts = [], []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/html", "text/plain"):
            continue
        try:
            content = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True) or b""
            content = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        (html_parts if ctype == "text/html" else text_parts).append(content)
    return ("\n".join(html_parts) or None), ("\n".join(text_parts) or None)


def _received_at(msg) -> str | None:
    raw = _hdr(msg, "Date")
    if not raw:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return None


def parse_email(raw: bytes, senders: dict, accept_all: bool = False) -> ParsedEmail | None:
    """Parse one raw RFC 822 message. Returns None if it isn't a recognized alert."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    sender, subject = _hdr(msg, "From"), _hdr(msg, "Subject")
    html, text = _bodies(msg)

    source = classify_sender(sender, senders)
    if source is None and html and "gse_alrt_title" in html:
        source = "scholar"
    if source is None:
        if not accept_all:
            return None
        source = "generic"

    alert_name, alert_kind, sightings, parser, n_reported = subject, None, None, "generic", None
    if source == "scholar":
        res = scholar.parse(html, subject) if html else None
        parser = "scholar"
        if res is None and text:
            res, parser = scholar_text.parse(text, subject), "scholar_text"
        if res is not None:
            alert_name, alert_kind, sightings = res
    elif source in ("biorxiv", "stork"):
        module = biorxiv if source == "biorxiv" else stork
        # Plain text (e.g. Outlook export) first, then the HTML rendered the same way.
        for body in (text, html_to_linktext(html) if html else None):
            res = module.parse(body, subject) if body else None
            if res is not None:
                alert_name, alert_kind, sightings, n_reported = res
                parser = source
                break
    if sightings is None:
        sightings = generic.parse(html, text, source)
        parser = "generic"

    received_at = _received_at(msg)
    if received_at is None:  # e.g. Outlook CSV exports carry no dates
        posted = [s.posted for s in sightings if s.posted]
        received_at = max(posted) if posted else None

    return ParsedEmail(
        message_id=message_id_of(msg),
        source=source,
        sender=sender,
        subject=subject,
        received_at=received_at,
        alert_name=alert_name,
        alert_kind=alert_kind,
        parser=parser,
        sightings=sightings,
        n_reported=n_reported,
    )
