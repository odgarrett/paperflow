"""Where raw alert emails come from: IMAP, an mbox file (e.g. Gmail Takeout), or .eml files."""
from __future__ import annotations

import csv
import email
import email.policy
import email.utils
import hashlib
import sys
import imaplib
import logging
import mailbox
import os
import re
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterator

from .parsers import _hdr, message_id_of

log = logging.getLogger(__name__)


def iter_mbox(path: Path) -> Iterator[bytes]:
    for msg in mailbox.mbox(str(path)):
        yield msg.as_bytes()


def iter_eml_dir(path: Path) -> Iterator[bytes]:
    for f in sorted(Path(path).rglob("*.eml")):
        yield f.read_bytes()


def iter_outlook_csv(path: Path) -> Iterator[bytes]:
    """Outlook's "Export to CSV" (Subject, Body, From: (Name), From: (Address), …).

    The export holds only the plain-text body, with links rendered as "text <url>", and has
    no dates or Message-IDs. Each row is rebuilt as a minimal email so the rest of the
    pipeline treats it like any other message; the Message-ID is a hash of sender, subject
    and body, so re-importing the same export is harmless.
    """
    csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows on Windows
    required = {"Subject", "Body", "From: (Name)", "From: (Address)"}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path} doesn't look like an Outlook CSV export: missing column(s) "
                f"{', '.join(sorted(missing))}. Found: {', '.join(reader.fieldnames or [])}. "
                "(Outlook in another language names these columns differently; rename them in the "
                "file's first line to Subject, Body, From: (Name), From: (Address).)")
        for row in reader:
            body = row.get("Body") or ""
            addr = row.get("From: (Address)") or ""
            subject = " ".join((row.get("Subject") or "").split())
            digest = hashlib.sha1("\x1f".join((addr, subject, body)).encode("utf-8")).hexdigest()
            msg = EmailMessage()
            msg["From"] = email.utils.formataddr((row.get("From: (Name)") or "", addr))
            msg["Subject"] = subject
            msg["Message-ID"] = f"<{digest}@outlook-csv.paperflow>"
            msg["X-Paperflow-Origin"] = "outlook-csv"
            msg.set_content(body)
            yield msg.as_bytes()


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i: i + n]


def _imap_quote(folder: str) -> str:
    return '"' + folder.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _since_str(since) -> str | None:
    if not since:
        return None
    if isinstance(since, str):
        since = datetime.strptime(since, "%Y-%m-%d").date()
    if isinstance(since, datetime):
        since = since.date()
    if isinstance(since, date):
        return since.strftime("%d-%b-%Y")
    return None


def iter_imap(
    imap_cfg: dict,
    accept: Callable[[str, str], bool],
    skip_message_ids: set[str],
) -> Iterator[bytes]:
    """Yield raw messages from the configured folders.

    Headers are fetched first so that already-ingested and non-alert messages are skipped
    without downloading their bodies. Messages are opened read-only (BODY.PEEK), so nothing
    is marked as read.
    """
    host = imap_cfg.get("host")
    user = imap_cfg.get("username")
    pw_env = imap_cfg.get("password_env", "PAPERFLOW_IMAP_PASSWORD")
    password = os.environ.get(pw_env)
    if not (host and user):
        raise SystemExit("imap.host and imap.username must be set in the config file")
    if not password:
        raise SystemExit(f"Set the IMAP password in the environment variable {pw_env}")

    conn = imaplib.IMAP4_SSL(host, int(imap_cfg.get("port", 993)))
    conn.login(user, password)
    try:
        for folder in imap_cfg.get("folders") or ["INBOX"]:
            typ, _ = conn.select(_imap_quote(folder), readonly=True)
            if typ != "OK":
                log.warning("Could not open IMAP folder %r", folder)
                continue
            since = _since_str(imap_cfg.get("since"))
            criteria = ("SINCE", since) if since else ("ALL",)
            typ, data = conn.uid("SEARCH", None, *criteria)
            uids = data[0].split() if typ == "OK" and data and data[0] else []
            log.info("Folder %r: %d messages", folder, len(uids))

            wanted = []
            for chunk in _chunks(uids, 200):
                typ, resp = conn.uid(
                    "FETCH", b",".join(chunk),
                    "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID FROM SUBJECT DATE)])",
                )
                for part in resp or []:
                    if not isinstance(part, tuple):
                        continue
                    m = re.search(rb"UID (\d+)", part[0])
                    if not m:
                        continue
                    hdr = email.message_from_bytes(part[1], policy=email.policy.default)
                    if message_id_of(hdr) in skip_message_ids:
                        continue
                    if not accept(_hdr(hdr, "From"), _hdr(hdr, "Subject")):
                        continue
                    wanted.append(m.group(1))
            log.info("Folder %r: %d new alert messages to download", folder, len(wanted))

            for chunk in _chunks(wanted, 25):
                typ, resp = conn.uid("FETCH", b",".join(chunk), "(BODY.PEEK[])")
                for part in resp or []:
                    if isinstance(part, tuple):
                        yield part[1]
    finally:
        try:
            conn.logout()
        except Exception:
            pass
