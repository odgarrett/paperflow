"""A small EPUB 3 writer (no dependencies): a front page, one chapter per section, and a
table of contents that lists every paper. Readable by KOReader and other e-readers."""
from __future__ import annotations

import html
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CSS = """
body { font-family: serif; line-height: 1.45; margin: 0 0.4em; }
h1 { font-size: 1.5em; margin: 0.6em 0 0.8em; }
h2 { font-size: 1.15em; margin: 1.6em 0 0.2em; }
p { margin: 0 0 0.7em; text-align: left; }
p.meta { font-size: 0.85em; color: #444; margin-bottom: 0.9em; }
h2.small { font-size: 1em; margin: 0.8em 0 0.1em; }
p.note { font-style: italic; }
"""


@dataclass
class Item:
    anchor: str
    heading: str
    meta: str
    paragraphs: list[str]


@dataclass
class Chapter:
    title: str
    intro: list[str] = field(default_factory=list)       # italic note under the heading
    paragraphs: list[str] = field(default_factory=list)  # body text before the items (e.g. an overview)
    items: list[Item] = field(default_factory=list)


def _inline(text: str) -> str:
    """Escape, then allow *italics* (species names, etc.)."""
    t = html.escape(text, quote=False)
    return re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<em>\1</em>", t)


def _page(title: str, body: str) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">\n'
            f'<head><meta charset="utf-8"/><title>{html.escape(title)}</title>'
            '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
            f'<body>\n{body}\n</body>\n</html>\n')


def write_epub(path: Path, title: str, front: list[str], chapters: list[Chapter], author: str = "paperflow") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    book_id = f"urn:uuid:{uuid.uuid4()}"
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {}

    files["front.xhtml"] = _page(title, f"<h1>{_inline(title)}</h1>\n" +
                                 "\n".join(f"<p>{_inline(p)}</p>" for p in front))
    for i, ch in enumerate(chapters, 1):
        parts = [f"<h1>{_inline(ch.title)}</h1>"]
        parts += [f'<p class="note">{_inline(p)}</p>' for p in ch.intro]
        parts += [f"<p>{_inline(p)}</p>" for p in ch.paragraphs]
        for it in ch.items:
            cls = ' class="small"' if not it.paragraphs else ""
            parts.append(f'<section id="{it.anchor}"><h2{cls}>{_inline(it.heading)}</h2>')
            if it.meta:
                parts.append(f'<p class="meta">{_inline(it.meta)}</p>')
            parts += [f"<p>{_inline(p)}</p>" for p in it.paragraphs]
            parts.append("</section>")
        files[f"ch{i:02d}.xhtml"] = _page(ch.title, "\n".join(parts))

    toc = ['<li><a href="front.xhtml">About this briefing</a></li>']
    for i, ch in enumerate(chapters, 1):
        sub = "".join(f'<li><a href="ch{i:02d}.xhtml#{it.anchor}">{_inline(it.heading)}</a></li>' for it in ch.items)
        toc.append(f'<li><a href="ch{i:02d}.xhtml">{_inline(ch.title)}</a>' + (f"<ol>{sub}</ol>" if sub else "") + "</li>")
    files["nav.xhtml"] = _page("Contents", '<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>'
                               + "".join(toc) + "</ol></nav>")
    files["style.css"] = CSS

    manifest = ['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
                '<item id="css" href="style.css" media-type="text/css"/>',
                '<item id="front" href="front.xhtml" media-type="application/xhtml+xml"/>']
    spine = ['<itemref idref="front"/>']
    for i in range(1, len(chapters) + 1):
        manifest.append(f'<item id="ch{i:02d}" href="ch{i:02d}.xhtml" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="ch{i:02d}"/>')
    opf = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">\n'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
           f'<dc:identifier id="bookid">{book_id}</dc:identifier>\n<dc:title>{html.escape(title)}</dc:title>\n'
           f'<dc:creator>{html.escape(author)}</dc:creator>\n<dc:language>en</dc:language>\n'
           f'<meta property="dcterms:modified">{modified}</meta>\n</metadata>\n'
           f'<manifest>\n{chr(10).join(manifest)}\n</manifest>\n<spine>\n{chr(10).join(spine)}\n</spine>\n</package>\n')
    container = ('<?xml version="1.0" encoding="utf-8"?>\n'
                 '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                 '</rootfiles></container>\n')

    tmp = path.with_suffix(".epub.tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/content.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        for name, content in files.items():
            z.writestr(f"OEBPS/{name}", content, compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(path)
