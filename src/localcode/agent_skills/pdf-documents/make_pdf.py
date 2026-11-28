#!/usr/bin/env python3
"""Write a PDF from a Markdown-style text file, with the Python standard library only.

    python3 make_pdf.py in.md out.pdf

Understands "# ", "## " and "### " headings, "- " or "* " bullets and blank-line
paragraphs; everything else is wrapped text. A4 pages, Helvetica, page breaks
as needed. No packages are installed or needed.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

W, H, MARGIN = 595, 842, 56  # A4 in points


def esc(s: str) -> str:
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def layout(text: str) -> list[list[tuple[str, int, float, str]]]:
    """Lines as (font, size, x, text), split into pages."""
    pages, page, y = [], [], H - MARGIN
    def put(font, size, x, s, gap):
        nonlocal page, y
        if y - size < MARGIN:
            pages.append(page); page, y = [], H - MARGIN
        y -= size + gap
        page.append((font, size, x, y, s))
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            y -= 6
            continue
        for mark, size in (("### ", 13), ("## ", 16), ("# ", 22)):
            if line.startswith(mark):
                for part in textwrap.wrap(line[len(mark):], int((W - 2 * MARGIN) / (size * 0.55))) or [""]:
                    put("F2", size, MARGIN, part, 8)
                break
        else:
            bullet = line.lstrip().startswith(("- ", "* "))
            body = line.lstrip()[2:] if bullet else line
            x = MARGIN + (14 if bullet else 0)
            for i, part in enumerate(textwrap.wrap(body, int((W - 2 * MARGIN - (x - MARGIN)) / 5.6)) or [""]):
                put("F1", 11, x, ("• " if bullet and i == 0 else ("  " if bullet else "")) + part, 4)
    pages.append(page)
    return pages


def main(argv: list[str]) -> int:
    if len(argv) != 3 or not argv[2].endswith(".pdf"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    pages = layout(Path(argv[1]).read_text(encoding="utf-8"))
    objs: list[bytes] = []
    def add(b: str | bytes) -> int:
        objs.append(b.encode("latin-1") if isinstance(b, str) else b)
        return len(objs)
    f1 = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    pages_id = len(objs) + 1 + 2 * len(pages)  # reserved after the page objects
    kids = []
    for page in pages:
        stream = "".join(f"BT /{font} {size} Tf {x:.1f} {y:.1f} Td ({esc(s)}) Tj ET\n" for font, size, x, y, s in page)
        data = stream.encode("latin-1", "replace")
        c = add(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
        kids.append(add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {W} {H}] "
                        f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> /Contents {c} 0 R >>"))
    assert add(f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>") == pages_id
    catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, catalog, xref)
    Path(argv[2]).write_bytes(bytes(out))
    print(f"wrote {argv[2]} ({len(kids)} page{'s' if len(kids) != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
