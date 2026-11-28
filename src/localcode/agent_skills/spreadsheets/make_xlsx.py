#!/usr/bin/env python3
"""Write an Excel .xlsx from CSV files, with the Python standard library only.

    python3 make_xlsx.py out.xlsx data.csv [more.csv ...]

Each CSV becomes one sheet named after the file. Numbers stay numbers; the
first row is bold. No packages are installed or needed.
"""
from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def sheet_xml(rows: list[list[str]]) -> str:
    out = []
    for r, row in enumerate(rows, 1):
        cells = []
        for c, v in enumerate(row, 1):
            ref = f"{col(c)}{r}"
            style = ' s="1"' if r == 1 else ""
            if NUM.match(v.strip()):
                cells.append(f'<c r="{ref}"{style}><v>{v.strip()}</v></c>')
            elif v != "":
                cells.append(f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{escape(v)}</t></is></c>')
        out.append(f'<row r="{r}">{"".join(cells)}</row>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(out)}</sheetData></worksheet>')


def main(argv: list[str]) -> int:
    if len(argv) < 3 or not argv[1].endswith(".xlsx"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    out, sources = Path(argv[1]), [Path(p) for p in argv[2:]]
    names, sheets = [], []
    for i, src in enumerate(sources, 1):
        with src.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        name = re.sub(r"[\[\]:*?/\\]", "_", src.stem)[:31] or f"Sheet{i}"
        names.append(name)
        sheets.append(sheet_xml(rows))
    ct = "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                 for i in range(1, len(sheets) + 1))
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{ct}</Types>",
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            + "".join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, 1))
            + "</sheets></workbook>",
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                      for i in range(1, len(sheets) + 1))
            + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>",
        "xl/styles.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2"><font/><font><b/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="2"><xf/><xf fontId="1" applyFont="1"/></cellXfs>'
            "</styleSheet>",
    }
    for i, xml in enumerate(sheets, 1):
        files[f"xl/worksheets/sheet{i}.xml"] = xml
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    print(f"wrote {out} ({len(sheets)} sheet{'s' if len(sheets) != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
