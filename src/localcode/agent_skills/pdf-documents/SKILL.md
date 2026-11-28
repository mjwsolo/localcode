---
name: pdf-documents
description: Create PDF files from text, notes or Markdown, and read text out of PDFs. Use when the user asks for a PDF or to export or save something as PDF.
---

No packages needed: `make_pdf.py` in this skill's folder uses only Python's standard library.

1. Write the content as Markdown-style text to a temporary `.md` file: `# Title`, `## Section`, `- bullet`, blank lines between paragraphs.
2. Run: `python3 <this skill's folder>/make_pdf.py in.md "out.pdf"`
3. Verify: `file "out.pdf"` says "PDF document" with a page count.
4. Delete the temporary file and tell the user the path of the PDF.

Plain text only, no layout needed: `cupsfilter in.txt > out.pdf` also works.
Read the text of a PDF: the read tool opens PDFs; `mdls file.pdf` shows page count and title.

Never save a different format than the one asked for.
