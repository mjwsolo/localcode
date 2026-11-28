---
name: word-documents
description: Create or convert Word and text documents (.docx, .doc, .rtf, .odt). Use when the user asks for a Word file, a .docx, or a document, letter, report or memo to open in Word or Pages.
---

Built into macOS, nothing to install.

1. Write the content as HTML (headings `<h1>`, lists `<ul><li>`, tables `<table border=1>`, `<b>`, `<i>`) to a temporary `.html` file.
2. Convert: `textutil -convert docx in.html -output "out.docx"` (or `doc`, `rtf`, `odt`).
3. Verify: `textutil -convert txt -stdout "out.docx" | head` shows the text.
4. Delete the temporary HTML and tell the user the path of the file.

Convert an existing file: `textutil -convert docx file.rtf` (inputs: txt, html, rtf, doc, docx, odt, webarchive).
Read a .docx: `textutil -convert txt -stdout file.docx`.

Never save a different format than the one asked for.
