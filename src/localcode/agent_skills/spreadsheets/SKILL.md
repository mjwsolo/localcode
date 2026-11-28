---
name: spreadsheets
description: Create Excel spreadsheets (.xlsx) and CSV files from data or tables. Use when the user asks for a spreadsheet, an Excel or Numbers file, a .xlsx, or a table to open in Excel.
---

No packages needed: `make_xlsx.py` in this skill's folder uses only Python's standard library.

1. Write each sheet as a CSV file (first row = headers). Use Python's `csv` module so commas and quotes are escaped.
2. Run: `python3 <this skill's folder>/make_xlsx.py "out.xlsx" sheet1.csv [sheet2.csv ...]`. Each CSV becomes a sheet named after the file; the header row is bold; numbers stay numbers.
3. Verify: `file "out.xlsx"` says "Microsoft Excel 2007+".
4. Delete temporary CSVs unless the user wanted them, and tell the user the path.

The user asked only for CSV: write it with the `csv` module, no conversion.
Formulas are not supported by the script; compute values in Python first.

Never save a different format than the one asked for.
