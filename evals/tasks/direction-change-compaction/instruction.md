## Turn 1

Read every file under docs/ and summarize each one in one line. Do not write any code yet.

## Turn 2

Plan for today: add a function to_csv(rows) to app/report.py that returns the rows (a list of dicts) as CSV text with a header line. Before coding, read docs/fields.md again and list the field names in order. Do not write code yet.

## Turn 3

Change of direction: do not add CSV output at all. Instead add to_markdown(rows) to app/report.py. It returns a Markdown table as a string: a header row with the keys of the first dict in order, then a separator row with a --- cell per column, then one row per dict. Every row starts with '| ', ends with ' |' and separates cells with ' | '; rows are joined with newlines. Do not write code yet: first read docs/history.md again and tell me in one line how many weekly entries it has.

## Turn 4

Now implement it in app/report.py, add a test for it to tests/test_report.py, and run the tests.
