# LocalCode Feature Sequences — Complete Blueprint

**Principle:** Model is a content generator inside a deterministic harness.
Harness owns sequencing, validation, retries, and file safety. Model fills in the blanks.

## Architecture

```
USER INPUT → INTENT CLASSIFIER (regex) → STATE MACHINE (per feature) → TOOL EXECUTOR
```

Intent classification: rule-based keyword matching, not LLM.
State machine: GATHER → LLM_CALL → APPLY → VERIFY → FIX_LOOP → DONE.
Tools: read, write, edit, glob, grep, bash. All local. Parallel where independent.

---

## Feature 1: FILE CREATION

User: "create a flask app" / "make pong.py"

```
1. glob(pattern="*.py")                          # check what exists
2. glob(pattern="**/requirements*.txt")           # check deps
3. bash(command="python3 --version")              # confirm runtime
   → Steps 1-3 PARALLEL

4. llm_call(prompt="Write complete {desc}. Output ONLY code.", max_tokens=4096)
   → HARNESS: strip markdown fences, validate syntax

5. write(path="{target}", content="{llm_output}")

6. bash(command="python3 -c 'import ast; ast.parse(open(\"{target}\").read())'")
   → Syntax check

   IF step 6 fails (max 2 retries):
7.   llm_call(prompt="Fix this error: {error}\n{code}\nOutput ONLY corrected file.")
8.   write(path="{target}", content="{fixed}")
9.   bash(verify again)

10. bash(command="pip3 install {detected_imports}")
    → ONLY if non-stdlib imports detected
```

**Total: 6 clean, 10 max.**

---

## Feature 2: FILE EDITING

User: "add error handling to /login" / "rename Player to Paddle"

```
1. glob(pattern="**/*.py")
   → Find candidates if file not specified

2. llm_call(prompt="Which file contains {desc}? Choose from: {files}. Just the filename.", max_tokens=50)
   → SKIP steps 1-2 if user specified file

3. read(path="{target_file}")

4. llm_call(prompt="Here is {file}:\n```\n{content}\n```\nTask: {request}\n\nOutput SEARCH/REPLACE blocks:\n<<<SEARCH\n{exact lines}\n===\n{replacement}\nSEARCH>>>", max_tokens=2048)
   → HARNESS: parse blocks, validate SEARCH exists in file

5. edit(path="{target}", old_string="{search}", new_string="{replace}")
   → One edit per block. If old_string not found: re-prompt with more context.

6. bash(command="python3 -c 'import ast; ast.parse(...)'")

   IF fails:
7.   read(path="{target}")
8.   llm_call(prompt="Fix errors after editing: {error}\n{file}")
9.   edit(...)
10.  bash(verify)
```

**Total: 6 clean, 10 max.**

**HARNESS validation before applying edits:**
```python
count = file_content.count(search)
if count == 0: return ERROR_NOT_FOUND    # re-prompt model
if count > 1:  return ERROR_AMBIGUOUS    # ask for more context
return file_content.replace(search, replace, 1)
```

---

## Feature 3: CODE REVIEW

User: "review this file" / "what's wrong with auth.py"

```
1. read(path="{target}")

2. llm_call(prompt="Review for bugs, security, improvements:\n```\n{content}\n```\nFormat:\n- LINE: {n}\n- SEVERITY: high|medium|low\n- ISSUE: one line\n- FIX: one line", max_tokens=2048)
```

**Total: 2. Read-only, no further calls.**

For diff review:
```
1. bash(command="git diff --staged")
2. llm_call(prompt="Review this diff. Flag: bugs, security, logic errors.\nFormat: FILE:LINE — ISSUE")
```

---

## Feature 4: CODE EXPLANATION

User: "explain this function" / "what does this file do"

```
1. read(path="{target}", offset={start}, limit={end})
2. llm_call(prompt="Explain concisely. What it does, why, non-obvious parts:\n```\n{code}\n```", max_tokens=1024)
```

**Total: 2.**

---

## Feature 5: BUG FIXING

User: "crashes with KeyError on line 45" / "fix the login bug"

```
1. read(path="{target}")
2. bash(command="python3 {target} 2>&1 | head -50")
   → Steps 1-2 PARALLEL

3. llm_call(prompt="Bug in code.\nFile:\n```\n{content}\n```\nError: {error}\nUser: {desc}\n\nOutput SEARCH/REPLACE to fix.", max_tokens=2048)

4. edit(path="{target}", old_string="{search}", new_string="{replace}")

5. bash(command="python3 {target} 2>&1 | head -50")
   → Verify fix

   IF still failing (max 3 iterations):
6.   llm_call(prompt="Still broken: {new_error}\nPrevious fix: {diff}\nTry again.")
7.   edit(...)
8.   bash(verify)
```

**Total: 5 clean, 11 max.**

---

## Feature 6: REFACTORING

User: "extract into separate module" / "refactor to use classes"

```
1. read(path="{target}")
2. glob(pattern="**/*.py")
   → Steps 1-2 PARALLEL

3. llm_call(prompt="Refactor: {instruction}\n```\n{content}\n```\nOutput COMPLETE new file(s):\n---FILE: path---\n{content}\n---END---", max_tokens=4096)

4. write(path="{file1}", content="{content1}")
5. write(path="{file2}", content="{content2}")

6. bash(command="python3 -c 'import ast; ast.parse(open(\"{file1}\").read())'")
7. bash(command="python3 -c 'import ast; ast.parse(open(\"{file2}\").read())'")
   → Steps 6-7 PARALLEL

8. bash(command="python3 -c 'import {module}'")
```

**Total: 8.**

---

## Feature 7: TEST GENERATION

User: "write tests for auth.py"

```
1. read(path="{source}")
2. glob(pattern="**/test_*")
   → Steps 1-2 PARALLEL

3. llm_call(prompt="Write pytest tests for:\n```\n{content}\n```\nTest every public function. Output ONLY test code.", max_tokens=4096)

4. write(path="test_{source}.py", content="{llm_output}")

5. bash(command="python3 -m pytest test_{source}.py -v 2>&1 | tail -30")

   IF tests fail (bad test code):
6.   llm_call(prompt="Fix test errors: {output}")
7.   write(...)
8.   bash(pytest again)
```

**Total: 5 clean, 8 max.**

---

## Feature 8: TEST RUNNING

User: "run the tests"

```
1. glob(pattern="**/test_*.py")
2. bash(command="python3 -m pytest {files} -v --tb=short 2>&1 | tail -50")

   IF failures:
3.   llm_call(prompt="Tests failed:\n{output}\nSummarize what's broken in 2-3 lines.", max_tokens=512)
```

**Total: 2-3.**

---

## Feature 9: PROJECT SCAFFOLDING

User: "start a new FastAPI project"

```
1. bash(command="ls -la")

2. llm_call(prompt="Generate file tree + contents for {framework} project.\nFormat:\n---FILE: path---\n{content}\n---END---", max_tokens=4096)

3. bash(command="mkdir -p {dirs}")
4. write(path="{file1}", content="{c1}")
5. write(path="{file2}", content="{c2}")
6. write(path="{file3}", content="{c3}")
   → Steps 4-6 PARALLEL

7. bash(command="pip3 install -r requirements.txt")
8. bash(command="python3 -c 'import {main_module}'")
```

**Total: 8-12.**

---

## Feature 10: GIT OPERATIONS

User: "commit this" / "what changed"

```
1. bash(command="git status --short")
2. bash(command="git diff --staged")
3. bash(command="git log --oneline -5")
   → Steps 1-3 PARALLEL

4. llm_call(prompt="Commit message for this diff:\n{diff}\nRecent:\n{log}\nOutput ONLY the message.", max_tokens=200)

5. bash(command="git add {files}")
6. bash(command="git commit -m '{msg}'")
```

**Total: 6.**

---

## Feature 11: SEARCH / FIND

User: "where is the auth logic" / "find all API routes"

```
1. grep(pattern="{term}", path=".", output_mode="content")

   IF too many results:
2.   llm_call(prompt="User looking for: {query}\nMatches:\n{results}\nTop 3 most relevant?", max_tokens=512)
```

**Total: 1-2.**

---

## Feature 12: DEPENDENCY MANAGEMENT

User: "add requests library"

```
1. glob(pattern="**/requirements*.txt")
2. glob(pattern="**/pyproject.toml")
   → Steps 1-2 PARALLEL

3. read(path="{dep_file}")
4. bash(command="pip3 install {package}")
5. edit(path="{dep_file}", add package to list)
```

**Total: 5.**

---

## Feature 13: MULTI-FILE CONTEXT

User: "understand how payment flow works"

```
1. grep(pattern="payment|checkout|charge", output_mode="files_with_matches")

2. read(path="{file1}")
3. read(path="{file2}")
4. read(path="{file3}")
   → Steps 2-4 PARALLEL. Cap at 3-4 files.

5. llm_call(prompt="Explain payment flow across files:\n{summaries}\nTrace from user action to completion.", max_tokens=2048)
```

**Total: 5.**
