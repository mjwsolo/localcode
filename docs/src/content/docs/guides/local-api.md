---
title: Local API
description: Use a running LocalCode model from another application.
---

While LocalCode is open, its model server offers an OpenAI compatible API on `127.0.0.1`. It requires a different random API key for each run. Get the current address and model with:

```bash
localcode api
```

For an application that needs the key, request machine readable details explicitly:

```bash
localcode api --json --show-key
```

For example, a Python application using the OpenAI SDK can use the same model:

```python
import json
import subprocess
from openai import OpenAI

info = json.loads(subprocess.check_output(
    ["localcode", "api", "--json", "--show-key"], text=True
))
client = OpenAI(base_url=info["base_url"], api_key=info["api_key"])
answer = client.chat.completions.create(
    model=info["model"],
    messages=[{"role": "user", "content": "Summarize this change."}],
)
print(answer.choices[0].message.content)
```

Start LocalCode and load a model before using the API. The endpoint ends when the LocalCode session closes, and its port and key can change on the next run. Keep the key private. The server listens only on your Mac, so another machine cannot connect to it.

This is the model inference API. LocalCode currently consumes MCP servers as a client; it does not expose its coding actions as an MCP server. System One decision models need a separate runtime and response schema and are not part of this endpoint yet.
