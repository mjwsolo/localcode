---
title: Local API
description: Use a running LocalCode model from another application.
---

While LocalCode is open, its model server offers an OpenAI compatible API on `127.0.0.1`. It requires a different random API key for each run. Get the current address and model with:

```bash
localcode api
```

In a session, open `/server` or click the dot beside **localcode** in the bottom bar.
The dot is green when the model server is ready, amber while downloading or loading,
and grey when unavailable. Status refreshes every two seconds; a failed request
can take another 1.5 seconds to time out. The menu shows the endpoint and model,
and offers explicit actions to copy the API key or an authenticated curl example.
Copying a key puts a secret on your system clipboard.

### Choose a fixed port

Use **Change API port now** in `/server`, or:

```bash
localcode api --port 9234
# Restore automatic port selection:
localcode api --port auto
```

With LocalCode running, the change takes effect immediately without reloading the
model or closing your conversation. LocalCode windows use a stable internal
connection. Existing external connections can finish on the old listener; new
connections must use the new address. Update the base URL in other apps. The API
key stays the same during a port change.

Ports must be between 1024 and 65535. Invalid, occupied or internal service ports
are rejected without changing the current endpoint. The new listener starts
before the old one retires. If saving the preference fails, the new listener is
closed and the original remains active. Concurrent changes are rejected; refresh
`/server` before retrying. After a timeout, refresh to check whether the change
completed before trying again.

The preference is saved in `server-port.json` in LocalCode's runtime directory.
When no LocalCode service is running, the CLI saves it for the next launch.
An occupied saved port produces a startup error; `localcode api --port auto`
restores automatic selection. The API key still changes between runs.

The public API is a loopback TCP forwarding listener. It preserves the model
server's authentication and streaming responses, with bounded buffers, up to 64
simultaneous external connections and a 30-minute inactivity timeout. LocalCode's
own inference does not pass through this listener.

### Connect another application

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

This is the model inference API. LocalCode currently consumes MCP servers as a client; it does not expose its coding actions as an MCP server.

## Typed decisions (System One)

A decision model answers typed questions about a state instead of writing text: a choice between your own labels, a yes/no probability, or a score. The model list has one, **OpenJev 27B (decision model)**. It is not a coding model, and its weights are CC BY-NC 4.0, so research and non-commercial use only.

Load it from `/models`, then post to `/v1/systemone` on the same local endpoint:

```python
import json
import subprocess
import urllib.request

info = json.loads(subprocess.check_output(
    ["localcode", "api", "--json", "--show-key"], text=True
))
body = {
    "state": "Customer message: I was charged twice and nobody has replied.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": None, "shipping": None, "technical": None}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today", "right now"]},
    },
}
request = urllib.request.Request(
    info["base_url"] + "/systemone",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + info["api_key"]},
)
print(json.load(urllib.request.urlopen(request))["answers"])
```

Each answer carries a probability for every option. With the vision projector downloaded, an `images` list of data URLs lets the questions refer to a screenshot. A model that is not a decision model answers this endpoint with a 501 error.

LocalCode's [Decisions screen and CLI](/localcode/guides/decisions) call this endpoint through the authenticated supervisor. Normal Chat uses chat completions.
