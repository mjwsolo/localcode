# agent-ts — localcode's agent plane

A **pinned Pi distribution**, not a fork:

```
localcode agent plane = pi @ PINNED_PI  +  extensions/*
```

We carry zero patches to pi. If a patch ever becomes unavoidable it goes in
`patches/`, numbered and individually applicable, with an entry in a PATCHES.md
saying when it can be deleted — the same contract `llama-cpp-turboquant` carries
for llama.cpp, for the same reason.

## Build

```bash
npm install && ./scripts/build.sh      # → dist/localcode-agent (71 MB, ~26 MB in the wheel)
```

## Verify

```bash
./scripts/smoke.sh                     # router + provider + a real 12B agentic turn
```

## Two upstreams, one loop each

| Upstream | Pin | Loop | Breaks as |
| --- | --- | --- | --- |
| pi | `PINNED_PI` | `.github/workflows/pi-bump.yml`, Tuesdays | extension hook renamed/removed → contract test red |
| llama.cpp | `patches/PINNED_UPSTREAM` | `upstream-bump.yml`, Mondays | **router mode regressed → smoke step 1/2 red** |

The llama.cpp loop already exists. What changed is that **router mode is now
load-bearing**: before pi, localcode drove single-model `llama-server`; now the
front end discovers and loads models through `/models`, `/models/load`,
`/models/unload`. `scripts/smoke.sh` asserts those routes, so an upstream
refactor cannot silently break the front end between releases.

## Extensions

| File | Replaces | Status |
| --- | --- | --- |
| `localcode-provider.ts` | model discovery + naming | **working** — registers the whole catalog headlessly, no `/login` |
| `localcode-runtime.ts` | `runtime.py` local-model behavior | Phase 1 |
| `localcode-safety.ts` | `permissions_v2`, `execution_policy`, `injection_defense` | Phase 2 |

Every extension gets a contract test asserting the pi hooks it depends on still
exist and still fire. That test, not the pin, is what makes weekly bumps safe.
