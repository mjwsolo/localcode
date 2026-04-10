# Troubleshooting

## "Unsupported cache type: turbo4"

Stock llama-server is running instead of the TurboQuant fork.

**Fix**: Delete `~/.gem/config.toml` and re-run `localcode` to trigger auto-build. Or run `localcode setup` to reconfigure.

## Server fails to start

Check the server logs:

```bash
ls ~/.gem/jobs/*.log
cat ~/.gem/jobs/<latest>.log
```

Usually a memory issue. Close heavy apps (browsers, Docker) and retry.

## GPU password prompt every boot

Normal on 16GB Macs. The sysctl GPU unlock resets on reboot — this is by design (safe, no permanent changes). 24GB+ Macs skip this automatically.

## Build fails with cmake errors

Make sure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

## Slow response (18 tok/s instead of 27)

The GPU unlock isn't active. Check:

```bash
sysctl iogpu.wired_limit_mb
```

If it shows the default (~11GB), run:

```bash
sudo sysctl iogpu.wired_limit_mb=14336
```

Or restart LocalCode — it auto-prompts for this.

## Model outputs garbage or `<eos>` spam

Usually a temperature issue. Check your config:

```bash
cat ~/.gem/config.toml | grep temperature
```

Should be `0.7` or higher. Google recommends `1.0` for Gemma 4.

## "Out of memory" during generation

Context window is too large for available RAM. Options:

1. Close other apps to free memory
2. Reduce context: set `max_context_chars = 20000` in config
3. Use `speed` mode (CPU-only, smaller context)

## pytest picks up vendored tests

Run tests from the repo root with:

```bash
pytest tests
```

## Something else?

```bash
localcode status
```

This checks your runtime, model, config, and reports what's wrong.

!!! tip
    If you're stuck, check [GitHub Issues](https://github.com/mjwsolo/localcode/issues) or open a new one.
