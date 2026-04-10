---
hide:
  - navigation
  - toc
---

## The local AI coding agent

Run Gemma 4 26B on your Mac. LocalCode reads your repo, edits files, runs commands, and verifies changes without sending your code to the cloud by default.

=== "pip"

    ```bash
    pip install localcode && localcode
    ```

=== "pipx"

    ```bash
    pipx install localcode
    ```

=== "source"

    ```bash
    git clone https://github.com/mjwsolo/localcode.git
    cd localcode
    pip install -e .
    ```

<div class="landing-demo">
  <div class="landing-demo__left">
    <div class="landing-demo__line">I’ll help you add JWT auth to the API and update the tests.</div>
    <div class="landing-demo__line landing-demo__line--muted">• Listed src/api</div>
    <div class="landing-demo__line landing-demo__line--muted">• Read src/api/server.py</div>
    <div class="landing-demo__line landing-demo__line--muted">• Read tests/test_auth.py</div>
    <div class="landing-demo__task">
      <div class="landing-demo__task-title">Explore task</div>
      <div class="landing-demo__task-line">├ Read src/api/server.py</div>
      <div class="landing-demo__task-line">├ Edit auth middleware and routes</div>
      <div class="landing-demo__task-line">├ Update tests/test_auth.py</div>
      <div class="landing-demo__task-line">└ Run pytest tests/test_auth.py</div>
    </div>
  </div>
  <div class="landing-demo__right">
    <div class="landing-demo__panel-title">Implementing JWT auth</div>
    <div class="landing-demo__meta">Context</div>
    <div class="landing-demo__meta-line">26,081 tokens</div>
    <div class="landing-demo__meta-line">4 files read</div>
    <div class="landing-demo__meta-line">$0.00 spent</div>
    <div class="landing-demo__meta landing-demo__meta--spaced">Runtime</div>
    <div class="landing-demo__meta-line">Gemma 4 26B</div>
    <div class="landing-demo__meta-line">TurboQuant llama-server</div>
    <div class="landing-demo__meta-line">Apple Silicon local</div>
  </div>
</div>

### What is LocalCode?

LocalCode is an open source coding agent that helps you write code in your terminal while keeping the runtime local-first.

<div class="landing-list">
  <div class="landing-list__row">
    <span class="landing-list__mark">[*]</span>
    <strong>Build and fix</strong>
    <span>Implement features, trace bugs, and make direct file changes.</span>
  </div>
  <div class="landing-list__row">
    <span class="landing-list__mark">[*]</span>
    <strong>Understand repos</strong>
    <span>Read files, search patterns, and explain architecture clearly.</span>
  </div>
  <div class="landing-list__row">
    <span class="landing-list__mark">[*]</span>
    <strong>Run and verify</strong>
    <span>Use shell commands, tests, and linters to check the result.</span>
  </div>
  <div class="landing-list__row">
    <span class="landing-list__mark">[*]</span>
    <strong>Small-hardware ready</strong>
    <span>Runs Gemma 4 26B locally on 16GB Apple Silicon with TurboQuant KV compression.</span>
  </div>
  <div class="landing-list__row">
    <span class="landing-list__mark">[*]</span>
    <strong>Private by default</strong>
    <span>Keep code, prompts, and file operations on your machine in the local path.</span>
  </div>
</div>

[Read docs](docs-overview.md)

### Built for privacy first

LocalCode is designed for privacy-sensitive environments. The local path keeps code, prompts, and file operations on your machine instead of routing them through a hosted coding product.

### FAQ

??? faq "What is LocalCode?"

    LocalCode is a terminal-first coding agent that runs Gemma 4 26B locally on your Mac. It can read code, edit files, run commands, and iterate on tasks end-to-end.

??? faq "How do I use LocalCode?"

    Install it, open any project directory, and run `localcode`. Then describe the task in plain English, like fixing a bug, adding a feature, or explaining part of the codebase.

??? faq "Do I need API keys or a cloud account?"

    No. LocalCode is designed to run locally with Gemma 4 26B on your Mac, so you do not need API keys for the default setup.

??? faq "Can it only work in the terminal?"

    The main workflow is terminal-first. That keeps the tool loop explicit: read files, edit code, run commands, and verify results in the same place you already work.

??? faq "What about privacy?"

    The model, prompts, code reads, and file edits stay on your machine in the default local setup. Your code does not need to be sent to a hosted model provider.

??? faq "Is LocalCode open source?"

    Yes. The app, docs, and supporting runtime work are open under Apache 2.0.

<div class="newsletter-block">
  <h3>Be the first to know when we release new products</h3>
  <p class="newsletter-copy">Join the waitlist for early access.</p>
  <form class="newsletter-form" data-newsletter-form>
    <input
      class="newsletter-input"
      type="email"
      name="email"
      placeholder="Email address"
      autocomplete="email"
      required
    >
    <button class="newsletter-button" type="submit">Subscribe</button>
  </form>
</div>

<div class="footer-links">
  <a href="https://github.com/mjwsolo/localcode">GitHub</a>
  <a href="docs-overview.md">Docs</a>
  <a href="https://github.com/mjwsolo/localcode/releases">Changelog</a>
  <a href="enterprise.md">Enterprise</a>
  <a href="https://x.com/">X</a>
</div>
