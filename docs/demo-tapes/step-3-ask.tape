# Typing the Git Wrapped prompt and sending it.
Source common.tape
Output out/step-3-ask.png
Hide
Type "cd $LC_DEMO_PROJECT; clear"
Enter
Sleep 500ms
Type "localcode --model Qwen3.6-35B-A3B-UD-Q8_K_XL"
Enter
Sleep 30s  # TUNE: model load + first paint; the longest cold start you expect
Show
Sleep 1s
Type@25ms "Build Git Wrapped for this repo: a Spotify-Wrapped-style page for its git history. Write wrapped.py (Python standard library only) that reads git log and computes total commits, top contributor, busiest weekday and hour, longest daily streak, most-edited file, commits per day for the last 365 days, and a one-word vibe for the year. It writes wrapped.html with the data inlined in a script tag (no fetch, no CDN), so it opens straight from disk. Make it look great: full-screen cards on bold gradient backgrounds, one stat per card, huge numbers that count up, a GitHub-style heatmap grid of the 365 days, and a closing 'your year in one word' card. Arrow keys or a click move between cards. Run wrapped.py, extract the page's script and check it with node --check, fix anything that fails, then open wrapped.html."
Sleep 1.5s
Enter
Sleep 8s  # TUNE: first tool calls visible, then stop
Hide
Type "/exit"
Enter
Sleep 2s
