# The end of the run: the agent has built, run and opened Git Wrapped.
Source common.tape
Output out/step-4-verify.png
Hide
Type "cd $LC_DEMO_PROJECT; clear"
Enter
Sleep 500ms
Type "localcode --model Qwen3.6-35B-A3B-UD-Q8_K_XL"
Enter
Sleep 30s  # TUNE: model load + first paint; the longest cold start you expect
Type@5ms "Build Git Wrapped for this repo, a Spotify-Wrapped-style page for its git history. First write wrapped.py (standard library only) that reads git log and writes wrapped.json: total commits, top contributor, busiest weekday and hour, longest streak of days with commits, most-edited file, commits per day, and a one-word summary of the year. Then write one self-contained wrapped.html that embeds that JSON and shows full-screen animated cards, one per stat, plus a commit heatmap and a closing 'your year in one word' card. Arrow keys or a click move between cards. No network, no CDN. Run wrapped.py, then open wrapped.html."
Sleep 500ms
Enter
Sleep 240s  # TUNE: whole run hidden; must exceed the slowest run on the recording Mac
Show
Sleep 10s  # TUNE: confirm the final summary is on screen (wrapped.json and wrapped.html written)
Screenshot out/step-4-verify-final.png
Sleep 2s
Hide
Type "/exit"
Enter
Sleep 2s
