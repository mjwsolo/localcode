# README hero: launch, type the Git Wrapped prompt, watch the agent build, see the result.
# Hide/Show cuts out the long waits so about 35-45 s of the run stays visible.
Source common.tape
Output out/first-change.png
Hide
Type "cd $LC_DEMO_PROJECT; clear"
Enter
Sleep 500ms
Type "localcode --model Qwen3.6-35B-A3B-UD-Q8_K_XL"
Enter
Sleep 30s  # TUNE: model load + first paint; the longest cold start you expect
Show
Sleep 2s
Type@8ms "Build Git Wrapped for this repo, a Spotify-Wrapped-style page for its git history. First write wrapped.py (standard library only) that reads git log and writes wrapped.json: total commits, top contributor, busiest weekday and hour, longest streak of days with commits, most-edited file, commits per day, and a one-word summary of the year. Then write one self-contained wrapped.html that embeds that JSON and shows full-screen animated cards, one per stat, plus a commit heatmap and a closing 'your year in one word' card. Arrow keys or a click move between cards. No network, no CDN. Run wrapped.py, then open wrapped.html."
Sleep 800ms
Enter
Sleep 12s  # TUNE: agent plans and starts writing wrapped.py
Hide
Sleep 70s  # TUNE: skipped, agent writes wrapped.py
Show
Sleep 5s  # TUNE: show the script landing
Hide
Sleep 80s  # TUNE: skipped, agent writes wrapped.html
Show
Sleep 5s  # TUNE: show the page being written
Hide
Sleep 60s  # TUNE: skipped, agent runs wrapped.py, checks output, opens the page
Show
Sleep 8s  # TUNE: must land on the finished summary, not mid-run
Screenshot out/first-change-final.png
Sleep 3s
Hide
Type "/exit"
Enter
Sleep 2s
