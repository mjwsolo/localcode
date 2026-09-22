Source common.tape
Output out/first-change.gif
Set Height 720
Hide
Type "cd $LC_DEMO_PROJECT; clear"
Enter
Sleep 500ms
Type "localcode --model gemma-4-12b-it-UD-Q4_K_XL"
Enter
Sleep 25s
Show
Sleep 1s
Type "Implement the retry decorator in retry.py so every test in test_retry.py passes. Then run: python3 -m pytest -q"
Sleep 600ms
Enter
Sleep 45s
Screenshot out/first-change-final.png
Sleep 2s
Hide
Type "/exit"
Enter
Sleep 2s
