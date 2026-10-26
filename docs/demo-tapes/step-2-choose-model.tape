# The /models picker: model list, then the quants for one model.
Source common.tape
Output out/step-2-choose-model.png
Hide
Type "cd $LC_DEMO_PROJECT; clear"
Enter
Sleep 500ms
Type "localcode --model Qwen3.6-35B-A3B-UD-Q8_K_XL"
Enter
Sleep 30s  # TUNE: model load + first paint; the longest cold start you expect
Type "/models"
Enter
Sleep 2s
Show
Sleep 2s
Down
Sleep 900ms
Down
Sleep 900ms
Up
Sleep 900ms
Up
Sleep 1.5s
Enter
Sleep 3s  # TUNE: quant list for the highlighted model, let the fit column be readable
Down
Sleep 900ms
Down
Sleep 900ms
Up
Sleep 2s
Hide
Escape
Sleep 500ms
Escape
Sleep 500ms
Type "/exit"
Enter
Sleep 2s
