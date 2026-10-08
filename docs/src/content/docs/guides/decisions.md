---
title: Chat and Decisions
description: Choose normal chat or typed System One inference with OpenJev.
---

OpenJev supports two ways to answer. **Chat** generates text in the usual conversation. **Decisions** returns a choice, a yes/no probability, or an ordered score about content you supply.

## Choose a mode

Load **OpenJev** from `/models`, then type `/mode` and choose **Decisions**. You can also use `/decide` or start with:

```bash
localcode --mode decisions
```

In the decision screen, enter a question and optional content. Choose an answer type:

| Type | Input | Result |
| --- | --- | --- |
| Yes / no | A question such as “Was the customer charged twice?” | Probabilities for yes and no |
| Choice | Between 2 and 52 different labels, one per line | A selected label and probabilities for every label |
| Score | Between 2 and 10 levels, lowest to highest | Probabilities for each level and a weighted score from zero to the last level's index |

Optionally enter a PNG, JPEG or WebP image path, up to 5 MB. Image decisions require OpenJev's vision projector. Select **Run decision** to evaluate; **Back to chat** returns to the conversation you left. Chat is the default on each launch.

Decisions call `/v1/systemone` directly through the authenticated local supervisor. They do not send a chat prompt, execute coding tools, or take actions. The form and result stay in the current interface's memory; switching back preserves your chat session. A model switch during evaluation invalidates its result.

The windows share the loaded model. Decisions currently require OpenJev; other models produce an explicit error. There is no automatic fallback to a text answer.

## Use the CLI

Keep LocalCode open with OpenJev loaded, then run:

```bash
localcode decide --question "Which team should handle this?" \
  --type choice --option billing --option delivery \
  --state "I was charged twice for the same order."
```

The command prints JSON with the model, answers and usage. `--type yes-no` is the default. Repeat `--option` for each choice or score level. Use `--state-file PATH` for UTF-8 content, or `--image PATH` for an image. Run `localcode decide --help` for the full options.

For multiple questions in one request, use the [Local API](/localcode/guides/local-api#typed-decisions-system-one).

## Interpret the result

Probabilities describe the model's prediction. Check calibration and accuracy on your own examples before using them to route work; a high probability does not authorize an action. Score levels are ordinal, so a weighted score depends on the levels you supplied.

OpenJev's weights use **CC BY-NC 4.0**. The catalog identifies their non-commercial license.
