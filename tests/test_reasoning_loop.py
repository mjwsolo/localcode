"""Degenerate reasoning-loop detector — periodicity-based, content-agnostic.

Covers the real failure (a phrase repeating until the length cap), the cases
Codex flagged that a line-based detector would miss (no newlines, chunk-split
phrases), and the false-positive guards (varied reasoning, short input,
legitimately repetitive structured content).
"""
from localcode.agent.reasoning_loop import reasoning_is_looping


def test_detects_repeated_phrase_loop():
    text = "".join("Let me read the file now.\n" for _ in range(60))
    assert reasoning_is_looping(text) is True


def test_detects_two_phrase_oscillation():
    text = "".join(
        ["OK let me call read_file now.\n", "Actually let me read the HTML output.\n"][i % 2]
        for i in range(60)
    )
    assert reasoning_is_looping(text) is True


def test_detects_loop_without_any_newline():
    # Codex's case: the repeated unit has no newline; line-splitting would fail.
    text = "I should read the config file. " * 80
    assert reasoning_is_looping(text) is True


def test_detects_loop_after_legit_preamble():
    # Real reasoning first, THEN degeneration — tail-based detection still fires.
    preamble = (
        "The build error points at DeckCard.tsx where dueDate is a string.\n"
        "I need to parse it before handing it to the scheduler.\n"
    )
    text = preamble + "Let me call read_file. " * 60
    assert reasoning_is_looping(text) is True


def test_does_not_fire_on_varied_reasoning():
    text = (
        "The build fails with a type error in DeckCard.tsx.\n"
        "The prop dueDate is typed as Date but passed as string.\n"
        "I'll change the interface to accept string and parse it.\n"
        "Then the FSRS scheduler needs the parsed Date, not the raw prop.\n"
        "Let me check how ReviewSession consumes dueDate.\n"
        "It calls scheduler.next(card) which expects a Date.\n"
        "So I'll parse at the boundary in DeckCard and keep the model typed.\n"
        "After that I'll re-run tsc to confirm the error clears.\n"
        "Also need to update the seed data which uses ISO strings.\n"
        "And the IndexedDB migration that stored the old shape.\n"
    )
    assert reasoning_is_looping(text) is False


def test_does_not_fire_on_short_input():
    assert reasoning_is_looping("Let me read the file.\n" * 3) is False
    assert reasoning_is_looping("") is False


def test_does_not_fire_on_long_varied_prose():
    words = ["parse", "scheduler", "deck", "review", "migrate", "seed", "render",
             "typecheck", "boundary", "interface", "prop", "commit"]
    text = " ".join(words[(i * 7) % len(words)] + str(i) for i in range(400))
    assert reasoning_is_looping(text) is False
