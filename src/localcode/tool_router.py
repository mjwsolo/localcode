"""Smart tool routing — selects relevant tools per query instead of dumping all tools.

Problem: Small local models (2B-4B) choke when given 30+ tool schemas.
They ignore tools entirely or hallucinate tool names.

Solution: Analyze the user's message to predict which tools are needed,
send only those (3-8 tools), and progressively add more if the model
asks for capabilities it doesn't have.

This is a zero-cost classifier — no LLM call, just pattern matching.
It runs in <1ms and dramatically improves tool-use accuracy on small models.

Architecture:
  1. Intent detection (keyword + pattern matching)
  2. Tool selection (map intents → tool sets)
  3. Always-include base tools
  4. Progressive disclosure (retry with more tools on failure)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RoutingResult:
    """Which tools to send and why."""
    tool_names: set[str]
    intents: list[str]
    confidence: float  # 0-1, how confident we are in the routing

    @property
    def summary(self) -> str:
        return f"{len(self.tool_names)} tools for [{', '.join(self.intents)}]"


# ── Intent patterns ──────────────────────────────────────────────────────
# Each intent maps to keywords/patterns that trigger it

INTENT_PATTERNS: dict[str, list[str]] = {
    "quality_create": [
        r"\bclone\b", r"\bpolish\b", r"\bfeel like\b", r"\blook like\b",
        r"\bauthentic\b", r"\bhigh quality\b", r"\bfidelity\b",
        r"\bplayable\b", r"\bsonic\b", r"\bbeautiful\b",
    ],
    "file_read": [
        r"\bread\b", r"\bshow\b", r"\bcat\b", r"\bview\b", r"\blook at\b",
        r"\bopen\b", r"\bcheck\b.*file", r"\bwhat.s in\b",
        r"\.[a-z]{1,4}\b",  # file extensions like .py, .ts, .json
        r"src/", r"lib/", r"app/", r"test",
    ],
    "file_write": [
        r"\bwrite\b", r"\bcreate\b", r"\breate\b",
        r"\bnew file\b", r"\badd\b.*file", r"\bmake\b.*file",
        r"\bsave\b", r"\bgenerate\b", r"\bpopulate\b",
        r"\bscaffold\b", r"\bboilerplate\b", r"\bstub\b",
        r"\bsetup\b.*file", r"\binit\b.*file",
        r"\.py\b", r"\.js\b", r"\.ts\b", r"\.html\b",
        r"\bcalled\b.*\.\w{2,4}\b",
        r"\bnamed\b.*\.\w{2,4}\b",
        r"\bmake\b.*\bapp\b", r"\bbuild\b.*\bapp\b",  # "make an app"
        r"\bmake\b.*\bgame\b", r"\bbuild\b.*\bgame\b",  # "make a game"
        r"\bmake\b.*\bscript\b", r"\bmake\b.*\bprogram\b",
        r"\bmake\b.*\btool\b", r"\bmake\b.*\bbot\b",
    ],
    "file_edit": [
        r"\bedit\b", r"\bchange\b", r"\bupdate\b", r"\bmodify\b", r"\bfix\b",
        r"\brefactor\b", r"\breplace\b", r"\brename\b", r"\bremove\b.*from",
        r"\badd\b.*to\b", r"\binsert\b", r"\bdelete\b.*line", r"\bpatch\b",
    ],
    "search_code": [
        r"\bfind\b", r"\bsearch\b.*code", r"\bgrep\b", r"\bwhere\b.*is\b",
        r"\bwhich file\b", r"\blocate\b", r"\blook for\b",
        r"\bdefinition\b", r"\bimport\b", r"\bclass\b", r"\bfunction\b",
        r"\bglob\b", r"\bpattern\b",
    ],
    "time": [
        r"\btime\b", r"\bdate\b", r"\btoday\b", r"\bnow\b", r"\bclock\b",
        r"\bwhat day\b",
    ],
    "web": [
        r"\bsearch\b(?!.*code)", r"\blook up\b", r"\bonline\b", r"\bweb\b",
        r"\bgoogle\b", r"\blatest\b", r"\bnews\b",
        r"\bweather\b", r"\bdocumentation\b", r"\bdocs\b",
        r"\burl\b", r"\bwebsite\b", r"\bfetch\b",
        r"\bworld cup\b", r"\bprice\b", r"\bcost\b",
        # Only current events need web — general knowledge doesn't
    ],
    "shell": [
        r"\brun\b", r"\bexecute\b", r"\bbuild\b", r"\btest\b",
        r"\binstall\b", r"\bnpm\b", r"\bpip\b", r"\bcargo\b",
        r"\bmake\b", r"\bcompile\b", r"\bstart\b.*server",
        r"\bpytest\b", r"\bls\b", r"\bwhich\b",
    ],
    "git": [
        r"\bgit\b", r"\bcommit\b", r"\bdiff\b", r"\bstatus\b",
        r"\bbranch\b", r"\bpush\b", r"\bpull\b", r"\blog\b",
        r"\bhistory\b", r"\bblame\b", r"\bchanges\b",
    ],
}

# ── Intent → tool mapping ────────────────────────────────────────────────

INTENT_TOOLS: dict[str, list[str]] = {
    "quality_create": ["read_file", "write_file", "bash"],
    "time":        ["current_datetime", "bash"],
    "file_read":   ["read_file"],
    "file_write":  ["write_file", "read_file"],
    "file_edit":   ["read_file", "edit_file", "write_file"],
    "search_code": ["grep", "glob", "read_file", "search_code", "list_files"],
    "web":         ["web_search", "web_fetch"],
    "shell":       ["bash"],
    "git":         ["git_status", "git_diff", "git_log", "git_commit"],
}

# Tools always included regardless of intent
BASE_TOOLS = {"bash"}

# Pairs that should always go together
TOOL_BUDDIES: dict[str, list[str]] = {
    "edit_file": ["read_file"],      # always read before edit
    "write_file": ["read_file"],     # check if file exists first
    "git_commit": ["git_status", "git_diff"],
    "web_fetch": ["web_search"],
}


def route_tools(
    user_message: str,
    available_tools: list[str],
    conversation_history: list[dict] | None = None,
    max_tools: int = 10,
    online: bool | None = None,
) -> RoutingResult:
    """Select which tools to send based on the user's message.

    Args:
        user_message: The current user input
        available_tools: All tool names registered in the toolkit
        conversation_history: Recent messages for context
        max_tools: Maximum tools to return

    Returns:
        RoutingResult with selected tool names and detected intents
    """
    text = user_message.lower()
    detected_intents: list[str] = []
    selected_tools: set[str] = set(BASE_TOOLS)
    total_score = 0

    # Score each intent
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        if score > 0:
            detected_intents.append(intent)
            total_score += score
            for tool_name in INTENT_TOOLS.get(intent, []):
                if tool_name in available_tools:
                    selected_tools.add(tool_name)

    # Intent conflict resolution:
    # If creating/editing files, "time" and "web" are about code content, not standalone queries
    if "file_write" in detected_intents or "file_edit" in detected_intents:
        for noise in ("time", "web"):
            if noise in detected_intents:
                detected_intents.remove(noise)
                # Remove tools that were added for the noise intent
                for t in INTENT_TOOLS.get(noise, []):
                    selected_tools.discard(t)
        # Make sure file tools are included
        for t in ("write_file", "read_file", "edit_file"):
            if t in available_tools:
                selected_tools.add(t)

    # Add buddy tools
    for tool in list(selected_tools):
        for buddy in TOOL_BUDDIES.get(tool, []):
            if buddy in available_tools:
                selected_tools.add(buddy)

    # Offline mode: remove web tools if no internet
    if online is not None and not online:
        web_tools = {"web_search", "web_fetch"}
        selected_tools -= web_tools
        if "web" in detected_intents:
            detected_intents.remove("web")

    # If no intents detected, check if it's casual chat (no tools needed)
    if not detected_intents:
        casual_patterns = [
            r"^(hi|hey|hello|yo|sup|what.?s up|how are you|thanks|thank you|ok|bye|cool|nice|lol)\b",
            r"^.{0,15}$",  # very short messages are usually casual
            r"^\?+$",      # just question marks
            r"^(yes|no|yep|nope|sure|nah|hmm|huh|what|why|how)\??$",  # single words
        ]
        import re as _re
        is_casual = any(_re.search(p, text.strip(), _re.IGNORECASE) for p in casual_patterns)
        if is_casual:
            # No tools for casual chat — let the model just respond
            return RoutingResult(tool_names=set(), intents=["chat"], confidence=0.9)
        detected_intents.append("general")
        general_tools = ["read_file", "bash"]  # NO web_search for general questions
        for t in general_tools:
            if t in available_tools:
                selected_tools.add(t)

    # Check conversation history for recent tool patterns
    if conversation_history:
        recent_tools = _extract_recent_tool_names(conversation_history)
        for tool_name in recent_tools:
            if tool_name in available_tools:
                selected_tools.add(tool_name)

    # Cap at max_tools
    if len(selected_tools) > max_tools:
        # Prioritize: base tools + intent-matched tools
        prioritized = list(BASE_TOOLS)
        for intent in detected_intents:
            for t in INTENT_TOOLS.get(intent, []):
                if t not in prioritized and t in available_tools:
                    prioritized.append(t)
        selected_tools = set(prioritized[:max_tools])

    confidence = min(1.0, total_score / 3.0) if total_score > 0 else 0.3

    return RoutingResult(
        tool_names=selected_tools & set(available_tools),
        intents=detected_intents,
        confidence=confidence,
    )


def expand_tools_for_retry(
    current_tools: set[str],
    model_response: str,
    available_tools: list[str],
) -> set[str] | None:
    """If the model said it can't do something, figure out which tool to add.

    Returns expanded tool set, or None if no expansion needed.

    This is the "progressive disclosure" step — if the model says
    "I don't have access to X", we add the relevant tool and retry.
    """
    response_lower = model_response.lower()

    expansion_triggers: dict[str, list[str]] = {
        "search": ["web_search", "web_fetch", "grep"],
        "browse": ["web_search", "web_fetch"],
        "internet": ["web_search", "web_fetch"],
        "file": ["read_file", "write_file", "edit_file", "glob"],
        "time": ["web_search", "bash"],
        "run": ["bash"],
        "test": ["bash"],
        "commit": ["git_commit", "git_status", "git_diff"],
        "diff": ["git_diff"],
    }

    # Check if model expressed inability
    inability_patterns = [
        r"i (?:don.t|do not|cannot|can.t) have access",
        r"i (?:don.t|do not|cannot|can.t) (?:search|browse|access|run)",
        r"i (?:am|'m) (?:unable|not able) to",
        r"(?:no|not) (?:available|capable|able)",
        r"beyond my (?:capabilities|ability)",
    ]

    expressed_inability = any(
        re.search(p, response_lower) for p in inability_patterns
    )

    if not expressed_inability:
        return None

    # Find which tools to add
    new_tools = set(current_tools)
    added = False
    for keyword, tools in expansion_triggers.items():
        if keyword in response_lower:
            for t in tools:
                if t in available_tools and t not in current_tools:
                    new_tools.add(t)
                    added = True

    # Generic expansion: add web_search if not present
    if not added and "web_search" not in current_tools and "web_search" in available_tools:
        new_tools.add("web_search")
        added = True

    return new_tools if added else None


def _extract_recent_tool_names(history: list[dict], lookback: int = 4) -> list[str]:
    """Extract tool names from recent conversation for continuity."""
    tool_names = []
    for msg in history[-lookback:]:
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            if name:
                tool_names.append(name)
    return tool_names
