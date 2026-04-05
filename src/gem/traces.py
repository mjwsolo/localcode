from __future__ import annotations

import json
from pathlib import Path

from .session import SessionStore


def export_training_traces(output_path: Path) -> tuple[int, Path]:
    store = SessionStore()
    rows = []
    for session_id, _created_at, _repo_root in store.list_sessions():
        session = store.load(session_id)
        for idx in range(1, len(session.messages)):
            current = session.messages[idx]
            previous = session.messages[idx - 1]
            if current.get("role") != "assistant":
                continue
            if previous.get("role") != "user":
                continue
            rows.append(
                {
                    "session_id": session.session_id,
                    "profile": session.profile,
                    "model": session.model,
                    "prompt": previous.get("content", ""),
                    "response": current.get("content", ""),
                    "events": session.events[-20:],
                }
            )
    output_path.write_text("\n".join(json.dumps(row) for row in rows))
    return len(rows), output_path
