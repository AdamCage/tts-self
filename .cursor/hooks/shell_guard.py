#!/usr/bin/env python3
"""beforeShellExecution: gate risky destructive / data-commit commands."""

from __future__ import annotations

import json
import re
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"permission": "allow"}))
        return 0

    cmd = payload.get("command") or payload.get("tool_input", {}).get("command") or ""
    cmd_l = cmd.lower()

    deny_patterns = [
        (r"git\s+clean\s+-.*[fd]", "Refusing git clean -fd (destructive)."),
        (r"rm\s+-rf\s+(/|~|/home|/mnt)", "Refusing recursive rm of a broad path."),
    ]
    ask_patterns = [
        (r"git\s+add\s+.*\b(data/|runs/|.*\.safetensors|.*\.wav)\b", "Staging data/checkpoints/audio — confirm intentional."),
        (r"hf\s+download\s+.*balalaika|biggest_ru_book", "Large corpus download — confirm intentional."),
        (r"torch\.cuda\.empty_cache|nvidia-smi\s+-r", "GPU reset/clear — confirm intentional."),
    ]

    for pat, msg in deny_patterns:
        if re.search(pat, cmd_l):
            print(
                json.dumps(
                    {
                        "permission": "deny",
                        "user_message": msg,
                        "agent_message": msg + " Choose a narrower, safer command.",
                    }
                )
            )
            return 0

    for pat, msg in ask_patterns:
        if re.search(pat, cmd_l):
            print(
                json.dumps(
                    {
                        "permission": "ask",
                        "user_message": msg,
                        "agent_message": msg,
                    }
                )
            )
            return 0

    print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
