"""System prompts for agents — one markdown file per agent, loaded by name."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a system prompt by agent name (e.g. load_prompt("hypothesis"))."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file for agent {name!r}: {path}")
    return path.read_text(encoding="utf-8")
