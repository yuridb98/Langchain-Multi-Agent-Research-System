"""CLI entrypoint for the research pipeline.

Example:
    python main.py --topic "The impact of AI on the job market in 2026"
    python main.py --topic "..." --writer-model gpt-5.4-mini --temperature 0.2
"""

from __future__ import annotations

import argparse
import sys

from src.config.models import AGENT_DEFAULTS, CHEAP_MODELS
from src.core.logging import get_logger
from src.pipeline.pipeline import run_research_pipeline

logger = get_logger(__name__)


def _print_event(step: str, status: str) -> None:
    icon = {"running": "...", "done": "OK", "error": "FAILED"}.get(status, status)
    print(f"[{step:<7}] {icon}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    model_choices = [m.id for m in CHEAP_MODELS]
    parser = argparse.ArgumentParser(description="Run the multi-agent research pipeline.")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--search-model", default=AGENT_DEFAULTS["search"], choices=model_choices)
    parser.add_argument("--reader-model", default=AGENT_DEFAULTS["reader"], choices=model_choices)
    parser.add_argument("--writer-model", default=AGENT_DEFAULTS["writer"], choices=model_choices)
    parser.add_argument("--critic-model", default=AGENT_DEFAULTS["critic"], choices=model_choices)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = {
        "search": args.search_model,
        "reader": args.reader_model,
        "writer": args.writer_model,
        "critic": args.critic_model,
    }

    result = run_research_pipeline(
        args.topic,
        models=models,
        temperature=args.temperature,
        on_event=_print_event,
    )

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.report or "(no report produced)")

    print("\n" + "=" * 60)
    print("CRITIC FEEDBACK")
    print("=" * 60)
    print(result.feedback or "(no feedback produced)")

    if result.errors:
        print("\n" + "=" * 60)
        print("ERRORS")
        print("=" * 60)
        for err in result.errors:
            print(f"- {err}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
