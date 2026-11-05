from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .agent import run_agent
from .llm import OllamaLLMClient, RecordingLLMClient, ReplayLLMClient, ScriptedLLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal finance agent")
    parser.add_argument("--data", required=True, help="Directory containing CSV inputs")
    parser.add_argument("--out", required=True, help="Path to write report.json")
    parser.add_argument("--replay", action="store_true", help="Replay saved LLM responses without calling a model")
    parser.add_argument("--recording", default="llm_recording.json", help="LLM recording JSON for replay and persistence")
    parser.add_argument("--provider", choices=["fake", "ollama"], default=os.environ.get("FINANCE_AGENT_PROVIDER", "fake"), help="Which LLM backend to use")
    parser.add_argument("--model", default=os.environ.get("FINANCE_AGENT_MODEL", "llama3.1:8b"), help="Model name to use for the real LLM provider")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    recording_path = Path(args.recording)
    if args.replay or os.environ.get("FINANCE_AGENT_REPLAY") == "1":
        llm = ReplayLLMClient(recording_path)
    else:
        if args.provider == "ollama":
            base_llm = OllamaLLMClient(model=args.model)
        else:
            base_llm = ScriptedLLMClient([
                {"tool": "categorize_uncategorized", "arguments": {}},
                {"tool": "compute_totals", "arguments": {}},
                {"tool": "write_report", "arguments": {}},
                {"tool": "invalid", "arguments": {}},
            ])
        llm = RecordingLLMClient(base_llm, recording_path)
    try:
        report = run_agent(args.data, args.out, llm)
    except Exception as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Financial agent failed: {exc}") from exc
    if report:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
