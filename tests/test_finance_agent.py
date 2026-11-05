import json
from pathlib import Path

from finance_agent.agent import build_report, parse_category_response, run_agent
from finance_agent.data import load_transactions
from finance_agent.llm import ScriptedLLMClient, parse_tool_choice


class CategorizationAndToolLLMClient:
    def __init__(self, tools):
        self.tools = list(tools)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Select one of these tools"):
            return json.dumps({"tool": self.tools.pop(0)})
        return json.dumps({"category": "Food"})


def test_llm_json_parsers_accept_markdown_wrapped_responses():
    assert parse_tool_choice('```json\n{"tool": "compute_totals"}\n```')["tool"] == "compute_totals"
    assert parse_category_response('Here is the result:\n```json\n{"category": "Food"}\n```') == "Food"
    assert parse_category_response('```\n{"category": "Income"}\n```') == "Income"


def test_parser_normalizes_mixed_dates_and_flags_odd_rows():
    transactions = load_transactions(Path("sample_data"))
    assert any(t.description == "WHOLEFDS MKT #10452" and t.date.isoformat() == "2024-01-02" for t in transactions)
    assert any(t.description == "SHELL OIL 5748291" and t.date.isoformat() == "2024-01-03" for t in transactions)
    pending = next(t for t in transactions if t.description == "PENDING AUTH TARGET T-8821")
    assert pending.amount == -0.0
    assert "Zero-amount pending row" in pending.flagged


def test_totals_and_by_category_are_computed_in_code():
    transactions = load_transactions(Path("sample_data"))
    report = build_report(transactions)
    totals = report["totals"]
    assert totals["income"] > 0
    assert totals["expenses"] > 0
    assert totals["net"] == round(totals["income"] - totals["expenses"], 2)
    assert report["by_category"]


def test_agent_handles_invalid_tool_choice_and_still_writes_report(tmp_path):
    llm = ScriptedLLMClient([
        "bad response",
        {"tool": "categorize_uncategorized", "arguments": {}},
        {"tool": "compute_totals", "arguments": {}},
        {"tool": "write_report", "arguments": {}},
    ])
    output_path = tmp_path / "report.json"
    report = run_agent(Path("sample_data"), output_path, llm, step_limit=5)
    assert output_path.exists()
    assert report["totals"]["income"] > 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]


def test_agent_passes_tool_observations_to_next_model_decision(tmp_path):
    llm = CategorizationAndToolLLMClient(["write_report", "compute_totals", "write_report"])
    output_path = tmp_path / "report.json"
    initial_transactions = load_transactions(Path("sample_data"))
    missing_count = sum(tx.category in (None, "") for tx in initial_transactions)

    report = run_agent(Path("sample_data"), output_path, llm, step_limit=3)

    tool_prompts = [prompt for prompt in llm.prompts if prompt.startswith("Select one of these tools")]
    category_prompts = [prompt for prompt in llm.prompts if prompt.startswith("Categorize this one transaction")]
    assert missing_count > 10
    assert len(category_prompts) == missing_count
    assert all(prompt.count("\nDescription:") == 1 for prompt in category_prompts)
    assert "No tools have run yet" in tool_prompts[0]
    assert "Categorization tool completed" in tool_prompts[1]
    assert all(tx["category"] for tx in report["transactions"])
