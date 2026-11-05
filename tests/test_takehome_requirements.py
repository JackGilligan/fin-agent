import csv
import json
import sys
from datetime import date

import pytest

from finance_agent.agent import (
    build_report,
    categorize_uncategorized,
    compute_totals,
    parse_category_response,
    run_agent,
)
from finance_agent.data import ALLOWED_CATEGORIES, Transaction, load_transactions, normalize_date
from finance_agent.llm import (
    OllamaLLMClient,
    RecordingLLMClient,
    ReplayLLMClient,
    ScriptedLLMClient,
)


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sample_inputs(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        data_dir / "income.csv",
        ["Date", "Description", "Amount", "Source"],
        [["2024-01-01", "Salary", "100.00", "Employer"]],
    )
    _write_csv(
        data_dir / "expenses.csv",
        ["Date", "Description", "Amount", "Category"],
        [["2024-01-02", "Grocery Store", "-20.00", "Food"]],
    )
    _write_csv(
        data_dir / "bank_statement.csv",
        ["Date", "Description", "Amount", "Balance", "Type"],
        [
            ["2024-01-02", "Grocery Store", "-20.00", "80.00", "Debit"],
            ["2024-01-03", "Interest Credit", "1.00", "81.00", "Credit"],
        ],
    )
    _write_csv(
        data_dir / "transactions_uncategorized.csv",
        ["Date", "Description", "Amount", "Category"],
        [["01/04/2024", "Corner Cafe", "-4.00", ""]],
    )


class PromptAwareLLM:
    def __init__(self, tool_choices, categories):
        self.tool_choices = iter(tool_choices)
        self.categories = iter(categories)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Select one of these tools"):
            return json.dumps({"tool": next(self.tool_choices)})
        return json.dumps({"category": next(self.categories)})


def test_ingests_all_four_csvs_normalizes_dates_and_collapses_exact_overlap(tmp_path):
    _write_sample_inputs(tmp_path)

    transactions = load_transactions(tmp_path)

    assert {tx.source_file for tx in transactions} == {
        "income.csv",
        "expenses.csv",
        "bank_statement.csv",
        "transactions_uncategorized.csv",
    }
    assert len([tx for tx in transactions if tx.description == "Grocery Store"]) == 1
    cafe = next(tx for tx in transactions if tx.description == "Corner Cafe")
    assert cafe.date.isoformat() == "2024-01-04"
    assert normalize_date("2024-1-6") == date(2024, 1, 6)
    assert normalize_date("01/03/2024") == date(2024, 1, 3)


def test_missing_required_csv_is_an_unrecoverable_input_error(tmp_path):
    _write_sample_inputs(tmp_path)
    (tmp_path / "bank_statement.csv").unlink()

    with pytest.raises(FileNotFoundError, match="bank_statement.csv"):
        load_transactions(tmp_path)


def test_llm_categorizes_every_missing_transaction_individually_without_batch_cap():
    transactions = [
        Transaction(date(2024, 1, index), f"Merchant {index}", -float(index))
        for index in range(1, 13)
    ]

    class CategoryLLM:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt):
            self.prompts.append(prompt)
            return json.dumps({"category": "Food"})

    llm = CategoryLLM()
    categorize_uncategorized(transactions, llm)

    assert len(llm.prompts) == len(transactions)
    assert all(prompt.count("\nDescription:") == 1 for prompt in llm.prompts)
    assert all(tx.category == "Food" for tx in transactions)


def test_invalid_or_unlisted_llm_category_falls_back_to_fixed_categories():
    transactions = [
        Transaction(date(2024, 1, 1), "SAFEWAY MKT #10452", -12.00),
        Transaction(date(2024, 1, 2), "Unknown Merchant", -3.00),
    ]
    llm = ScriptedLLMClient([
        '{"category": "Not A Category"}',
        "not json",
    ])

    categorize_uncategorized(transactions, llm)

    assert transactions[0].category == "Food"
    assert transactions[1].category == "Other"
    assert all(tx.category in ALLOWED_CATEGORIES for tx in transactions)
    assert parse_category_response('```json\n{"category":"food"}\n```') == "Food"


def test_totals_and_category_sums_are_deterministic_and_exact():
    transactions = [
        Transaction(date(2024, 1, 1), "Paycheck", 1000.00, "Income"),
        Transaction(date(2024, 1, 2), "Groceries", -40.00, "Food"),
        Transaction(date(2024, 1, 3), "Internet", -10.00, "Utilities"),
        Transaction(date(2024, 1, 4), "Pending", 0.00, "Other"),
    ]

    totals = compute_totals(transactions)

    assert totals == {
        "income": 1000.00,
        "expenses": 50.00,
        "net": 950.00,
        "savings_rate": 0.95,
        "by_category": {"Food": 40.00, "Utilities": 10.00},
    }


def test_llm_cannot_override_report_arithmetic(tmp_path):
    _write_sample_inputs(tmp_path)
    llm = ScriptedLLMClient([
        {"tool": "categorize_uncategorized"},
        {"category": "Income"},
        {"category": "Food"},
        {
            "tool": "compute_totals",
            "arguments": {
                "income": 999999,
                "expenses": 0,
                "net": 999999,
                "savings_rate": 1,
            },
        },
        {"tool": "write_report", "arguments": {"totals": {"income": 0}}},
    ])

    report = run_agent(tmp_path, tmp_path / "report.json", llm)

    assert report["totals"] == {
        "income": 101.00,
        "expenses": 24.00,
        "net": 77.00,
        "savings_rate": 0.76,
    }


def test_report_satisfies_output_contract_and_uses_normalized_dates(tmp_path):
    _write_sample_inputs(tmp_path)
    transactions = load_transactions(tmp_path)
    report = build_report(transactions)

    assert set(report) >= {"totals", "by_category", "flagged", "transactions", "summary"}
    assert set(report["totals"]) >= {"income", "expenses", "net", "savings_rate"}
    assert isinstance(report["flagged"], list)
    assert report["summary"]
    assert "$101.00" in report["summary"]
    assert "$24.00" in report["summary"]
    assert "$77.00" in report["summary"]
    for tx in report["transactions"]:
        assert set(tx) >= {"date", "description", "amount", "category"}
        assert date.fromisoformat(tx["date"]).isoformat() == tx["date"]


def test_bad_tool_and_category_responses_still_produce_report(tmp_path):
    _write_sample_inputs(tmp_path)
    llm = ScriptedLLMClient([
        "garbage tool response",
        "garbage category response",
        '{"category":"Food"}',
        '{"tool":"write_report"}',
    ])
    output_path = tmp_path / "report.json"

    report = run_agent(tmp_path, output_path, llm, step_limit=1)

    assert output_path.exists()
    assert report["totals"]["income"] == 101.00
    assert report["totals"]["expenses"] == 24.00
    assert all(tx["category"] for tx in report["transactions"])


def test_recording_and_replay_reproduce_report_without_live_model(tmp_path):
    data_dir = tmp_path / "data"
    _write_sample_inputs(data_dir)
    recording_path = tmp_path / "llm_recording.json"
    live_output = tmp_path / "report.json"
    replay_output = tmp_path / "replay_report.json"
    live_llm = RecordingLLMClient(
        ScriptedLLMClient([
            {"tool": "categorize_uncategorized"},
            {"category": "Income"},
            {"category": "Income"},
            {"category": "Food"},
            {"tool": "compute_totals"},
            {"tool": "write_report"},
        ]),
        recording_path,
    )

    live_report = run_agent(data_dir, live_output, live_llm)
    replay_report = run_agent(data_dir, replay_output, ReplayLLMClient(recording_path))

    saved_recording = json.loads(recording_path.read_text(encoding="utf-8"))
    assert len(saved_recording["responses"]) == 6
    assert replay_report == live_report
    assert json.loads(replay_output.read_text(encoding="utf-8")) == live_report


def test_replay_rejects_mismatched_prompts_and_exhausted_recordings(tmp_path):
    recording_path = tmp_path / "llm_recording.json"
    recording_path.write_text(
        json.dumps({"responses": [{"prompt": "recorded prompt", "response": '{"category":"Food"}'}]}),
        encoding="utf-8",
    )
    replay = ReplayLLMClient(recording_path)

    with pytest.raises(RuntimeError, match="prompt mismatch"):
        replay.complete("different prompt")
    assert replay.complete("recorded prompt") == '{"category":"Food"}'
    with pytest.raises(RuntimeError, match="no response for call 2"):
        replay.complete("another call")


def test_replay_rejects_tool_response_recorded_for_category_prompt(tmp_path):
    recording_path = tmp_path / "misaligned-recording.json"
    recording_path.write_text(
        json.dumps({
            "responses": [{
                "prompt": "Categorize this one transaction: details",
                "response": '{"tool":"compute_totals","arguments":{}}',
            }]
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expects a transaction category"):
        ReplayLLMClient(recording_path).complete("Categorize this one transaction: details")


def test_ollama_client_uses_json_mode_temperature_zero_and_returns_text(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"tool":"compute_totals"}'}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("finance_agent.llm.requests.post", fake_post)
    client = OllamaLLMClient(model="test-model", base_url="http://ollama.test")

    result = client.complete("choose a tool")

    assert result == '{"tool":"compute_totals"}'
    assert captured["url"] == "http://ollama.test/api/generate"
    assert captured["json"]["format"] == "json"
    assert captured["json"]["options"]["temperature"] == 0


def test_cli_returns_zero_on_success_and_nonzero_on_missing_input(tmp_path, monkeypatch):
    from finance_agent.__main__ import main

    data_dir = tmp_path / "data"
    _write_sample_inputs(data_dir)
    output_path = tmp_path / "report.json"
    recording_path = tmp_path / "cli-recording.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finance_agent",
            "--data",
            str(data_dir),
            "--out",
            str(output_path),
            "--recording",
            str(recording_path),
        ],
    )
    assert main() == 0
    assert output_path.exists()
    assert recording_path.exists()

    missing_data = tmp_path / "missing"
    missing_data.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["finance_agent", "--data", str(missing_data), "--out", str(tmp_path / "bad.json")],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code
