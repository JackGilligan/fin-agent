from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .data import ALLOWED_CATEGORIES, Transaction, dedupe_transactions, load_transactions
from .llm import LLMClient, parse_json_response, parse_tool_choice


VALID_TOOLS = {"categorize_uncategorized", "compute_totals", "write_report"}
logger = logging.getLogger(__name__)


def parse_category_response(raw_response: str) -> str | None:
    parsed = parse_json_response(raw_response)
    if isinstance(parsed, dict):
        candidate = parsed.get("category")
    elif isinstance(parsed, str):
        candidate = parsed
    else:
        return None
    if not isinstance(candidate, str):
        return None
    normalized = candidate.strip().casefold()
    return next((category for category in ALLOWED_CATEGORIES if category.casefold() == normalized), None)


def fallback_category(transaction: Transaction) -> str:
    description = re.sub(r"\s+", " ", transaction.description.lower())
    if "transfer" in description or "savings" in description or "landlord" in description:
        return "Transfer"
    if transaction.amount > 0:
        return "Income"
    if "home" in description or "depot" in description:
        return "Home"
    if any(term in description for term in ("netflix", "spotify", "apple")):
        return "Entertainment"
    if any(term in description for term in ("gas", "uber", "lyft")):
        return "Transportation"
    if any(term in description for term in ("grocery", "coffee", "pizza", "safeway", "starbucks", "doordash")):
        return "Food"
    if any(term in description for term in ("pharmacy", "cvs", "urgent")):
        return "Healthcare"
    if any(term in description for term in ("amazon", "etsy", "target", "costco")):
        return "Shopping"
    if any(term in description for term in ("comcast", "utility", "electric")):
        return "Utilities"
    if any(term in description for term in ("insurance", "geico")):
        return "Insurance"
    return "Other"


def compute_totals(transactions: list[Transaction]) -> dict[str, float | dict[str, float]]:
    income = sum(tx.amount for tx in transactions if tx.amount > 0)
    expenses = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
    net = income - expenses
    savings_rate = (net / income) if income else 0.0
    by_category: dict[str, float] = {}
    for tx in transactions:
        if tx.amount >= 0:
            continue
        category = tx.category or "Other"
        by_category[category] = by_category.get(category, 0.0) + abs(tx.amount)
    return {
        "income": round(income, 2),
        "expenses": round(expenses, 2),
        "net": round(net, 2),
        "savings_rate": round(savings_rate, 2),
        "by_category": {k: round(v, 2) for k, v in sorted(by_category.items())},
    }


def build_summary(totals: dict[str, float | dict[str, float]]) -> str:
    income = float(totals["income"])  # type: ignore[arg-type]
    expenses = float(totals["expenses"])  # type: ignore[arg-type]
    net = float(totals["net"])  # type: ignore[arg-type]
    top_category = max((items for items in (totals.get("by_category") or {}).items()), default=("Other", 0.0), key=lambda item: item[1])
    return (
        f"This month generated ${income:,.2f} in income and spent ${expenses:,.2f}, "
        f"for a net of ${net:,.2f}. The largest expense category was {top_category[0]} at ${top_category[1]:,.2f}."
    )


def categorize_uncategorized(transactions: list[Transaction], llm: LLMClient) -> list[Transaction]:
    missing = [tx for tx in transactions if tx.category in (None, "")]
    for index, tx in enumerate(missing, start=1):
        prompt = (
            "Categorize this one transaction. Return only a JSON object with a 'category' field. "
            f"The category must be exactly one of: {', '.join(ALLOWED_CATEGORIES)}. "
            "For a refund, transfer, income, or pending transaction, choose the closest applicable category.\n"
            f"Date: {tx.date.isoformat()}\nDescription: {tx.description}\nAmount: {tx.amount}"
        )
        logger.info("Categorizing transaction %d/%d: %s", index, len(missing), tx.description)
        raw_response = llm.complete(prompt)
        category = parse_category_response(raw_response)
        if category is None:
            logger.warning(
                "Invalid category response for %r (%r); using deterministic fallback",
                tx.description,
                raw_response[:200],
            )
            category = fallback_category(tx)
        tx.category = category
    return transactions


def build_report(transactions: list[Transaction]) -> dict[str, Any]:
    totals = compute_totals(transactions)
    report = {
        "totals": {"income": totals["income"], "expenses": totals["expenses"], "net": totals["net"], "savings_rate": totals["savings_rate"]},
        "by_category": totals["by_category"],
        "flagged": [
            {"description": tx.description, "reason": "; ".join(tx.flagged)}
            for tx in transactions
            if tx.flagged
        ],
        "transactions": [
            {
                "date": tx.date.isoformat(),
                "description": tx.description,
                "amount": round(float(tx.amount), 2),
                "category": tx.category or "Other",
            }
            for tx in transactions
        ],
        "summary": build_summary(totals),
    }
    return report


def run_agent(data_dir: str | Path, output_path: str | Path, llm: LLMClient, step_limit: int = 6) -> dict[str, Any]:
    transactions = dedupe_transactions(load_transactions(data_dir))
    logger.info("Loaded %d transactions from %s", len(transactions), data_dir)
    state: dict[str, Any] = {
        "transactions": transactions,
        "step": 0,
        "last_tool": None,
        "errors": [],
        "observation": "No tools have run yet.",
    }
    while state["step"] < step_limit:
        state["step"] += 1
        logger.info("Agent step %d/%d: asking the model to select a tool", state["step"], step_limit)
        prompt = (
            f"Select one of these tools for the finance dataset: {', '.join(sorted(VALID_TOOLS))}. "
            "Return JSON with a 'tool' field. Use the previous tool observation to decide what to do next.\n"
            f"Previous tool observation: {state['observation']}"
        )
        model_pick = parse_tool_choice(llm.complete(prompt))
        tool_name = model_pick.get("tool", "invalid")
        if tool_name not in VALID_TOOLS:
            requested_tool = tool_name
            tool_name = "categorize_uncategorized" if any(tx.category in (None, "") for tx in state["transactions"]) else "compute_totals"
            logger.warning("Model selected invalid tool %r; falling back to %s", requested_tool, tool_name)
            state["errors"].append(f"Invalid tool selection: {requested_tool}")
        missing_count = sum(tx.category in (None, "") for tx in state["transactions"])
        if missing_count and tool_name != "categorize_uncategorized":
            logger.info("Deferring %s until %d uncategorized transaction(s) are handled", tool_name, missing_count)
            tool_name = "categorize_uncategorized"
        state["last_tool"] = tool_name
        logger.info("Running tool: %s", tool_name)
        if tool_name == "categorize_uncategorized":
            missing_count = sum(tx.category in (None, "") for tx in state["transactions"])
            logger.info("Categorizing %d transactions without a category", missing_count)
            state["transactions"] = categorize_uncategorized(state["transactions"], llm)
            remaining_count = sum(tx.category in (None, "") for tx in state["transactions"])
            categorized_count = missing_count - remaining_count
            state["observation"] = (
                f"Categorization tool completed: {categorized_count} transaction(s) now have categories; "
                f"{remaining_count} remain uncategorized."
            )
            logger.info("%s", state["observation"])
            continue
        if tool_name == "compute_totals":
            state["totals"] = compute_totals(state["transactions"])
            state["observation"] = f"Computed totals: {json.dumps(state['totals'], sort_keys=True)}"
            logger.info("Computed transaction totals")
            continue
        if tool_name == "write_report":
            report = build_report(state["transactions"])
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Wrote report to %s", out_path)
            return report
    logger.warning("Reached the %d-step limit; writing a report with the current results", step_limit)
    report = build_report(state["transactions"])
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote report to %s", out_path)
    return report


def load_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
