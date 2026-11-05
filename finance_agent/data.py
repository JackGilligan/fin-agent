from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


ALLOWED_CATEGORIES = [
    "Food",
    "Transportation",
    "Entertainment",
    "Healthcare",
    "Home",
    "Shopping",
    "Utilities",
    "Insurance",
    "Travel",
    "Transfer",
    "Health",
    "Income",
    "Other",
]


@dataclass
class Transaction:
    date: date
    description: str
    amount: float
    category: str | None = None
    source_file: str = ""
    flagged: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, float]:
        return (self.date.isoformat(), self.description.strip().lower(), float(self.amount))


def normalize_date(value: str) -> date:
    text = (value or "").strip()
    if not text:
        raise ValueError("Date is empty")
    text = text.replace(" ", "")
    parts = re.split(r"[-/]", text)
    if len(parts) != 3:
        raise ValueError(f"Unsupported date format: {value!r}")
    if len(parts[0]) == 4:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts[2]) == 4:
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
    return date(year, month, day)


def parse_amount(value: str) -> float:
    text = (value or "").strip().replace(",", "")
    if not text:
        return 0.0
    if text.startswith("$"):
        text = text[1:]
    return float(text)


def normalize_description(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def infer_flag(transaction: Transaction) -> list[str]:
    reasons: list[str] = []
    text = transaction.description.lower()
    if transaction.amount == 0:
        reasons.append("Zero-amount pending row")
    if transaction.amount >= 1000:
        reasons.append("Large transaction amount")
    if "refund" in text:
        reasons.append("Refund transaction")
    if "unknown merchant" in text or "pending auth" in text:
        reasons.append("Unclear merchant")
    return reasons


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_transactions(data_dir: str | Path) -> list[Transaction]:
    data_path = Path(data_dir)
    merged: dict[tuple[str, str, float], Transaction] = {}
    for filename in ["income.csv", "expenses.csv", "bank_statement.csv", "transactions_uncategorized.csv"]:
        csv_path = data_path / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Required input CSV is missing: {csv_path}")
        for row in read_csv_rows(csv_path):
            try:
                tx = build_transaction(row, csv_path.name)
            except (KeyError, ValueError, TypeError):
                continue
            key = tx.key
            if key in merged:
                existing = merged[key]
                if not existing.category and tx.category:
                    existing.category = tx.category
                if not existing.flagged and tx.flagged:
                    existing.flagged = tx.flagged
                continue
            merged[key] = tx
    ordered = sorted(merged.values(), key=lambda item: (item.date, item.description))
    for tx in ordered:
        tx.flagged = infer_flag(tx)
    return ordered


def build_transaction(row: dict[str, str], source_file: str) -> Transaction:
    description = normalize_description(row.get("Description") or row.get("description") or "")
    raw_amount = row.get("Amount") or row.get("amount") or "0"
    amount = parse_amount(raw_amount)
    date_raw = row.get("Date") or row.get("date") or ""
    tx_date = normalize_date(date_raw)
    category = (row.get("Category") or row.get("category") or "").strip()
    if not category and source_file == "transactions_uncategorized.csv":
        category = ""
    if not category and ("Bank" in source_file or source_file == "expenses.csv"):
        category = row.get("Category") or row.get("category") or ""
    tx = Transaction(date=tx_date, description=description, amount=amount, category=category or None, source_file=source_file)
    tx.flagged = infer_flag(tx)
    return tx


def dedupe_transactions(transactions: Iterable[Transaction]) -> list[Transaction]:
    seen: dict[tuple[str, str, float], Transaction] = {}
    for tx in transactions:
        key = tx.key
        if key not in seen:
            seen[key] = tx
        elif tx.category and not seen[key].category:
            seen[key].category = tx.category
    return sorted(seen.values(), key=lambda item: (item.date, item.description))
