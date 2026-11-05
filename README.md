# Personal Finance Agent

A command-line personal-finance agent that loads four CSV exports, asks an LLM
to categorize transactions with missing labels, and computes all financial
arithmetic in Python. It writes a JSON report and can replay saved model
responses offline.

## Setup and run

Requires Python 3.12+ and, for live Ollama use, a running Ollama installation.
Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The default provider is a scripted fake, useful for a smoke test:

```powershell
python -m finance_agent --data sample_data --out report.json
```

For the real model run, install Ollama, download the model, then run:

```powershell
ollama pull llama3.1:8b
python -m finance_agent --data sample_data --out report.json --provider ollama --model llama3.1:8b --recording llm_recording.json
```

Ollama must be running locally. The command returns `0` on success and a
non-zero exit code on an unrecoverable error. The `--data` directory must
contain `income.csv`, `expenses.csv`, `bank_statement.csv`, and
`transactions_uncategorized.csv`.

### Offline replay

The live run records each model prompt and raw response in
`llm_recording.json`. Replay the recording without contacting a model:

```powershell
python -m finance_agent --data sample_data --out replay_report.json --replay --recording llm_recording.json
```


## Data handling and cleaning rules

- All four CSV files are required. A missing file fails the run. Rows that
  cannot be parsed are currently skipped.
- Supported dates include ISO dates (`2024-01-02`), single-digit ISO
  components (`2024-1-6`), and US slash dates (`01/03/2024`). Report dates are
  normalized to `YYYY-MM-DD`.
- Descriptions are trimmed and repeated whitespace is collapsed.
- Exact duplicate rows are merged using date + case-insensitive description +
  amount. A known category is retained when the duplicate has one. This is
  deliberately conservative: it does not reconcile different descriptions or
  dates, and identical legitimate same-day purchases could be merged.
- Refunds remain in the data and are flagged. Zero-amount pending rows remain
  in the report and are flagged. Large positive amounts, unknown merchants,
  and pending authorizations are also flagged.
- Positive amounts count as income; negative amounts count as expenses.
  `net = income - expenses`; savings rate is `net / income` (a decimal ratio,
  or zero when income is zero). Per-category totals sum the absolute values of
  negative transactions only. Thus, for example, refunds and interest are
  included in positive income under this simple cash-flow rule.
- The project uses fixed categories. Missing categories are requested from the
  LLM one transaction at a time. Responses must resolve to an allowed
  category; malformed or unsupported answers use a deterministic keyword
  fallback.

## LLM boundary and agent loop

`LLMClient.complete(prompt) -> text` is the provider interface. The live
provider used for the real run is Ollama with `llama3.1:8b`, temperature zero.
Ollama is asked for JSON; the response parser also accepts JSON inside
Markdown fences or surrounding prose.

The model selects among `categorize_uncategorized`, `compute_totals`, and
`write_report`. The loop dispatches those tools, feeds a concise observation
from each tool into the next selection prompt, and has a step limit. Invalid
tool selections fall back to a safe action; if the step limit is reached, the
agent still writes a report. Categorization is completed before totals or
report writing when missing categories remain.

The LLM is not trusted with arithmetic. Python parses and normalizes data,
deduplicates exact matches, computes `income`, `expenses`, `net`,
`savings_rate`, and `by_category`, and assembles the report. Model output is
used only for tool selection and transaction categorization; it is never
copied into report totals.

## Output

The output JSON contains `totals`, `by_category`, `flagged`, `transactions`,
and `summary`. Each transaction has an ISO date, description, numeric amount,
and category. The summary is generated from code using the computed totals;
its figures are not supplied by the LLM.

## Tests

Tests use scripted/fake clients and mocked Ollama HTTP responses. They run
offline without a model:

```powershell
python -m pytest -q
```

They cover date parsing and four-file ingestion, category parsing and
fallbacks, per-transaction categorization without a batch cap, deterministic
totals and report shape, bad tool responses and failure recovery,
recording/replay, the Ollama request settings, and CLI exit behavior.

## AI usage and design decisions

Copilot was used to generate and iterate on the initial project structure,
CSV handling, client interface, agent loop, and tests. I reviewed and revised
the implementation to keep arithmetic deterministic, use an explicit
tool-dispatch loop, validate categories against a fixed list, recover from
invalid model output, and support recorded offline replay. These choices,
including the exact-match deduplication and cash-flow rules above, are
deliberate scope decisions; the data does not provide transaction IDs that
would make fuzzy cross-file matching safe.

## Scope and next steps

The optional duplicate-charge detection and categorization-accuracy
measurement were not implemented. The current exact-match deduplication is
not duplicate-charge detection. There is no UI, database, or fuzzy
cross-source transaction reconciliation. With more time, I would first add
source-aware reconciliation with ambiguous matches preserved for review,
then measure categorization quality against rows with existing labels.

## Walkthrough notes

- Run the agent in replay mode and summarize the report.
- Trace a messy transaction from CSV parsing through categorization into
  `report.json`.
- Show the totals function and where the model is kept out of its arithmetic.
- Demonstrate malformed model-response fallback and the step-limit report path.
- Explain the exact-match deduplication limitation and how recording/replay
  supports an offline run.

## Time spent

Approximately **3-4 hours** in total
