"""
LLM-free live smoke test against the fevm-stable Genie space.

Drives the REAL Genie Conversation API path (_run_genie_conversation) with the
sentiment-breakdown question, checks the 72-row totals, then runs the REAL
injection helpers to confirm exactly one CHART_DATA block placed before the
first EXECUTABLE chart <script> (with RCDATA traps present).

Run:  DATABRICKS_CONFIG_PROFILE=fevm-stable uv run python tests/smoke_live.py
"""
import re
import sys

from nurix_agent.config import AppConfig
from nurix_agent.nodes.genie import _run_genie_conversation
from nurix_agent.nodes.visualizer import _inject_chart_data, _split_chart_data

QUESTION = "show me sentiment breakdown by product area"
EXPECTED_TOTALS = {"negative": 2394, "neutral": 2298, "positive": 2760}

_BLOCK_RE = re.compile(r"window\.CHART_DATA\s*=", re.IGNORECASE)


def main():
    cfg = AppConfig()
    print(f"host={cfg.databricks_host}\nspace={cfg.genie_space_id}\nquestion={QUESTION!r}\n")

    result = _run_genie_conversation(cfg.genie_space_id, cfg.databricks_host, QUESTION)
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    if result.get("result_error"):
        print(f"RESULT ERROR: {result['result_error']}")
    print(f"SQL:\n{result.get('sql', '')[:500]}\n")
    print(f"columns ({len(columns)}): {[c.get('name') if isinstance(c, dict) else c for c in columns]}")
    print(f"row_count: {len(rows)}")

    # Genie chooses the grouping non-deterministically: product×feature×sentiment
    # -> 72 rows, or feature×sentiment -> 24 rows, etc. The per-sentiment TOTALS
    # aggregate the whole table and are therefore invariant to the grouping.
    print(f"NOTE: row_count={len(rows)} (72 = product×feature×sentiment; fewer = coarser grouping)")

    # Locate the sentiment-label and count columns by name.
    names = [c.get("name", "").lower() if isinstance(c, dict) else str(c).lower() for c in columns]
    sent_i = next(i for i, n in enumerate(names) if "sentiment" in n)
    count_i = next(i for i, n in enumerate(names) if "count" in n or "review" in n or "num" in n)

    totals = {}
    for r in rows:
        label = str(r[sent_i]).lower()
        totals[label] = totals.get(label, 0) + int(r[count_i])
    print(f"totals: {totals}")
    assert totals == EXPECTED_TOTALS, f"totals mismatch: {totals} != {EXPECTED_TOTALS}"

    # Now exercise the real injection path with RCDATA + comment traps present.
    scaffold = (
        "<script>console.log('early');</script>"  # pre-<head> script
        "<!DOCTYPE html><html><head>"
        "<title>Sentiment <script>evil()</script></title>"
        "<!-- <script>foo()</script> -->"
        "</head><body>"
        "<textarea><script>evil2()</script></textarea>"
        "<canvas></canvas>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
        "<script id='chart'>const d=window.CHART_DATA;</script>"
        "</body></html>"
    )
    _, cleaned = _split_chart_data(scaffold)
    final = _inject_chart_data(cleaned, {"columns": columns, "rows": rows})

    count = len(_BLOCK_RE.findall(final))
    assert count == 1, f"expected exactly one CHART_DATA block, found {count}"

    data_pos = final.index("window.CHART_DATA =")
    first_exec = re.search(r"<script(?:>| )", final, re.IGNORECASE)
    # First executable script tag in the doc must be our data block.
    assert data_pos - len("<script>") == first_exec.start(), (
        "data block is not the first executable script"
    )
    early_pos = final.index("console.log('early')")
    chart_pos = final.index("<script id='chart'>")
    assert data_pos < early_pos < chart_pos, "data must precede all executable scripts"
    print(f"\ninjection: one CHART_DATA block, before first executable script (data@{data_pos})")
    print("SMOKE PASS: 72 rows, totals 2394/2298/2760, single CHART_DATA before first exec script")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"SMOKE FAIL: {e}")
        sys.exit(1)
