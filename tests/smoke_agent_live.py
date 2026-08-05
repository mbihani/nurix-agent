"""
LLM-free live smoke test for Genie Agent mode (deep research).

Drives the REAL agent-mode path (run_agent_mode) against the fevm-stable space,
including the parallel Statement Execution re-runs, and reports: reasoning frames
emitted, sub-queries recovered, how many carry chartable data, and wall-clock.

No LLM is involved — the router and visualizer are bypassed, so this isolates the
agent-mode contract from chart generation.

Run:  DATABRICKS_CONFIG_PROFILE=fevm-stable uv run python tests/smoke_agent_live.py
"""
import asyncio
import sys
import time

from nurix_agent.config import AppConfig
from nurix_agent.genie_agent import run_agent_mode

QUESTION = (
    "What are the top feature areas driving negative sentiment, "
    "and how has that changed over time?"
)


async def main():
    cfg = AppConfig()
    print(f"host={cfg.databricks_host}\nspace={cfg.genie_space_id}\n"
          f"warehouse={cfg.warehouse_id}\nquestion={QUESTION!r}\n")

    t0 = time.time()
    events = []

    def emit(e):
        events.append((round(time.time() - t0, 2), e))
        text = e.get("text") or e.get("sql") or ""
        print(f"  +{time.time() - t0:6.2f}s  {e['type']:<10} {text[:110]}", flush=True)

    result = await run_agent_mode(
        QUESTION, emit,
        host=cfg.databricks_host,
        space_id=cfg.genie_space_id,
        warehouse_id=cfg.warehouse_id,
    )
    elapsed = time.time() - t0

    if result.get("result_error"):
        print(f"\nRESULT ERROR: {result['result_error']}")

    subs = result.get("sub_queries", [])
    chartable = [s for s in subs if s.get("columns") and s.get("rows")]
    by_source = {}
    for s in subs:
        by_source[s["source"]] = by_source.get(s["source"], 0) + 1

    print(f"\n{'=' * 70}")
    print(f"wall clock:            {elapsed:.2f}s")
    print(f"reasoning frames:      {result.get('reasoning_count')}")
    print(f"thinking events:       {sum(1 for _, e in events if e['type'] == 'thinking')}")
    print(f"sub-queries recovered: {len(subs)}")
    print(f"  by source:           {by_source}")
    print(f"chartable sub-queries: {len(chartable)}")
    print(f"narrative arrived:     {bool(result.get('text'))} ({len(result.get('text') or '')} chars)")
    print(f"conversation_id:       {result.get('conversation_id')}")
    print(f"{'=' * 70}\n")

    for i, s in enumerate(subs):
        cols = [c["name"] for c in s.get("columns", [])]
        print(f"[{i}] source={s['source']:<11} rows={len(s.get('rows', [])):<5} "
              f"title={s.get('title')!r}")
        print(f"     columns={cols}")
        if s.get("error"):
            print(f"     ERROR: {s['error']}")
        if s.get("rows"):
            print(f"     first row={s['rows'][0]}")

    print(f"\n--- narrative ---\n{(result.get('text') or '')[:900]}\n")

    assert result.get("reasoning_count", 0) > 0, "no reasoning frames emitted"
    assert subs, "no sub-queries recovered"
    assert chartable, "no sub-query yielded chartable columns+rows"
    assert result.get("text"), "no narrative arrived"
    # Every row of a chartable sub-query must be positionally aligned to its columns.
    for s in chartable:
        for r in s["rows"]:
            assert len(r) == len(s["columns"]), (
                f"row/column arity mismatch in {s.get('title')!r}: "
                f"{len(r)} cells vs {len(s['columns'])} columns"
            )
    print(f"SMOKE PASS: {len(subs)} sub-queries, {len(chartable)} chartable, "
          f"{result.get('reasoning_count')} reasoning frames, {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"SMOKE FAIL: {e}")
        sys.exit(1)
