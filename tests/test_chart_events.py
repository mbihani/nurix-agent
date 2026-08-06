"""
Deterministic, LLM-free and network-free tests for the chart SSE event contract
and the deep-research reconnaissance filter.

Covers the two failure modes this change exists to close:
  1. The consumer (nurix-nlviz) reads `chart_index`/`chart_total`; the agent emitted
     only `index`/`total`, so its multi-chart branch never fired and 6 of 7 charts
     were silently discarded.
  2. Deep research charted scouting sub-queries (a 1-row date-range probe), and
     filtering them out must renumber DENSELY — a sparse chart_index leaves an
     undefined slot in the consumer's array.

Run:  uv run python tests/test_chart_events.py
Exits non-zero on any failed assertion. Imports the REAL helpers so it exercises
shipping code, not a copy.
"""
import asyncio
import sys

from nurix_agent.nodes.genie import _has_numeric_column
from nurix_agent.nodes.genie_agent_node import (
    _partition_chartable,
    _recon_skip_reason,
    genie_agent_node,
)
from nurix_agent.nodes import visualizer as viz

NUM = {"name": "review_count", "type": "number"}
STR = {"name": "feature_area", "type": "string"}


def _sq(title, rows, columns=(STR, NUM), sql=None, source="reexecuted"):
    return {
        "title": title,
        "sql": sql if sql is not None else f"SELECT * FROM t -- {title}",
        "columns": list(columns),
        "rows": rows,
        "source": source,
    }


def _multirow(title, n=4, **kw):
    return _sq(title, [[f"area{i}", i * 10] for i in range(n)], **kw)


# --------------------------------------------------------------------------
# CHANGE 1 — chart_index / chart_total on the wire
# --------------------------------------------------------------------------

def _capture_chart_event(index, total, sql="SELECT 1", rows=None, columns=None):
    """
    Drive the REAL _generate_chart emit path with the LLM call stubbed out.

    Only the network/LLM boundary is replaced; the event dict itself is built by
    shipping code, so the assertions are about what actually reaches the wire.
    """
    events = []

    class _FakeResponse:
        content = "<!DOCTYPE html><html><body><script>const d=window.CHART_DATA;</script></body></html>"

    class _FakeLLM:
        def __init__(self, *a, **kw):
            pass

        async def ainvoke(self, messages):
            return _FakeResponse()

    real_llm = viz.ChatOpenAI
    viz.ChatOpenAI = _FakeLLM
    try:
        genie_result = {
            "sql": sql,
            "columns": columns if columns is not None else [STR, NUM],
            "rows": rows if rows is not None else [["a", 1], ["b", 2]],
        }

        class _Cfg:
            ai_gateway_url = "http://unused"
            claude_model = "unused"

        asyncio.run(viz._generate_chart(
            "q", "auto", genie_result, _Cfg(), index, total, events.append, "tok",
        ))
    finally:
        viz.ChatOpenAI = real_llm
    charts = [e for e in events if e["type"] == "chart"]
    assert len(charts) == 1, f"expected exactly one chart event, got {len(charts)}"
    return charts[0]


def test_chart_event_carries_chart_index_and_chart_total():
    """The keys the nurix-nlviz consumer actually reads must be present and numeric."""
    ev = _capture_chart_event(5, 7)
    assert ev["chart_index"] == 5, ev
    assert ev["chart_total"] == 7, ev
    # This is the consumer's exact multi-chart branch — it must now be True.
    is_multi = isinstance(ev.get("chart_total"), int) and ev["chart_total"] > 1
    assert is_multi, f"consumer isMulti branch still False: {ev}"


def test_legacy_index_and_total_keys_are_retained():
    """Compatibility alias: the deployed proxy passes events through unchanged."""
    ev = _capture_chart_event(5, 7)
    assert ev["index"] == 5 and ev["total"] == 7, ev
    assert ev["index"] == ev["chart_index"] and ev["total"] == ev["chart_total"], ev


def test_chart_event_carries_its_sql():
    """A pinned chart needs the query that produced it (ChartEvent declares `sql`)."""
    ev = _capture_chart_event(0, 1, sql="SELECT feature_area, COUNT(*) FROM reviews GROUP BY 1")
    assert ev["sql"] == "SELECT feature_area, COUNT(*) FROM reviews GROUP BY 1", ev


def test_chart_total_one_keeps_consumer_on_the_single_chart_branch():
    """A genuine single chart must NOT be forced onto the multi branch."""
    ev = _capture_chart_event(0, 1)
    assert ev["chart_total"] == 1 and ev["chart_index"] == 0, ev
    assert not (isinstance(ev.get("chart_total"), int) and ev["chart_total"] > 1)


# --------------------------------------------------------------------------
# CHANGE 2 — the reconnaissance filter
# --------------------------------------------------------------------------

def test_single_row_result_is_dropped():
    """(a) A 1-row result is a scalar probe — chart4-of-7 in the live run."""
    reason = _recon_skip_reason(_sq("date range of the review data", [["2023-01-01", "2024-06-30", 7452]],
                                    columns=(STR, STR, NUM)))
    assert reason is not None and "single row" in reason, reason


def test_zero_row_result_is_dropped():
    """(b) Nothing to plot."""
    reason = _recon_skip_reason(_sq("empty probe", []))
    assert reason is not None and "no rows" in reason, reason


def test_all_non_numeric_result_is_dropped():
    """(c) No measure column means no value axis."""
    sq = _sq("list of feature areas", [["billing"], ["search"], ["auth"]], columns=(STR,))
    reason = _recon_skip_reason(sq)
    assert reason is not None and "numeric" in reason, reason
    # The reason names the columns so the user can see WHAT was skipped.
    assert "feature_area" in reason, reason


def test_healthy_multirow_numeric_result_is_kept():
    """The conservative half of the contract: real results are never dropped."""
    assert _recon_skip_reason(_multirow("negative sentiment by feature area", 6)) is None


def test_numeric_detection_reuses_the_shared_helper():
    """No second, inconsistent numeric detector: normalized AND raw types both work."""
    assert _has_numeric_column([STR, NUM])
    assert not _has_numeric_column([STR, {"name": "d", "type": "string"}])
    assert not _has_numeric_column([])
    # Raw Databricks type names still resolve through _is_numeric_type.
    assert _has_numeric_column([{"name": "n", "type": "BIGINT"}])
    assert _has_numeric_column([{"name": "n", "type": "DECIMAL(10,2)"}])
    assert not _has_numeric_column([{"name": "s", "type": "STRING"}])
    # Junk must not crash or false-positive.
    assert not _has_numeric_column([None, "bare string", {"name": "x"}])


def test_failed_subqueries_are_dropped_without_a_duplicate_message():
    """run_agent_mode already explained these with real error text."""
    chartable, dropped = _partition_chartable([
        _multirow("good"),
        _sq("fetch failed", [], source="error"),
        _sq("over the cap", [], source="skipped"),
    ])
    assert len(chartable) == 1
    assert dropped == [], f"error/skipped entries must not be re-reported: {dropped}"


# --------------------------------------------------------------------------
# CHANGE 2 — dense renumbering (the off-by-one risk)
# --------------------------------------------------------------------------

class _FakeSpan:
    def set_inputs(self, *a, **kw):
        pass

    def set_outputs(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _run_node(sub_queries, text="Research narrative.", result_error=None):
    """
    Drive the REAL genie_agent_node with run_agent_mode + mlflow stubbed out.

    Returns (events, node_return). Only the network boundary is faked, so the
    filtering, renumbering and thinking-event behaviour under test is shipping code.
    """
    import nurix_agent.nodes.genie_agent_node as node_mod

    events = []
    result = {
        "text": text,
        "sub_queries": sub_queries,
        "reasoning_count": len(sub_queries),
    }
    if result_error:
        result["result_error"] = result_error

    async def fake_run_agent_mode(question, emit, **kw):
        return result

    real_run, real_mlflow = node_mod.run_agent_mode, node_mod.mlflow
    node_mod.run_agent_mode = fake_run_agent_mode

    class _FakeMlflow:
        @staticmethod
        def start_span(name=None):
            return _FakeSpan()

    node_mod.mlflow = _FakeMlflow
    try:
        state = {"emit": events.append, "question": "q", "deep_research": True}

        class _Cfg:
            databricks_host = "http://unused"
            genie_space_id = "s"
            warehouse_id = "w"

        out = asyncio.run(genie_agent_node(state, {"configurable": {"app_config": _Cfg()}}))
    finally:
        node_mod.run_agent_mode, node_mod.mlflow = real_run, real_mlflow
    return events, out


def test_middle_chart_filtered_out_renumbers_densely():
    """
    THE off-by-one test: filter chart index 3 of 7, expect surviving indices 0..5
    with chart_total 6 — no gap, no sparse slot, no undefined entry.
    """
    subs = [_multirow(f"real query {i}", 4) for i in range(7)]
    # Replace index 3 with a 1-row scouting probe, exactly like the live run.
    subs[3] = _sq("date range of the review data", [["2023-01-01", "2024-06-30", 7452]],
                  columns=(STR, STR, NUM))

    events, out = _run_node(subs)

    assert len(out["sub_questions"]) == 6, out["sub_questions"]
    assert "date range of the review data" not in out["sub_questions"]

    sql_events = [e for e in events if e["type"] == "sql"]
    assert [e["chart_index"] for e in sql_events] == [0, 1, 2, 3, 4, 5], sql_events
    assert [e["index"] for e in sql_events] == [0, 1, 2, 3, 4, 5], sql_events

    # The chart events are numbered by the visualizer over the FILTERED list, so
    # assert the total it will be handed is the surviving count.
    total = len(out["sub_questions"])
    assert total == 6
    observed = [_capture_chart_event(i, total)["chart_index"] for i in range(total)]
    assert observed == [0, 1, 2, 3, 4, 5], observed
    assert all(_capture_chart_event(i, total)["chart_total"] == 6 for i in range(total))

    # Survivors keep their original identity, in order (nothing shuffled).
    assert out["sub_questions"] == [f"real query {i}" for i in (0, 1, 2, 4, 5, 6)]
    # Each emitted sql pairs with the sub-query at the same chart_index.
    for ev, sq_title in zip(sql_events, out["sub_questions"]):
        assert sq_title in ev["sql"], (ev, sq_title)


def test_a_thinking_event_is_emitted_for_every_drop():
    """Cardinal sin guard: a filter that drops work invisibly is not acceptable."""
    subs = [
        _multirow("good one", 5),
        _sq("date range probe", [["2023-01-01", "2024-06-30", 7452]], columns=(STR, STR, NUM)),
        _sq("distinct feature areas", [["billing"], ["search"]], columns=(STR,)),
        _sq("empty probe", []),
    ]
    events, out = _run_node(subs)

    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    for label in ("date range probe", "distinct feature areas", "empty probe"):
        assert any(label in t for t in thinking), f"no thinking event named {label!r}: {thinking}"
    # Each drop states a REASON, not just a name.
    assert any("single row" in t for t in thinking), thinking
    assert any("numeric" in t for t in thinking), thinking
    assert any("no rows" in t for t in thinking), thinking
    # And the user can see the N-ran / M-charted arithmetic.
    assert any("4 sub-queries ran" in t and "charting 1" in t for t in thinking), thinking
    assert len(out["sub_questions"]) == 1


def test_zero_charts_falls_back_to_the_best_available_result():
    """Filtering must never produce a silent empty response."""
    subs = [
        _sq("date range probe", [["2023-01-01", "2024-06-30", 7452]], columns=(STR, STR, NUM)),
        _sq("row count probe", [[7452]], columns=(NUM,)),
        _sq("empty probe", []),
    ]
    events, out = _run_node(subs)

    assert len(out["sub_questions"]) == 1, out["sub_questions"]
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert any("Charting the best available result" in t for t in thinking), thinking
    # The fallback picks the result with the most rows, and still explains itself.
    assert any("date range probe" in t and "single row" in t for t in thinking), thinking
    # A chart really is emitted downstream (one sql event, dense index 0).
    sql_events = [e for e in events if e["type"] == "sql"]
    assert [e["chart_index"] for e in sql_events] == [0], sql_events


def test_no_subqueries_at_all_still_explains_itself():
    """Genuinely nothing recovered: still not silent."""
    events, out = _run_node([])
    assert out["sub_questions"] == []
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert any("No chartable result sets were recovered" in t for t in thinking), thinking


def test_no_drops_means_no_skip_chatter():
    """A clean run must not gain noise from this change."""
    events, out = _run_node([_multirow(f"q{i}", 4) for i in range(3)])
    assert len(out["sub_questions"]) == 3
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert not any("Skipped charting" in t for t in thinking), thinking
    assert not any("skipped as low-value" in t for t in thinking), thinking


# --------------------------------------------------------------------------
# The plain path must be COMPLETELY unaffected
# --------------------------------------------------------------------------

def test_plain_path_single_row_still_charts():
    """
    'How many reviews are there?' -> one row, one counter. The filter lives in the
    deep-research node ONLY, so the plain path must still hand it to the visualizer
    and the chart event must still be emitted.
    """
    from nurix_agent.nodes import genie as genie_mod
    # The filter helper must not be reachable from the plain path's node module.
    assert not hasattr(genie_mod, "_recon_skip_reason")
    assert not hasattr(genie_mod, "_partition_chartable")

    # A 1-row/1-column result is exactly what the filter would reject...
    single = _sq("How many reviews are there in total?", [[7452]], columns=(NUM,))
    assert _recon_skip_reason(single) is not None, "sanity: deep research would drop this"

    # ...yet the shared chart emitter charts it unconditionally.
    ev = _capture_chart_event(0, 1, sql="SELECT COUNT(*) FROM reviews",
                              rows=[[7452]], columns=[NUM])
    assert ev["type"] == "chart" and ev["chart_index"] == 0 and ev["chart_total"] == 1, ev
    assert "window.CHART_DATA" in ev["html"]


def test_visualizer_node_does_not_filter():
    """Structural guard: no filtering logic leaked into the shared visualizer."""
    import inspect
    src = inspect.getsource(viz)
    for leaked in ("_recon_skip_reason", "_partition_chartable", "_has_numeric_column"):
        assert leaked not in src, f"{leaked} leaked into the shared visualizer"


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
