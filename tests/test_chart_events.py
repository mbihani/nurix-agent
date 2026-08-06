"""
Deterministic, LLM-free and network-free tests for the chart SSE event contract,
the deep-research reconnaissance filter, and dense indexing under partial failure.

Covers the failure modes this change exists to close:
  1. The consumer (nurix-nlviz) reads `chart_index`/`chart_total`; the agent emitted
     only `index`/`total`, so its multi-chart branch never fired and 6 of 7 charts
     were silently discarded.
  2. Deep research charted scouting sub-queries (a 1-row date-range probe), and
     filtering them out must renumber DENSELY — a sparse chart_index leaves an
     undefined slot in the consumer's array.
  3. A chart whose GENERATION fails must not leave a gap in the emitted indices
     while chart_total still counts it, and must not have a stray `sql` event
     promising a chart that never arrives.
  4. A chart must never carry an empty-string `sql` pretending to be a query.

Run:  uv run python tests/test_chart_events.py
Exits non-zero on any failed assertion. Imports the REAL helpers so it exercises
shipping code, not a copy.
"""
import asyncio
import sys

from nurix_agent.nodes.genie import _column_values_are_numeric, _has_numeric_column
from nurix_agent.nodes.genie_agent_node import (
    _partition_chartable,
    _recon_skip_reason,
    genie_agent_node,
)
from nurix_agent.nodes import visualizer as viz

NUM = {"name": "review_count", "type": "number"}
STR = {"name": "feature_area", "type": "string"}
SQL = "SELECT feature_area, COUNT(*) FROM reviews GROUP BY 1"


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


class _Cfg:
    ai_gateway_url = "http://unused"
    claude_model = "unused"
    databricks_host = "http://unused"
    genie_space_id = "s"
    warehouse_id = "w"


def _run_visualizer(sub_questions, genie_results, fail_indices=(), mode="chat",
                    state_extra=None, deep_research=True):
    """
    Drive the REAL visualizer_node with only the LLM/token boundary stubbed.

    `fail_indices` makes those candidates raise during generation, simulating a
    timeout or LLM error. Everything about ordering, index assignment and event
    emission under test is shipping code.

    `deep_research` defaults True because that is the path whose deferred `sql`
    events these tests are mostly about; the plain path is covered explicitly by
    the sql-ownership tests below.
    """
    events = []
    calls = []

    class _FakeResponse:
        def __init__(self, text):
            self.content = text

    class _FakeLLM:
        def __init__(self, *a, **kw):
            pass

        async def ainvoke(self, messages):
            # Recover which candidate this is from the question in the user message.
            user = messages[-1]["content"]
            calls.append(user)
            for i, q in enumerate(sub_questions):
                if f"Question: {q}\n" in user and i in fail_indices:
                    raise RuntimeError(f"simulated generation failure for candidate {i}")
            return _FakeResponse(
                "<!DOCTYPE html><html><body>"
                "<script>const d=window.CHART_DATA;</script></body></html>"
            )

    real_llm, real_token = viz.ChatOpenAI, viz.get_databricks_token
    viz.ChatOpenAI = _FakeLLM
    viz.get_databricks_token = lambda cfg: "tok"
    try:
        state = {
            "emit": events.append,
            "mode": mode,
            "question": "q",
            "deep_research": deep_research,
            "sub_questions": list(sub_questions),
            "chart_hints": ["auto"] * len(sub_questions),
            "genie_results": list(genie_results),
            "existing_html": None,
            "existing_sql": None,
            "refine_instruction": None,
        }
        state.update(state_extra or {})
        out = asyncio.run(viz.visualizer_node(state, {"configurable": {"app_config": _Cfg()}}))
    finally:
        viz.ChatOpenAI, viz.get_databricks_token = real_llm, real_token
    return events, out


def _charts(events):
    return [e for e in events if e["type"] == "chart"]


def _sqls(events):
    return [e for e in events if e["type"] == "sql"]


def _thinking(events):
    return [e["text"] for e in events if e["type"] == "thinking"]


def _results_for(sub_questions, sql=SQL, rows=None, columns=None):
    return [
        {
            "text": "", "sql": sql,
            "columns": columns if columns is not None else [STR, NUM],
            "rows": rows if rows is not None else [["a", 1], ["b", 2]],
        }
        for _ in sub_questions
    ]


# --------------------------------------------------------------------------
# CHANGE 1 — chart_index / chart_total on the wire
# --------------------------------------------------------------------------

def test_chart_event_carries_chart_index_and_chart_total():
    """The keys the nurix-nlviz consumer actually reads must be present and numeric."""
    qs = [f"q{i}" for i in range(7)]
    events, _ = _run_visualizer(qs, _results_for(qs))
    charts = _charts(events)
    assert len(charts) == 7, len(charts)
    assert sorted(c["chart_index"] for c in charts) == list(range(7))
    assert all(c["chart_total"] == 7 for c in charts), charts
    # The consumer's exact multi-chart branch — it must now be True.
    assert all(isinstance(c.get("chart_total"), int) and c["chart_total"] > 1
               for c in charts), "consumer isMulti branch still False"


def test_legacy_index_and_total_keys_are_retained():
    """Compatibility alias: the deployed proxy passes events through unchanged."""
    qs = [f"q{i}" for i in range(3)]
    events, _ = _run_visualizer(qs, _results_for(qs))
    for c in _charts(events):
        assert c["index"] == c["chart_index"], c
        assert c["total"] == c["chart_total"] == 3, c


def test_chart_event_carries_its_sql():
    """A pinned chart needs the query that produced it (ChartEvent declares `sql`)."""
    events, _ = _run_visualizer(["q0"], _results_for(["q0"], sql=SQL))
    assert _charts(events)[0]["sql"] == SQL


def test_chart_total_one_keeps_consumer_on_the_single_chart_branch():
    """A genuine single chart must NOT be forced onto the multi branch."""
    events, _ = _run_visualizer(["q0"], _results_for(["q0"]))
    c = _charts(events)[0]
    assert c["chart_index"] == 0 and c["chart_total"] == 1, c
    assert not (isinstance(c["chart_total"], int) and c["chart_total"] > 1)


def test_every_chart_event_comes_from_the_shared_builder():
    """One construction site: no emit site may hand-roll a chart event."""
    import inspect
    src = inspect.getsource(viz)
    # The only literal '"type": "chart"' in the module belongs to _chart_event.
    assert src.count('"type": "chart"') == 1, (
        "a chart event is being built outside _chart_event"
    )


# --------------------------------------------------------------------------
# BLOCKER 4 — dense indexing under partial generation failure
# --------------------------------------------------------------------------

def test_failed_chart_generation_still_yields_dense_indices():
    """
    THE test that matters: candidate 2 of 5 fails generation.

    Emitted chart indices must be dense 0..3 with chart_total == 4, exactly 4 sql
    events whose indices match their charts, and a thinking event naming the failure.
    """
    qs = [f"question {i}" for i in range(5)]
    results = [
        {"text": "", "sql": f"SELECT {i}", "columns": [STR, NUM], "rows": [["a", i], ["b", i]]}
        for i in range(5)
    ]
    events, out = _run_visualizer(qs, results, fail_indices={2})

    charts, sqls = _charts(events), _sqls(events)
    assert len(charts) == 4, f"expected 4 charts, got {len(charts)}"
    assert [c["chart_index"] for c in charts] == [0, 1, 2, 3], charts
    assert all(c["chart_total"] == 4 for c in charts), charts
    assert [c["index"] for c in charts] == [0, 1, 2, 3], "legacy alias must be dense too"
    assert all(c["total"] == 4 for c in charts), charts

    # Exactly one sql event per rendered chart, index-matched.
    assert len(sqls) == 4, f"expected 4 sql events, got {len(sqls)}"
    assert [s["chart_index"] for s in sqls] == [0, 1, 2, 3], sqls
    for s, c in zip(sqls, charts):
        assert s["chart_index"] == c["chart_index"], (s, c)
        assert s["sql"] == c["sql"], (s, c)

    # The FAILED candidate's SQL must never have been promised.
    assert not any(s["sql"] == "SELECT 2" for s in sqls), "sql emitted for a failed chart"
    assert not any(c["sql"] == "SELECT 2" for c in charts)
    # Survivors keep their original relative order.
    assert [c["sql"] for c in charts] == ["SELECT 0", "SELECT 1", "SELECT 3", "SELECT 4"]

    # The failure is named, not invisible.
    thinking = _thinking(events)
    assert any("question 2" in t and "Could not render" in t for t in thinking), thinking
    assert any("5 charts attempted; 4 rendered successfully" in t for t in thinking), thinking
    assert len(out["chart_htmls"]) == 4


def test_multiple_failures_including_first_and_last_stay_dense():
    """Boundary cases: failures at both ends must not shift the survivors' numbering."""
    qs = [f"question {i}" for i in range(5)]
    events, _ = _run_visualizer(qs, _results_for(qs), fail_indices={0, 4})
    charts = _charts(events)
    assert [c["chart_index"] for c in charts] == [0, 1, 2], charts
    assert all(c["chart_total"] == 3 for c in charts), charts
    assert [s["chart_index"] for s in _sqls(events)] == [0, 1, 2]


def test_all_charts_failing_emits_no_chart_and_explains_every_failure():
    """Total generation failure must be loud, not an empty silent response."""
    qs = [f"question {i}" for i in range(3)]
    events, out = _run_visualizer(qs, _results_for(qs), fail_indices={0, 1, 2})
    assert _charts(events) == [] and _sqls(events) == []
    thinking = _thinking(events)
    for i in range(3):
        assert any(f"question {i}" in t and "Could not render" in t for t in thinking), thinking
    assert any("3 charts attempted; 0 rendered successfully" in t for t in thinking), thinking
    assert out["chart_htmls"] == []


def test_no_failures_means_no_failure_chatter():
    """A clean run must not gain noise from the failure handling."""
    qs = [f"q{i}" for i in range(3)]
    events, _ = _run_visualizer(qs, _results_for(qs))
    thinking = _thinking(events)
    assert not any("Could not render" in t for t in thinking), thinking
    assert not any("rendered successfully" in t for t in thinking), thinking


def test_progress_is_visible_during_generation():
    """
    Charts now batch at the end, so the per-chart progress heartbeat is what keeps
    the UI from looking frozen for the whole generation window.
    """
    qs = [f"q{i}" for i in range(4)]
    events, _ = _run_visualizer(qs, _results_for(qs))
    progress = [t for t in _thinking(events) if "Rendered chart" in t]
    assert len(progress) == 4, progress
    assert "Rendered chart 4 of 4..." in progress, progress
    # Progress must precede the first chart event, or it is not progress.
    first_chart = next(i for i, e in enumerate(events) if e["type"] == "chart")
    first_progress = next(i for i, e in enumerate(events)
                          if e["type"] == "thinking" and "Rendered chart" in e["text"])
    assert first_progress < first_chart, "progress must arrive before the batched charts"


# --------------------------------------------------------------------------
# BLOCKER 2 — chart sql must be real
# --------------------------------------------------------------------------

def test_sqlless_result_is_not_charted():
    """A complete-metadata orphan with sql="" must not become a chart."""
    reason = _recon_skip_reason(_multirow("sql-less orphan", 4, sql=""))
    assert reason is not None and "without the SQL" in reason, reason
    # Whitespace-only is just as absent.
    assert _recon_skip_reason(_multirow("blank sql", 4, sql="   \n ")) is not None


def test_sqlless_result_produces_an_explanatory_thinking_event():
    subs = [_multirow("good one", 5), _multirow("sql-less orphan", 4, sql="")]
    events, out = _run_node(subs)
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert any("sql-less orphan" in t and "without the SQL" in t for t in thinking), thinking
    assert out["sub_questions"] == ["good one"], out["sub_questions"]


def test_chart_event_omits_sql_rather_than_faking_an_empty_one():
    """An empty string masquerades as a query; the key must be absent instead."""
    ev = viz._chart_event("<html></html>", index=0, total=1, sql="")
    assert "sql" not in ev, ev
    assert viz._chart_event("<html></html>", index=0, total=1, sql=None).get("sql") is None
    assert viz._chart_event("<html></html>", index=0, total=1, sql=SQL)["sql"] == SQL


def test_missing_sql_at_emission_is_surfaced_not_papered_over():
    """Defence in depth: if the upstream contract broke, say so."""
    events, _ = _run_visualizer(["q0"], _results_for(["q0"], sql=""))
    assert any("missing the SQL that produced it" in t for t in _thinking(events)), \
        _thinking(events)
    assert "sql" not in _charts(events)[0], _charts(events)[0]


def test_zero_chart_fallback_never_picks_a_sqlless_result():
    """The fallback must not reintroduce the empty-sql chart via the back door."""
    subs = [
        _sq("sql-less big result", [["a", i] for i in range(50)], sql=""),
        _sq("row count probe", [[7452]], columns=(NUM,)),
    ]
    events, out = _run_node(subs)
    # The sql-less one has the most rows but is ineligible; the probe wins.
    assert out["sub_questions"] == ["row count probe"], out["sub_questions"]
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert any("row count probe" in t and "best available result" in t for t in thinking), thinking


# --------------------------------------------------------------------------
# WARNING 1 — refine path carries the aliases and a real sql
# --------------------------------------------------------------------------

def test_refine_chart_event_carries_aliases_and_real_sql():
    events, out = _run_visualizer(
        [], [], mode="refine",
        state_extra={
            "existing_html": "<html><body><script>window.CHART_DATA = {\"columns\":[],"
                             "\"rows\":[]};</script><script>c();</script></body></html>",
            "existing_sql": SQL,
            "refine_instruction": "make it a bar chart",
        },
    )
    charts = _charts(events)
    assert len(charts) == 1, charts
    c = charts[0]
    assert c["chart_index"] == 0 and c["chart_total"] == 1, c
    assert c["index"] == 0 and c["total"] == 1, c
    assert c["sql"] == SQL, c
    assert len(out["chart_htmls"]) == 1


def test_refine_without_sql_omits_the_key():
    """Older clients send no sql; the key is absent rather than an empty string."""
    events, _ = _run_visualizer(
        [], [], mode="refine",
        state_extra={
            "existing_html": "<html><body><script>c();</script></body></html>",
            "existing_sql": None,
            "refine_instruction": "make it a bar chart",
        },
    )
    assert "sql" not in _charts(events)[0], _charts(events)[0]


def test_refine_request_accepts_sql_and_stays_backward_compatible():
    from nurix_agent.models import RefineRequest
    assert RefineRequest(chart_html="<p/>", instruction="x").sql is None
    assert RefineRequest(chart_html="<p/>", instruction="x", sql=SQL).sql == SQL


# --------------------------------------------------------------------------
# CHANGE 2 — the reconnaissance filter
# --------------------------------------------------------------------------

def test_single_row_result_is_dropped():
    """(a) A 1-row result is a scalar probe — chart4-of-7 in the live run."""
    reason = _recon_skip_reason(_sq("date range of the review data",
                                    [["2023-01-01", "2024-06-30", 7452]],
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
    assert "feature_area" in reason, reason


def test_healthy_multirow_numeric_result_is_kept():
    """The conservative half of the contract: real results are never dropped."""
    assert _recon_skip_reason(_multirow("negative sentiment by feature area", 6)) is None


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
# WARNING 8 — a measure typed STRING must not be falsely dropped
# --------------------------------------------------------------------------

def test_string_typed_numeric_column_is_treated_as_chartable():
    """Genie sometimes types a real measure as STRING; values are the tie-breaker."""
    str_count = {"name": "review_count", "type": "string"}
    sq = _sq("negative reviews by area",
             [["billing", "120"], ["search", "84"], ["auth", "31"]],
             columns=(STR, str_count))
    assert _recon_skip_reason(sq) is None, _recon_skip_reason(sq)
    assert _has_numeric_column([STR, str_count], sq["rows"])


def test_formatted_numeric_strings_are_still_measures():
    """Pre-formatted values ("42%", "1,234") remain plottable magnitudes."""
    pct = {"name": "negative_rate_pct", "type": "string"}
    rows = [["billing", "42%"], ["search", "31.5%"], ["auth", "8%"]]
    assert _has_numeric_column([STR, pct], rows)
    assert _column_values_are_numeric([["a", "1,234"], ["b", "5,678"]], 1)
    assert _column_values_are_numeric([["a", "$12.50"], ["b", "$9.99"]], 1)


def test_genuinely_textual_column_is_still_dropped():
    """The conservative half: one stray label means it is text, not a measure."""
    sq = _sq("feature areas", [["billing", "high"], ["search", "low"]],
             columns=(STR, {"name": "urgency", "type": "string"}))
    assert _recon_skip_reason(sq) is not None
    # A single non-numeric value disqualifies the whole column.
    assert not _column_values_are_numeric([["a", "12"], ["b", "N/A"], ["c", "34"]], 1)
    assert not _column_values_are_numeric([["a", "billing"]], 1)


def test_value_based_detection_is_conservative_about_edge_cases():
    # Nulls/blanks do not disqualify, but an all-null column is not a measure.
    assert _column_values_are_numeric([["a", "12"], ["b", None], ["c", "34"]], 1)
    assert not _column_values_are_numeric([["a", None], ["b", None]], 1)
    assert not _column_values_are_numeric([], 1)
    # Booleans are flags, not measures.
    assert not _column_values_are_numeric([["a", True], ["b", False]], 1)
    # A short/ragged row must not raise.
    assert not _column_values_are_numeric([["a"]], 1)
    # Types still win first: no rows needed when metadata already says number.
    assert _has_numeric_column([NUM])
    assert not _has_numeric_column([STR])


def test_numeric_detection_reuses_the_shared_helper():
    """No second, inconsistent numeric detector: normalized AND raw types both work."""
    assert _has_numeric_column([STR, NUM])
    assert not _has_numeric_column([STR, {"name": "d", "type": "string"}])
    assert not _has_numeric_column([])
    assert _has_numeric_column([{"name": "n", "type": "BIGINT"}])
    assert _has_numeric_column([{"name": "n", "type": "DECIMAL(10,2)"}])
    assert not _has_numeric_column([{"name": "s", "type": "STRING"}])
    assert not _has_numeric_column([None, "bare string", {"name": "x"}])


# --------------------------------------------------------------------------
# CHANGE 2 — dense renumbering after filtering (the off-by-one risk)
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
        out = asyncio.run(genie_agent_node(state, {"configurable": {"app_config": _Cfg()}}))
    finally:
        node_mod.run_agent_mode, node_mod.mlflow = real_run, real_mlflow
    return events, out


def test_middle_chart_filtered_out_renumbers_densely():
    """
    Filter chart index 3 of 7, expect surviving indices 0..5 with chart_total 6 —
    no gap, no sparse slot, no undefined entry, end to end through the visualizer.
    """
    subs = [_multirow(f"real query {i}", 4, sql=f"SELECT {i}") for i in range(7)]
    # Replace index 3 with a 1-row scouting probe, exactly like the live run.
    subs[3] = _sq("date range of the review data", [["2023-01-01", "2024-06-30", 7452]],
                  columns=(STR, STR, NUM))

    node_events, out = _run_node(subs)
    assert len(out["sub_questions"]) == 6, out["sub_questions"]
    assert "date range of the review data" not in out["sub_questions"]
    # Survivors keep their original identity, in order (nothing shuffled).
    assert out["sub_questions"] == [f"real query {i}" for i in (0, 1, 2, 4, 5, 6)]
    # The node no longer emits sql events; the visualizer does, post-generation.
    assert [e for e in node_events if e["type"] == "sql"] == []

    # Feed the node's real output through the real visualizer.
    events, _ = _run_visualizer(out["sub_questions"], out["genie_results"])
    charts, sqls = _charts(events), _sqls(events)
    assert [c["chart_index"] for c in charts] == [0, 1, 2, 3, 4, 5], charts
    assert all(c["chart_total"] == 6 for c in charts), charts
    assert [s["chart_index"] for s in sqls] == [0, 1, 2, 3, 4, 5], sqls
    # Each chart carries the SQL of the sub-query at its index — the probe's is gone.
    assert [c["sql"] for c in charts] == ["SELECT 0", "SELECT 1", "SELECT 2",
                                          "SELECT 4", "SELECT 5", "SELECT 6"]
    assert all(c.get("sql") for c in charts), "every chart must carry a real sql"


def test_filtering_and_generation_failure_compose_to_stay_dense():
    """Both reducers at once: 7 candidates -> 1 filtered -> 1 generation failure -> 5."""
    subs = [_multirow(f"real query {i}", 4, sql=f"SELECT {i}") for i in range(7)]
    subs[3] = _sq("date range probe", [["2023-01-01", "2024-06-30", 7452]],
                  columns=(STR, STR, NUM))
    _, out = _run_node(subs)
    assert len(out["sub_questions"]) == 6

    # Candidate 2 of the SURVIVORS then fails to render.
    events, _ = _run_visualizer(out["sub_questions"], out["genie_results"], fail_indices={2})
    charts = _charts(events)
    assert [c["chart_index"] for c in charts] == [0, 1, 2, 3, 4], charts
    assert all(c["chart_total"] == 5 for c in charts), charts
    assert [s["chart_index"] for s in _sqls(events)] == [0, 1, 2, 3, 4]
    assert "SELECT 2" not in [c["sql"] for c in charts], "failed candidate leaked"


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
    assert any("single row" in t for t in thinking), thinking
    assert any("numeric" in t for t in thinking), thinking
    assert any("no rows" in t for t in thinking), thinking
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
    assert any("date range probe" in t and "single row" in t for t in thinking), thinking


def test_summary_is_emitted_after_the_zero_chart_fallback():
    """WARNING 5: the derived N-ran/M-charted arithmetic must survive the fallback."""
    subs = [
        _sq("date range probe", [["2023-01-01", "2024-06-30", 7452]], columns=(STR, STR, NUM)),
        _sq("row count probe", [[7452]], columns=(NUM,)),
        _sq("empty probe", []),
    ]
    events, _ = _run_node(subs)
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    # 3 ran, 1 charted (the fallback pick), so 2 skipped — all derived, not hardcoded.
    assert any("3 sub-queries ran" in t and "charting 1" in t and "2 skipped" in t
               for t in thinking), thinking


def test_summary_counts_are_derived_not_hardcoded():
    """Same shape, different arithmetic, to catch a hardcoded string."""
    subs = [_multirow(f"good {i}", 4) for i in range(5)] + [
        _sq("probe", [[1]], columns=(NUM,)),
        _sq("empty", []),
    ]
    events, out = _run_node(subs)
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert len(out["sub_questions"]) == 5
    assert any("7 sub-queries ran" in t and "charting 5" in t and "2 skipped" in t
               for t in thinking), thinking


def test_single_subquery_summary_is_grammatical():
    events, _ = _run_node([_sq("only probe", [[1]], columns=(NUM,))])
    thinking = [e["text"] for e in events if e["type"] == "thinking"]
    assert any("1 sub-query ran" in t for t in thinking), thinking


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
    deep-research node ONLY, so the plain path must still chart it.
    """
    from nurix_agent.nodes import genie as genie_mod
    assert not hasattr(genie_mod, "_recon_skip_reason")
    assert not hasattr(genie_mod, "_partition_chartable")

    # A 1-row/1-column result is exactly what the deep-research filter rejects...
    single = _sq("How many reviews are there in total?", [[7452]], columns=(NUM,))
    assert _recon_skip_reason(single) is not None, "sanity: deep research would drop this"

    # ...yet the plain path charts it unconditionally.
    events, out = _run_visualizer(
        ["How many reviews are there in total?"],
        [{"text": "", "sql": "SELECT COUNT(*) FROM reviews",
          "columns": [NUM], "rows": [[7452]]}],
        deep_research=False,
    )
    charts = _charts(events)
    assert len(charts) == 1, charts
    assert charts[0]["chart_index"] == 0 and charts[0]["chart_total"] == 1, charts[0]
    assert charts[0]["sql"] == "SELECT COUNT(*) FROM reviews"
    assert "window.CHART_DATA" in charts[0]["html"]
    assert len(out["chart_htmls"]) == 1


def test_plain_path_does_not_duplicate_the_sql_event():
    """
    genie_node already emits one sql event per sub-question, so the visualizer must
    NOT emit a second on the plain path — that would change its event shape and make
    sql count != chart count on the wire.
    """
    qs = [f"q{i}" for i in range(3)]
    events, _ = _run_visualizer(qs, _results_for(qs), deep_research=False)
    assert _sqls(events) == [], "plain path must not emit sql events from the visualizer"
    # The chart still carries its own sql, so pairing never depends on the sql event.
    assert all(c.get("sql") == SQL for c in _charts(events)), _charts(events)
    assert len(_charts(events)) == 3


def test_deep_research_path_emits_one_sql_per_rendered_chart():
    """Deep research defers its sql events to here so indices match real charts."""
    qs = [f"q{i}" for i in range(3)]
    events, _ = _run_visualizer(qs, _results_for(qs), deep_research=True)
    assert len(_sqls(events)) == len(_charts(events)) == 3
    assert [s["chart_index"] for s in _sqls(events)] == [0, 1, 2]


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
