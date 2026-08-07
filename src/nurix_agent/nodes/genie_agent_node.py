"""
Graph node for Genie Agent mode (deep research) — the opt-in counterpart to genie_node.

Where genie_node fans the router's sub-questions out across parallel Conversation API
calls, this node makes ONE agent-mode call and lets Genie do its own decomposition.
It then reshapes the recovered sub-queries into exactly the state the existing
visualizer already consumes (`sub_questions` / `chart_hints` / `genie_results`), so
chart generation and the window.CHART_DATA injection path are reused unchanged.
"""
import mlflow
from langchain_core.runnables import RunnableConfig

from ..config import AppConfig
from ..genie_agent import run_agent_mode
from ..narrative import clean_genie_narrative
from ..state import AgentState
from .genie import _has_numeric_column

# Genie picks its own chart-worthy result sets, so there is no per-sub-question hint
# to carry over from the router; let the visualizer choose from the data shape.
_CHART_HINT = "auto"


def _recon_skip_reason(sq: dict) -> str | None:
    """
    Why this deep-research sub-query is not worth charting, or None to chart it.

    Genie's own decomposition includes scouting/reconnaissance steps — "what is the
    date range of the review data" style probes — whose result sets are real but
    carry no chartable shape. A live run produced one 1-row/3-column probe among 7
    charts; dropping it reads better than an uneven grid.

    Deliberately CONSERVATIVE: only results that cannot make a meaningful chart at
    all are dropped, so nothing with a genuine trend or breakdown is ever lost.

    This function is reached ONLY from the deep-research path. The plain path never
    calls it, so a plain single-row answer ("how many reviews are there?" -> one
    counter value) still charts exactly as before.
    """
    rows = sq.get("rows") or []
    columns = sq.get("columns") or []

    # A chart with no query behind it cannot be pinned, refined or explained, and the
    # chart event's `sql` contract would be met only by an empty string pretending to
    # be a query. genie_agent.py admits a complete-metadata result whose `sql` is ""
    # (the `_sqlless_metadata` capture, which deliberately still REPORTS such results
    # upstream); those results are reported there and skipped here rather than charted
    # with a fabricated empty query.
    if not (sq.get("sql") or "").strip():
        return "arrived without the SQL that produced it, so it cannot be charted"

    # (b) Nothing to plot at all.
    if not rows:
        return "returned no rows"
    # (a) A single row is a scalar probe, not a chart. In deep research the narrative
    # already states such a figure, so a one-value chart adds nothing.
    if len(rows) <= 1:
        return "returned a single row (a scalar probe, not a chart)"
    # (c) No measure column means there is no value axis to plot against. Rows are
    # passed so a measure Genie mistyped as STRING is still recognised by its values.
    if not _has_numeric_column(columns, rows):
        col_names = ", ".join(
            str(c.get("name", "")) for c in columns if isinstance(c, dict)
        )
        return f"has no numeric column to plot ({col_names})" if col_names else \
            "has no numeric column to plot"
    return None


def _partition_chartable(sub_queries: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """
    Split recovered sub-queries into (chartable, [(dropped, reason), ...]).

    Runs AFTER results are collected and BEFORE any chart is generated, so the
    chart_index/chart_total the consumer sees are numbered over the SURVIVORS only
    (dense 0..N-1, chart_total == charts actually emitted). Numbering before the
    filter would leave a gap the consumer renders as an undefined slot.

    Sub-queries whose fetch failed (source "error"/"skipped") were already explained
    by run_agent_mode with their real error text, so they are dropped WITHOUT a second
    message — the reason list is only for results we successfully fetched and then
    judged unchartable.
    """
    chartable: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    for sq in sub_queries:
        reason = _recon_skip_reason(sq)
        if reason is None:
            chartable.append(sq)
        elif sq.get("source") in ("error", "skipped"):
            continue  # already reported upstream with the real cause
        else:
            dropped.append((sq, reason))
    return chartable, dropped


def _label(sq: dict) -> str:
    return sq.get("title") or (sq.get("sql") or "")[:60] or "an unnamed sub-query"


async def genie_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]
    question = state["question"]

    emit({"type": "thinking", "text": "Starting deep research (this takes ~40-70s)..."})

    with mlflow.start_span(name="genie_agent") as span:
        span.set_inputs({"question": question, "deep_research": True})
        result = await run_agent_mode(
            question,
            emit,
            host=cfg.databricks_host,
            space_id=cfg.genie_space_id,
            warehouse_id=cfg.warehouse_id,
        )
        span.set_outputs({
            "reasoning_count": result.get("reasoning_count", 0),
            "sub_query_count": len(result.get("sub_queries", [])),
            "narrative_chars": len(result.get("text") or ""),
            "result_error": result.get("result_error"),
        })

    if result.get("result_error"):
        emit({"type": "thinking", "text": f"Deep research error: {result['result_error'][:300]}"})

    # The narrative is the research answer — emit it before the charts so the user
    # reads the conclusion first.
    if result.get("text"):
        emit({
            "type": "genie_text",
            "text": clean_genie_narrative(result["text"]),
            "index": 0,
        })

    sub_queries = result.get("sub_queries", [])
    chartable, dropped = _partition_chartable(sub_queries)

    # Never drop work invisibly: name every skipped sub-query and why, so the user
    # can see that N sub-queries ran while M were charted.
    for sq, reason in dropped:
        emit({
            "type": "thinking",
            "text": f"Skipped charting '{_label(sq)}' — {reason}.",
        })

    # Filtering must never turn a run that HAD data into a silent empty response.
    # Fall back to the best available result (most rows) and say so. Results that
    # arrived WITHOUT their SQL are not eligible: charting one would reintroduce the
    # empty-`sql` chart this filter exists to prevent. They were already reported
    # individually above.
    if not chartable and dropped:
        eligible = [d for d in dropped if (d[0].get("sql") or "").strip()]
        if eligible:
            best, best_reason = max(eligible, key=lambda d: len(d[0].get("rows") or []))
            chartable = [best]
            emit({
                "type": "thinking",
                "text": f"Every sub-query looked like a low-value scouting query, so nothing "
                        f"would have been charted. Charting the best available result "
                        f"('{_label(best)}', which {best_reason}) rather than returning nothing.",
            })

    # The derived N-ran / M-charted / K-skipped summary. Emitted whenever anything was
    # skipped — INCLUDING after the fallback above, where the user would otherwise get
    # individual notices and fallback prose but no arithmetic tying them together.
    # Counts are always derived from the real lists, never hardcoded.
    if dropped:
        charted = len(chartable)
        skipped = len(sub_queries) - charted
        emit({
            "type": "thinking",
            "text": f"{len(sub_queries)} sub-quer"
                    f"{'y' if len(sub_queries) == 1 else 'ies'} ran; charting {charted} "
                    f"({skipped} skipped as low-value scouting queries).",
        })

    sub_questions: list[str] = []
    genie_results: list[dict] = []
    for sq in chartable:
        # Genie's own step title is the best chart heading; fall back to the question.
        sub_questions.append(sq.get("title") or question)
        genie_results.append({
            "text": "",
            "sql": sq.get("sql", ""),
            "columns": sq["columns"],
            "rows": sq["rows"],
        })

    # The `sql` events are emitted by the visualizer AFTER chart generation, so their
    # indices match the charts that actually rendered. Emitting them here would
    # promise a chart for every candidate, including ones whose generation later fails.

    if not chartable:
        emit({"type": "thinking", "text": "No chartable result sets were recovered."})

    return {
        "sub_questions": sub_questions,
        "chart_hints": [_CHART_HINT] * len(sub_questions),
        "genie_results": genie_results,
    }
