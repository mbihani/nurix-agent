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
from ..state import AgentState

# Genie picks its own chart-worthy result sets, so there is no per-sub-question hint
# to carry over from the router; let the visualizer choose from the data shape.
_CHART_HINT = "auto"


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
        emit({"type": "genie_text", "text": result["text"], "index": 0})

    # Only sub-queries that actually carry data can be charted.
    chartable = [
        sq for sq in result.get("sub_queries", [])
        if sq.get("columns") and sq.get("rows")
    ]

    sub_questions: list[str] = []
    genie_results: list[dict] = []
    for i, sq in enumerate(chartable):
        # Genie's own step title is the best chart heading; fall back to the question.
        sub_questions.append(sq.get("title") or question)
        genie_results.append({
            "text": "",
            "sql": sq.get("sql", ""),
            "columns": sq["columns"],
            "rows": sq["rows"],
        })
        emit({"type": "sql", "sql": sq.get("sql", ""), "index": i})

    if not chartable:
        emit({"type": "thinking", "text": "No chartable result sets were recovered."})

    return {
        "sub_questions": sub_questions,
        "chart_hints": [_CHART_HINT] * len(sub_questions),
        "genie_results": genie_results,
    }
