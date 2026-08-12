import asyncio
import json
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import mlflow
from .. import tracing
from ..config import AppConfig, get_databricks_token
from ..state import AgentState

ROUTER_SYSTEM_PROMPT = """
You are a routing agent for an Enterpret customer feedback analytics assistant.
The data is about: product feedback, customer reviews, sentiment, ratings, urgency scores, feature areas, AI categories.

Given a user question:
1. Is it relevant to customer feedback analytics data?
2. If yes, decompose into 1-3 focused sub-questions (each produces one chart).
3. For each sub-question, suggest a chart hint: "bar", "line", "pie", "scatter", "counter", or "auto".

Respond ONLY with valid JSON, no markdown fences:
{
  "is_relevant": true,
  "rejection_reason": null,
  "sub_questions": ["Top 5 feature areas by feedback count"],
  "chart_hints": ["bar"]
}

If not relevant:
{
  "is_relevant": false,
  "rejection_reason": "Not related to feedback analytics data",
  "sub_questions": [],
  "chart_hints": []
}

Examples:
- "show sentiment breakdown and rating trends" -> sub_questions: ["Sentiment distribution by label", "Average rating over time"], chart_hints: ["pie", "line"]
- "what is the weather?" -> is_relevant: false
"""

async def router_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]

    mode = state.get("mode", "chat")

    # Bypass for refine and ask_about_viz
    if mode in ("refine", "ask_about_viz"):
        return {
            "is_relevant": True,
            "rejection_reason": None,
            "sub_questions": [state["question"]],
            "chart_hints": [mode],
        }

    emit({"type": "thinking", "text": "Analysing your question..."})

    token = get_databricks_token(cfg)
    llm = ChatOpenAI(
        base_url=cfg.ai_gateway_url,
        api_key=token,
        model=cfg.claude_model,
    )

    with mlflow.start_span(name="router") as span:
        # The user's question is customer-adjacent free text; bounded, not unbounded.
        span.set_inputs({"question": tracing.truncate(state["question"])})
        try:
            async with asyncio.timeout(30):
                response = await llm.ainvoke([
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": state["question"]},
                ])
        except asyncio.TimeoutError:
            emit({"type": "rejected", "reason": "Router timed out"})
            return {"is_relevant": False, "rejection_reason": "Router timed out", "sub_questions": [], "chart_hints": []}
        content = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        content = content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"is_relevant": False, "rejection_reason": "Could not parse routing decision", "sub_questions": [], "chart_hints": []}
        # `result` is model-authored JSON, so it must not go into the span verbatim:
        # `rejection_reason` and the decomposed sub-questions are unbounded free text.
        # Bounded field-by-field through the one chokepoint rather than dumped whole.
        span.set_outputs({
            "is_relevant": bool(result.get("is_relevant")),
            "rejection_reason": tracing.truncate(result.get("rejection_reason")) or None,
            "sub_questions": [tracing.truncate(q) for q in result.get("sub_questions") or []],
            "chart_hints": [tracing.truncate(h) for h in result.get("chart_hints") or []],
        })

    if not result.get("is_relevant", False):
        emit({"type": "rejected", "reason": result.get("rejection_reason", "Not relevant")})
    elif state.get("deep_research"):
        # Deep research: keep the relevance gate (so an irrelevant question is still
        # rejected before spending 40-70s) but DISCARD the router's decomposition.
        # Agent mode decomposes the question itself; keeping both would decompose
        # twice and fan out one agent run per router sub-question.
        result["sub_questions"] = [state["question"]]
        result["chart_hints"] = []
    else:
        sub_qs = result.get("sub_questions", [])
        if not isinstance(sub_qs, list) or not sub_qs:
            sub_qs = [state["question"]]
        sub_qs = sub_qs[:3]
        hints = result.get("chart_hints", [])
        if not isinstance(hints, list):
            hints = []
        while len(hints) < len(sub_qs):
            hints.append("auto")
        result["sub_questions"] = sub_qs
        result["chart_hints"] = hints[:len(sub_qs)]
        emit({"type": "thinking", "text": f"Planning {len(sub_qs)} visualization(s)..."})

    return {
        "is_relevant": result.get("is_relevant", False),
        "rejection_reason": result.get("rejection_reason"),
        "sub_questions": result.get("sub_questions", [state["question"]]),
        "chart_hints": result.get("chart_hints", ["auto"]),
    }
