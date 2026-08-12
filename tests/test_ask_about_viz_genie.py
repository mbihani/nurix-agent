"""
Plain-assert checks for the `ask_about_viz` -> GENIE routing change.

The behaviour under test: a follow-up question about a chart is answered from a FRESH
Genie query, not from the chart HTML that is already on the user's screen. The tests
here pin down the parts that are easy to regress silently:

  * routing — ask_about_viz reaches the genie node; refine still does not
  * the composed Genie question carries the chart's SQL and NOT its HTML
  * the relevance gate does not run, so a context-free follow-up is not rejected
  * NO `chart` event is emitted on this path
  * every no-fresh-data case DISCLOSES itself instead of answering confidently

The last one is the point of the whole change. A confidently-worded answer with no data
behind it is the worst outcome on this endpoint, so the disclosure is asserted on the
wire (a `thinking` event) rather than trusted to the model's prose.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nurix_agent.graph import _route_after_router
from nurix_agent.models import AskAboutVizRequest
from nurix_agent.nodes.router import compose_viz_question, router_node
from nurix_agent.nodes.visualizer import _format_genie_result, visualizer_node

CHART_HTML = (
    "<html><head><title>Sentiment by product</title></head><body>"
    "<canvas id=c></canvas><script>window.CHART_DATA={\"rows\":[[\"Widget\",\"positive\",12]]};</script>"
    "</body></html>"
)
CHART_SQL = "SELECT product, sentiment_label, COUNT(*) AS c FROM cat.enterpret.enriched_reviews GROUP BY 1,2"


# --------------------------------------------------------------------------- routing


def _state(**over):
    base = {
        "mode": "ask_about_viz",
        "is_relevant": True,
        "existing_sql": CHART_SQL,
        "deep_research": False,
    }
    base.update(over)
    return base


def test_ask_about_viz_routes_to_genie():
    """The whole point of the change: the question reaches the Genie space."""
    assert _route_after_router(_state()) == "genie"
    print("PASS ask_about_viz routes to the genie node")


def test_ask_about_viz_never_routes_to_deep_research():
    """
    Deep research is ~70-90s versus ~15s. Even if a caller sets the flag, an
    interactive follow-up must use the PLAIN genie node.
    """
    assert _route_after_router(_state(deep_research=True)) == "genie"
    print("PASS ask_about_viz uses the plain genie node even with deep_research set")


def test_refine_still_bypasses_genie():
    """MUST NOT REGRESS: refine is a presentation instruction, not a data question."""
    assert _route_after_router(_state(mode="refine")) == "visualizer"
    assert _route_after_router(_state(mode="refine", deep_research=True)) == "visualizer"
    print("PASS refine still goes straight to the visualizer, skipping Genie")


def test_chat_routing_unchanged():
    assert _route_after_router(_state(mode="chat")) == "genie"
    assert _route_after_router(_state(mode="chat", deep_research=True)) == "genie_agent"
    assert _route_after_router(_state(mode="chat", is_relevant=False)) == "end_reject"
    print("PASS chat routing (plain / deep research / rejected) is unchanged")


def test_ask_about_viz_without_sql_skips_genie():
    """
    No source query means no chart context to ground a Genie question in. That case
    must fall through to the visualizer's DISCLOSED no-data path rather than sending
    Genie a context-free question and presenting the result as grounded.
    """
    for missing in (None, "", "   "):
        assert _route_after_router(_state(existing_sql=missing)) == "visualizer", missing
    print("PASS ask_about_viz with no stored SQL bypasses Genie (disclosed path)")


# ----------------------------------------------------------------- question composition


def test_composed_question_carries_sql_and_question_but_not_html():
    q = compose_viz_question("how does this compare to the overall average rating?", CHART_SQL)
    assert CHART_SQL in q, "the chart's source query is the grounding context"
    assert "how does this compare to the overall average rating?" in q
    # The HTML must never reach Genie: it is a full document, blows the prompt budget,
    # and Genie cannot act on markup.
    assert "<html" not in q and "canvas" not in q and "CHART_DATA" not in q
    assert "<" not in q, f"no markup may leak into the Genie question: {q!r}"
    print("PASS composed Genie question carries SQL + question, never the chart HTML")


def test_composed_question_is_a_single_sub_question():
    """One sub-question only — no fan-out on an interactive follow-up."""
    async def run():
        emitted = []
        return await router_node(
            {
                "mode": "ask_about_viz",
                "question": "why is this so high?",
                "existing_sql": CHART_SQL,
                "emit": emitted.append,
            },
            {"configurable": {"app_config": object()}},
        ), emitted

    out, emitted = asyncio.run(run())
    assert len(out["sub_questions"]) == 1, out["sub_questions"]
    assert CHART_SQL in out["sub_questions"][0]
    assert any(e["type"] == "thinking" for e in emitted), "the Genie step must be named"
    print("PASS router composes exactly ONE Genie sub-question and names the step")


def test_context_free_followup_is_not_rejected():
    """
    ROUTER_SYSTEM_PROMPT would reject these — they contain no analytics keywords
    because the context is the chart, not the sentence. The gate must not run.
    """
    for question in ("why is this so high?", "what's driving that spike?", "is that unusual?"):
        async def run(q=question):
            return await router_node(
                {
                    "mode": "ask_about_viz",
                    "question": q,
                    "existing_sql": CHART_SQL,
                    "emit": lambda e: None,
                },
                {"configurable": {"app_config": object()}},
            )

        out = asyncio.run(run())
        assert out["is_relevant"] is True, question
        assert out["rejection_reason"] is None
    print("PASS context-free follow-ups bypass the relevance gate (not rejected)")


# ------------------------------------------------------------------ result formatting


def test_format_genie_result_includes_narrative_schema_and_rows():
    text = _format_genie_result({
        "text": "The overall average rating is 3.4.",
        "columns": [{"name": "avg_rating", "type": "number"}],
        "rows": [[3.4]],
    })
    assert "The overall average rating is 3.4." in text
    assert "avg_rating" in text
    assert "Total rows returned: 1" in text
    assert "All 1 rows" in text
    print("PASS Genie result renders narrative + schema + rows for the phrasing layer")


def test_format_genie_result_labels_truncation():
    """
    A silently shortened row set invites the model to total a partial column and
    present it as the whole. Truncation must be labelled.
    """
    text = _format_genie_result({
        "text": "",
        "columns": [{"name": "c", "type": "number"}],
        "rows": [[i] for i in range(200)],
    })
    assert "Total rows returned: 200" in text
    assert "TRUNCATED" in text, "truncation must be explicit to the model"
    print("PASS truncated Genie rows are labelled TRUNCATED with the true total")


def test_format_genie_result_states_empty_results_plainly():
    text = _format_genie_result({"text": "", "columns": [], "rows": []})
    assert "no rows" in text
    assert "(none returned)" in text
    print("PASS an empty Genie result is described plainly, not left blank")


# --------------------------------------------------------------- the visualizer branch


class _FakeChunk:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, chunks=("Answer ", "text.")):
        self._chunks = chunks
        self.system_prompts: list[str] = []
        self.user_messages: list[str] = []

    async def astream(self, messages):
        self.system_prompts.append(messages[0]["content"])
        self.user_messages.append(messages[1]["content"])
        for c in self._chunks:
            yield _FakeChunk(c)


def _run_visualizer(monkeypatch_target, state, llm):
    """Run visualizer_node's ask_about_viz branch with a stubbed LLM and token."""
    import nurix_agent.nodes.visualizer as viz

    orig_llm, orig_token = viz.ChatOpenAI, viz.get_databricks_token
    viz.ChatOpenAI = lambda **kw: llm
    viz.get_databricks_token = lambda cfg: "tok"
    try:
        emitted: list[dict] = []
        state["emit"] = emitted.append

        class _Cfg:
            ai_gateway_url = "http://x"
            claude_model = "m"

        out = asyncio.run(viz.visualizer_node(state, {"configurable": {"app_config": _Cfg()}}))
        return out, emitted
    finally:
        viz.ChatOpenAI, viz.get_databricks_token = orig_llm, orig_token


def _viz_state(**over):
    base = {
        "mode": "ask_about_viz",
        "question": "how does this compare to the overall average?",
        "existing_html": CHART_HTML,
        "existing_sql": CHART_SQL,
        "genie_results": [{
            "text": "The overall average rating is 3.4 versus 4.1 for this product.",
            "sql": "SELECT AVG(rating) FROM cat.enterpret.enriched_reviews",
            "columns": [{"name": "avg_rating", "type": "number"}],
            "rows": [[3.4]],
            "error": None,
        }],
    }
    base.update(over)
    return base


def test_grounded_answer_emits_no_chart_event():
    """
    The chart is already on the user's screen. Re-sending one would make the client
    re-render it. This path emits the insight ONLY.
    """
    out, emitted = _run_visualizer(None, _viz_state(), _FakeLLM())
    types = [e["type"] for e in emitted]
    assert "chart" not in types, types
    assert types.count("insight") == 1, types
    assert out["insight_text"] == "Answer text."
    print(f"PASS ask_about_viz emits no chart event (events: {types})")


def test_grounded_answer_is_fed_genie_result_not_chart_html():
    """
    The insight must be grounded in Genie's result. The chart HTML is a competing fact
    source and must not be handed to the model on the grounded path.
    """
    llm = _FakeLLM()
    _run_visualizer(None, _viz_state(), llm)
    user_msg = llm.user_messages[0]
    assert "The overall average rating is 3.4" in user_msg, "Genie narrative must be present"
    assert "avg_rating" in user_msg, "Genie schema must be present"
    assert "CHART_DATA" not in user_msg and "<canvas" not in user_msg, "chart HTML must not be fed in"
    # And the prompt must forbid inventing facts.
    assert "MUST NOT introduce" in llm.system_prompts[0]
    print("PASS grounded insight is fed Genie's result, not the chart HTML")


def test_grounded_insight_event_is_marked_grounded():
    _, emitted = _run_visualizer(None, _viz_state(), _FakeLLM())
    insight = [e for e in emitted if e["type"] == "insight"][0]
    assert insight["grounded"] is True
    assert insight["partial"] is False, "the existing additive field must stay"
    print("PASS terminal insight carries grounded=True and partial=False")


def test_genie_error_is_surfaced_and_disclosed():
    """
    CARDINAL RULE: never silently degrade. The REAL error text must reach the client,
    and the answer must be produced under the mandatory-disclosure prompt.
    """
    state = _viz_state(genie_results=[{
        "text": "", "sql": "", "columns": [], "rows": [],
        "error": "PERMISSION_DENIED: warehouse abc is not accessible",
    }])
    llm = _FakeLLM()
    _, emitted = _run_visualizer(None, state, llm)

    thinking = " ".join(e["text"] for e in emitted if e["type"] == "thinking")
    assert "PERMISSION_DENIED: warehouse abc is not accessible" in thinking, thinking
    assert "MUST open your answer by stating" in llm.system_prompts[0]
    insight = [e for e in emitted if e["type"] == "insight"][0]
    assert insight["grounded"] is False
    print("PASS a Genie error is surfaced verbatim and forces the disclosure prompt")


def test_genie_empty_result_is_disclosed_not_dressed_up():
    """An empty result is a real failure to answer, not a quiet fallback."""
    state = _viz_state(genie_results=[{
        "text": "", "sql": "", "columns": [], "rows": [], "error": None,
    }])
    llm = _FakeLLM()
    _, emitted = _run_visualizer(None, state, llm)
    thinking = " ".join(e["text"] for e in emitted if e["type"] == "thinking")
    assert "no data" in thinking.lower(), thinking
    assert "MUST open your answer by stating" in llm.system_prompts[0]
    print("PASS an empty Genie result is disclosed, not presented as an answer")


def test_missing_sql_is_disclosed_with_its_own_reason():
    """
    The no-stored-SQL case never reached Genie at all, so it must say THAT rather than
    implying a query was attempted and failed.
    """
    state = _viz_state(existing_sql=None, genie_results=[])
    llm = _FakeLLM()
    _, emitted = _run_visualizer(None, state, llm)
    thinking = " ".join(e["text"] for e in emitted if e["type"] == "thinking")
    assert "not saved" in thinking, thinking
    assert "MUST open your answer by stating" in llm.system_prompts[0]
    # Here the chart IS the only source, so it is legitimately supplied.
    assert "CHART_DATA" in llm.user_messages[0]
    print("PASS a missing source query is disclosed with its own distinct reason")


# ------------------------------------------------------------------------- the request


def test_sql_is_optional_but_still_accepted():
    """
    A pin with no stored SQL must not 422. It degrades with an honest message instead.
    """
    r = AskAboutVizRequest(chart_html=CHART_HTML, question="why is this high?")
    assert r.sql is None
    assert r.conversation_id is None
    r2 = AskAboutVizRequest(chart_html=CHART_HTML, question="q", sql=CHART_SQL, conversation_id="conv-1")
    assert r2.sql == CHART_SQL
    assert r2.conversation_id == "conv-1"
    # The deployed nurix-nlviz always sends `sql`, and sends "" for an unsaved pin.
    r3 = AskAboutVizRequest(chart_html=CHART_HTML, question="q", sql="")
    assert r3.sql == ""
    print("PASS sql is optional (no 422) and conversation_id is accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nask_about_viz -> Genie tests passed")
