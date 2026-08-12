from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.router import router_node
from .nodes.genie import genie_node
from .nodes.genie_agent_node import genie_agent_node
from .nodes.visualizer import visualizer_node

def _route_after_router(s) -> str:
    """
    Which node runs after the router.

    `ask_about_viz` used to be grouped with `refine` and sent STRAIGHT to the
    visualizer, skipping Genie entirely. That made the answer structurally incapable
    of saying anything new: the insight was written by Claude from the chart HTML and
    the SQL text alone, so it could only restate what was already on screen. A
    follow-up like "how does this compare to the overall average?" cannot be answered
    from a chart that does not contain the overall average. It now goes through Genie
    so the answer is grounded in a fresh query against the underlying data.

    `refine` still goes straight to the visualizer, deliberately: it is a presentation
    instruction ("make it a pie chart", "sort descending"), not a data question, so
    querying Genie would add ~15s of latency for nothing.

    The PLAIN `genie` node is used, never `genie_agent`: deep research takes ~70-90s
    versus ~15s, and this is an interactive follow-up on a chart the user is looking
    at. Deep research is not enabled on this path at all.

    The one case that still bypasses Genie is `ask_about_viz` with NO stored SQL.
    Without the source query there is no chart context to ground a Genie question in —
    "why is this so high?" is unanswerable to a Genie space that cannot see the chart.
    Rather than 422 the request or send Genie a context-free question and dress the
    result up as grounded, that case falls through to the visualizer, which answers
    from the chart alone and DISCLOSES that it had no fresh data.
    """
    if not s["is_relevant"]:
        return "end_reject"
    if s["mode"] == "refine":
        return "visualizer"
    if s["mode"] == "ask_about_viz":
        return "genie" if (s.get("existing_sql") or "").strip() else "visualizer"
    # Opt-in deep research replaces the plain Genie fan-out with a single
    # agent-mode run that does its own decomposition.
    if s.get("deep_research"):
        return "genie_agent"
    return "genie"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("genie", genie_node)
    g.add_node("genie_agent", genie_agent_node)
    g.add_node("visualizer", visualizer_node)
    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"end_reject": END, "visualizer": "visualizer", "genie": "genie", "genie_agent": "genie_agent"}
    )
    g.add_edge("genie", "visualizer")
    g.add_edge("genie_agent", "visualizer")
    g.add_edge("visualizer", END)
    return g.compile()

agent_graph = build_graph()
