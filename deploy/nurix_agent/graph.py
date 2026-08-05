from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.router import router_node
from .nodes.genie import genie_node
from .nodes.genie_agent_node import genie_agent_node
from .nodes.visualizer import visualizer_node

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("genie", genie_node)
    g.add_node("genie_agent", genie_agent_node)
    g.add_node("visualizer", visualizer_node)
    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        lambda s: (
            "end_reject" if not s["is_relevant"]
            else "visualizer" if s["mode"] in ("refine", "ask_about_viz")
            # Opt-in deep research replaces the plain Genie fan-out with a single
            # agent-mode run that does its own decomposition.
            else "genie_agent" if s.get("deep_research")
            else "genie"
        ),
        {"end_reject": END, "visualizer": "visualizer", "genie": "genie", "genie_agent": "genie_agent"}
    )
    g.add_edge("genie", "visualizer")
    g.add_edge("genie_agent", "visualizer")
    g.add_edge("visualizer", END)
    return g.compile()

agent_graph = build_graph()
