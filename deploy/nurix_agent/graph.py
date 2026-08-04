from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.router import router_node
from .nodes.genie import genie_node
from .nodes.visualizer import visualizer_node

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("genie", genie_node)
    g.add_node("visualizer", visualizer_node)
    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        lambda s: (
            "end_reject" if not s["is_relevant"]
            else "visualizer" if s["mode"] in ("refine", "ask_about_viz")
            else "genie"
        ),
        {"end_reject": END, "visualizer": "visualizer", "genie": "genie"}
    )
    g.add_edge("genie", "visualizer")
    g.add_edge("visualizer", END)
    return g.compile()

agent_graph = build_graph()
