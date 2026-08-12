from typing import Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    question: str
    session_id: str
    mode: str  # "chat" | "refine" | "ask_about_viz"
    # Opt-in: route to Genie Agent mode (deep research) instead of the plain space
    # path. False keeps the default behaviour exactly as it was.
    deep_research: bool
    existing_html: str | None
    existing_sql: str | None
    refine_instruction: str | None
    # The Genie conversation that produced the chart being asked about, when the
    # client supplied one. Lets the `ask_about_viz` follow-up CONTINUE that
    # conversation (full prior context) instead of starting a fresh one from the SQL.
    genie_conversation_id: str | None
    # Router output
    is_relevant: bool
    rejection_reason: str | None
    sub_questions: list[str]
    chart_hints: list[str]
    # Genie output
    genie_results: list[dict]  # [{"text": str, "sql": str, "columns": list, "rows": list}]
    # Visualizer output
    chart_htmls: list[str]
    insight_text: str | None
    # SSE emitter (injected at runtime)
    emit: Any
