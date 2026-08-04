from typing import Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    question: str
    session_id: str
    mode: str  # "chat" | "refine" | "ask_about_viz"
    existing_html: str | None
    existing_sql: str | None
    refine_instruction: str | None
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
