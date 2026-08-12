from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
    # Opt in to Genie Agent mode: multi-step research with one chart per sub-query.
    # Slower (~40-70s vs ~15s). Defaults off so existing clients are unaffected.
    deep_research: bool = False

class RefineRequest(BaseModel):
    chart_html: str = Field(min_length=1, max_length=100000)
    instruction: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
    # The query behind the chart being refined. Optional so existing clients keep
    # working; when supplied it is echoed on the refined chart event, so a refined
    # chart stays pinnable/explainable instead of losing its query.
    sql: str | None = Field(default=None, max_length=10000)

class AskAboutVizRequest(BaseModel):
    chart_html: str = Field(min_length=1, max_length=100000)
    # RELAXED from `min_length=1` to optional. This endpoint now sends the question to
    # the Genie space, and the SQL is the chart context that grounds it — so a missing
    # SQL is a real, answerable degradation rather than a malformed request. Rejecting
    # it with a 422 would give the client a validation error for a pin that simply
    # predates SQL storage; instead the answer is produced WITHOUT fresh data and says
    # so in as many words (see the visualizer's ask_about_viz branch). Existing clients
    # that always send SQL are unaffected.
    sql: str | None = Field(default=None, max_length=10000)
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
    # The Genie conversation that originally produced this chart. Pins store it, and
    # passing it lets the follow-up CONTINUE that conversation, which gives Genie the
    # full prior context instead of the SQL-only reconstruction. Optional: when absent
    # (or when the conversation can no longer be resumed) a fresh conversation is
    # started with the SQL as context. See `_run_genie_conversation`.
    conversation_id: str | None = Field(default=None, max_length=200)
