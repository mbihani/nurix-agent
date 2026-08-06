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
    sql: str = Field(min_length=1, max_length=10000)
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
