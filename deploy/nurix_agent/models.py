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

class AskAboutVizRequest(BaseModel):
    chart_html: str = Field(min_length=1, max_length=100000)
    sql: str = Field(min_length=1, max_length=10000)
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
