from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"

class RefineRequest(BaseModel):
    chart_html: str = Field(min_length=1, max_length=100000)
    instruction: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"

class AskAboutVizRequest(BaseModel):
    chart_html: str = Field(min_length=1, max_length=100000)
    sql: str = Field(min_length=1, max_length=10000)
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
