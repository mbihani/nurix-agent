from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class RefineRequest(BaseModel):
    chart_html: str
    instruction: str
    session_id: str = "default"

class AskAboutVizRequest(BaseModel):
    chart_html: str
    sql: str
    question: str
    session_id: str = "default"
