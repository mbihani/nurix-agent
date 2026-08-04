import asyncio
import json
import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import AppConfig
from .graph import agent_graph
from .models import AskAboutVizRequest, ChatRequest, RefineRequest
from .state import AgentState

logger = logging.getLogger(__name__)

try:
    mlflow.langchain.autolog()
except Exception:
    pass

cfg = AppConfig()

try:
    mlflow.set_experiment(cfg.mlflow_experiment)
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("nurix-agent starting")
    yield
    logger.info("nurix-agent stopping")


app = FastAPI(title="nurix-agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_sse_stream(initial_state: AgentState):
    """Returns an async generator that runs the graph and yields SSE events."""
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: dict):
        queue.put_nowait(event)

    initial_state["emit"] = emit

    async def generator():
        async def run_graph():
            try:
                await agent_graph.ainvoke(
                    initial_state,
                    config={"configurable": {"app_config": cfg}},
                )
            except Exception as e:
                queue.put_nowait({"type": "error", "message": str(e)})
            finally:
                queue.put_nowait({"type": "done"})

        asyncio.create_task(run_graph())

        while True:
            event = await queue.get()
            yield {"data": json.dumps(event)}
            if event["type"] in ("done", "error", "rejected"):
                break

    return generator()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    state: AgentState = {
        "question": req.question,
        "session_id": req.session_id,
        "mode": "chat",
        "existing_html": None,
        "existing_sql": None,
        "refine_instruction": None,
        "is_relevant": False,
        "rejection_reason": None,
        "sub_questions": [],
        "chart_hints": [],
        "genie_results": [],
        "chart_htmls": [],
        "insight_text": None,
        "emit": None,
    }
    return EventSourceResponse(_make_sse_stream(state))


@app.post("/refine")
async def refine(req: RefineRequest):
    state: AgentState = {
        "question": req.instruction,
        "session_id": req.session_id,
        "mode": "refine",
        "existing_html": req.chart_html,
        "existing_sql": None,
        "refine_instruction": req.instruction,
        "is_relevant": False,
        "rejection_reason": None,
        "sub_questions": [req.instruction],
        "chart_hints": ["refine"],
        "genie_results": [],
        "chart_htmls": [],
        "insight_text": None,
        "emit": None,
    }
    return EventSourceResponse(_make_sse_stream(state))


@app.post("/ask_about_viz")
async def ask_about_viz(req: AskAboutVizRequest):
    state: AgentState = {
        "question": req.question,
        "session_id": req.session_id,
        "mode": "ask_about_viz",
        "existing_html": req.chart_html,
        "existing_sql": req.sql,
        "refine_instruction": None,
        "is_relevant": False,
        "rejection_reason": None,
        "sub_questions": [req.question],
        "chart_hints": ["insight"],
        "genie_results": [],
        "chart_htmls": [],
        "insight_text": None,
        "emit": None,
    }
    return EventSourceResponse(_make_sse_stream(state))
