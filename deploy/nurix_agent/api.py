import asyncio
import json
import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from . import tracing
from .config import AppConfig
from .graph import agent_graph
from .models import AskAboutVizRequest, ChatRequest, RefineRequest
from .state import AgentState

logger = logging.getLogger(__name__)

cfg = AppConfig()

# Sets the tracking URI *and* the experiment, and logs the real error if either
# fails (the previous `except Exception: pass` pair hid a total tracing outage).
# Never raises: tracing is observability, not a hard dependency, so a failure here
# must not stop the app from serving. Inspect health via GET /mlflow_status.
tracing.init(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("nurix-agent starting")
    yield
    logger.info("nurix-agent stopping")


app = FastAPI(title="nurix-agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_graph(queue: asyncio.Queue, initial_state: AgentState):
    """
    Runs graph and always puts a done sentinel on the queue.

    The whole run is wrapped in ONE root span so a single /chat call is ONE trace.
    The span is opened HERE — inside the task that invokes the graph — rather than in
    the request handler, because MLflow's span context is contextvar-based: a span
    opened outside `asyncio.create_task` is not the active parent inside the task, and
    every node span would start its own orphan trace. Verified on MLflow 3.14: without
    this, one /chat produced SIX separate traces (`LangGraph` plus one per hand-rolled
    span); with it, one trace with `LangGraph` and every node span nested underneath.

    `mlflow.langchain.autolog()` contributes its own `LangGraph` span, which nests
    UNDER this root rather than competing with it — so this adds the missing request
    root without duplicating what autolog already provides.
    """
    mode = initial_state.get("mode", "chat")
    try:
        with mlflow.start_span(name=f"{mode}_request") as span:
            # Question text is truncated and row/verbatim data is never recorded:
            # these are customer feedback payloads.
            span.set_inputs({
                "mode": mode,
                "question": tracing.truncate(initial_state.get("question")),
                "deep_research": bool(initial_state.get("deep_research")),
                "session_id": initial_state.get("session_id"),
            })
            result = await agent_graph.ainvoke(
                initial_state,
                config={"configurable": {"app_config": cfg}},
            )
            span.set_outputs({
                "chart_count": len((result or {}).get("chart_htmls") or []),
                "sub_question_count": len((result or {}).get("sub_questions") or []),
                "is_relevant": bool((result or {}).get("is_relevant")),
                "rejection_reason": (result or {}).get("rejection_reason"),
                "insight_chars": len((result or {}).get("insight_text") or ""),
            })
    except Exception as e:
        queue.put_nowait({"type": "error", "message": str(e)})
    finally:
        queue.put_nowait({"type": "done"})


def _make_sse_stream(initial_state: AgentState):
    """Returns an async generator that runs the graph and yields SSE events."""
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: dict):
        queue.put_nowait(event)

    initial_state["emit"] = emit

    async def generator():
        task = asyncio.create_task(run_graph(queue, initial_state))
        try:
            while True:
                event = await queue.get()
                yield {"data": json.dumps(event)}
                if event["type"] in ("done", "error", "rejected"):
                    break
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return generator()


@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""<!DOCTYPE html>
<html>
<head><title>nurix-agent</title>
<style>body{font-family:Arial,sans-serif;background:#1B1B1B;color:#fff;padding:40px;max-width:600px;margin:auto}
h1{color:#FF3621}code{background:#2a2a2a;padding:2px 8px;border-radius:4px;font-size:14px}
.ep{margin:12px 0;padding:12px;background:#2a2a2a;border-radius:6px}
.method{color:#00A972;font-weight:bold;margin-right:8px}</style>
</head>
<body>
<h1>nurix-agent</h1>
<p>Supervisor LangGraph agent — NL-to-viz on Databricks</p>
<h3>Endpoints</h3>
<div class="ep"><span class="method">GET</span><code>/health</code> — liveness check</div>
<div class="ep"><span class="method">POST</span><code>/chat</code> — NL question → SSE: thinking, genie_text, sql, chart, done<br><small>optional <code>deep_research: true</code> → Genie Agent mode: multi-step research, one chart per sub-query (~40-70s)</small></div>
<div class="ep"><span class="method">POST</span><code>/refine</code> — refine existing chart HTML → SSE: chart, done</div>
<div class="ep"><span class="method">POST</span><code>/ask_about_viz</code> — ask about a pinned chart → SSE: insight, done</div>
<p><a href="/docs" style="color:#2272B4">API docs (Swagger)</a></p>
</body></html>""")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/mlflow_status")
async def mlflow_status():
    """
    Tracing health, so a tracing outage is inspectable without container logs.

    `enabled` true with a real `experiment_id` means setup succeeded. When it is
    false, `last_error` carries the actual reason (bad path, missing permission,
    unreachable tracking URI) rather than the silence the old `except: pass` left.

    Note `enabled` reports that SETUP succeeded, not that a given trace was written;
    confirm a specific trace by querying the experiment.
    """
    return tracing.status()


@app.post("/chat")
async def chat(req: ChatRequest):
    state: AgentState = {
        "question": req.question,
        "session_id": req.session_id,
        "mode": "chat",
        "deep_research": req.deep_research,
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
        "deep_research": False,
        "existing_html": req.chart_html,
        "existing_sql": req.sql,
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
        "deep_research": False,
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
