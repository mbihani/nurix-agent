import asyncio
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage, MessageStatus
from langchain_core.runnables import RunnableConfig

from ..config import AppConfig
from ..state import AgentState

# Bound the whole Genie conversation (start + internal poll + result fetch) for a
# single sub-question so one stuck space can't hang the SSE stream forever.
GENIE_TIMEOUT_SECONDS = 90

# Databricks SQL numeric column types (base name, before any precision suffix).
_NUMERIC_TYPES = {
    "INT", "INTEGER", "BIGINT", "LONG", "SMALLINT", "TINYINT", "SHORT", "BYTE",
    "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC",
}


def _is_numeric_type(type_text: str | None) -> bool:
    if not type_text:
        return False
    base = type_text.upper().strip().split("(")[0].strip()
    return base in _NUMERIC_TYPES


def _coerce(value, numeric: bool):
    """Cast numeric-column string cells to int/float; leave everything else as-is."""
    if value is None or not numeric or not isinstance(value, str):
        return value
    v = value.replace(",", "").strip()
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return value


def _extract_columns_rows(statement_response) -> tuple[list[dict], list[list]]:
    """
    Extract columns [{"name","type"}] and rows [[val,...]] from a
    StatementResponse (Genie query-result payload).

    Defensive against None / missing manifest / chunked-away result data
    (data_array is None when the result is paged out).
    """
    if statement_response is None:
        return [], []

    manifest = statement_response.manifest
    result = statement_response.result

    columns: list[dict] = []
    numeric_flags: list[bool] = []
    if manifest and manifest.schema and manifest.schema.columns:
        for c in manifest.schema.columns:
            numeric = _is_numeric_type(c.type_text)
            numeric_flags.append(numeric)
            columns.append({
                "name": c.name,
                "type": "number" if numeric else "string",
            })

    rows: list[list] = []
    if result and result.data_array:
        for row in result.data_array:
            cells = list(row)
            if numeric_flags and len(numeric_flags) == len(cells):
                cells = [_coerce(v, numeric_flags[i]) for i, v in enumerate(cells)]
            rows.append(cells)

    return columns, rows


def _process_message(w: WorkspaceClient, space_id: str, msg: GenieMessage) -> dict:
    """
    Turn a terminal GenieMessage into {text, sql, columns, rows}.

    Pulls the SQL/description from the query attachment and the narrative from
    the text attachment, then fetches the tabular result. The message-level
    query-result endpoint was observed to return 0 rows for this space, so we
    fetch the attachment-scoped result (which carries the data) and only fall
    back to the message-level result if the attachment result is empty.
    """
    if msg.status in (MessageStatus.FAILED, MessageStatus.CANCELLED):
        err = ""
        if msg.error is not None:
            err = getattr(msg.error, "error", None) or str(msg.error)
        return {"text": err or "Genie query failed", "sql": "", "columns": [], "rows": [], "failed": True}

    sql = ""
    description = ""
    narrative = ""
    query_attachment_id: str | None = None

    for att in (msg.attachments or []):
        if att.text is not None and att.text.content:
            narrative = att.text.content
        if att.query is not None:
            sql = att.query.query or sql
            description = att.query.description or description
            query_attachment_id = att.attachment_id or query_attachment_id

    text = narrative or description

    columns: list[dict] = []
    rows: list[list] = []
    if query_attachment_id is not None:
        try:
            res = w.genie.get_message_attachment_query_result(
                space_id, msg.conversation_id, msg.message_id, query_attachment_id
            )
            columns, rows = _extract_columns_rows(res.statement_response)
        except Exception:
            columns, rows = [], []

        # Fallback: message-level result (rarely populated for this space, but
        # cheap to try when the attachment result came back empty/chunked).
        if not rows:
            try:
                res2 = w.genie.get_message_query_result(
                    space_id, msg.conversation_id, msg.message_id
                )
                cols2, rows2 = _extract_columns_rows(res2.statement_response)
                if rows2:
                    columns, rows = cols2, rows2
                elif cols2 and not columns:
                    columns = cols2
            except Exception:
                pass

    return {"text": text, "sql": sql, "columns": columns, "rows": rows}


def _run_genie_conversation(space_id: str, host: str, question: str) -> dict:
    """
    Blocking Genie conversation for one sub-question, run in a worker thread.

    Uses a fresh WorkspaceClient per call (per-request auth pattern) so the
    calls run as the deployed app's service principal in prod and as the local
    user's profile locally. start_conversation_and_wait handles polling until
    the message reaches a terminal status.
    """
    w = WorkspaceClient(host=host)
    msg = w.genie.start_conversation_and_wait(
        space_id,
        question,
        # Give the SDK poll loop a little less than the outer asyncio budget so
        # it surfaces a clean SDK timeout before the hard asyncio backstop fires.
        timeout=datetime.timedelta(seconds=GENIE_TIMEOUT_SECONDS - 5),
    )
    return _process_message(w, space_id, msg)


async def _call_genie_for_question(question: str, index: int, cfg: AppConfig, emit) -> dict:
    """Query the space-scoped Genie Conversation API for a single sub-question."""
    emit({"type": "thinking", "text": f"Querying Genie for: {question[:60]}...", "index": index})

    # SDK Genie calls are blocking (they poll); run in a thread so parallel
    # sub-questions actually run concurrently and the SSE loop stays responsive.
    # Catch SDK errors here (e.g. the space's warehouse denying the caller) so
    # the real cause reaches the client as a clean event instead of surfacing
    # as an empty "no data" chart or an abrupt disconnect.
    try:
        result = await asyncio.to_thread(
            _run_genie_conversation, cfg.genie_space_id, cfg.databricks_host, question
        )
    except Exception as e:
        emit({"type": "thinking", "text": f"Genie query failed: {str(e)[:200]}", "index": index})
        return {"text": "", "sql": "", "columns": [], "rows": []}

    if result.get("failed"):
        emit({"type": "thinking", "text": f"Genie query failed: {result.get('text', '')[:200]}", "index": index})
    else:
        if result.get("text"):
            emit({"type": "genie_text", "text": result["text"], "index": index})
        if result.get("sql"):
            emit({"type": "sql", "sql": result["sql"], "index": index})

    return {
        "text": result.get("text", ""),
        "sql": result.get("sql", ""),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
    }


async def genie_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]

    sub_questions = state["sub_questions"]

    async def _run_with_timeout(q: str, i: int) -> dict:
        # Hard backstop around the whole conversation so one hung task can't
        # block the gather (and thus the SSE stream) indefinitely.
        try:
            async with asyncio.timeout(GENIE_TIMEOUT_SECONDS):
                return await _call_genie_for_question(q, i, cfg, emit)
        except asyncio.TimeoutError:
            emit({"type": "thinking", "text": f"Genie timed out for: {q[:60]}", "index": i})
            return {"text": "", "sql": "", "columns": [], "rows": []}

    # Run all sub-questions in parallel.
    tasks = [_run_with_timeout(q, i) for i, q in enumerate(sub_questions)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    genie_results = []
    for r in results:
        if isinstance(r, Exception):
            genie_results.append({"text": "", "sql": "", "columns": [], "rows": []})
        else:
            genie_results.append(r)

    return {"genie_results": genie_results}
