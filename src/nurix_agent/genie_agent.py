"""
Genie Agent mode (deep research) — opt-in parallel path to the plain Genie space path.

The plain path (nodes/genie.py) asks the space one question per sub-question via the
Conversation API. Agent mode instead posts ONE question to the Genie *agents* endpoint,
which does its own multi-step decomposition (reasoning -> execute_sql -> reasoning -> ...)
and streams progress back over SSE.

Endpoint (POST only; the space id doubles as the agent id):

    POST /api/2.0/genie/agents/{space_id}/responses

Every GET route on this surface 404s, so **the SSE stream is the sole delivery
mechanism** — nothing is retrievable after the fact. Everything we need must be
captured while the stream is open.

The databricks-sdk does not expose this endpoint (no `genie/agents` in the SDK tree),
so the request is hand-rolled with httpx. The bearer token still comes from the SDK
auth chain (`WorkspaceClient.config.authenticate()`) so agent mode runs as the local
user's profile locally and as the app's service principal in prod — exactly like the
plain path.

Structured data, and why re-execution is needed
-----------------------------------------------
Genie exposes structured (columns + rows) data for only SOME of the queries it runs:
the final assistant `message` carries one or more content parts, and a part may have a
`metadata` object holding {sql, columns, preview_rows, status, total_row_count}. The
other queries surface only as markdown tables in `function_call_output.output`, which
is capped at 100 rows and is NOT parsed here (markdown is a lossy transport for chart
data).

But `function_call.arguments` carries the SQL for EVERY sub-query. So we recover all of
them and re-execute the ones that arrived without `metadata` through the Statement
Execution API, in parallel, to get real columns + rows. That turns a single-chart
answer into a research narrative plus one chart per sub-query.
"""
import asyncio
import json
import time

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from .nodes.genie import _extract_columns_rows, _coerce, _is_numeric_type

# Agent mode legitimately runs long: measured 40.9s-64.6s on multi-step questions
# (vs ~15s for the plain path). Budget generously; a timeout surfaces as a real
# error via emit rather than degrading to empty data.
AGENT_TIMEOUT_SECONDS = 180

# Bound cost/latency of the recovery step. Genie issued 7 function_calls (5 distinct)
# on the probe question; 6 re-executions is comfortably above the observed need.
MAX_REEXECUTIONS = 6

# Per-statement budget for a single re-execution (these are the same aggregate
# queries Genie already ran, so they are fast — 2.5s observed).
REEXEC_TIMEOUT_SECONDS = 60

# Cap re-executed result size so one accidental wide scan can't balloon the SSE
# payload. Charts never need more; truncation is reported via a thinking event.
REEXEC_ROW_LIMIT = 5000

# The only four SSE `event:` names this surface emits.
_EVENT_CREATED = "response.created"
_EVENT_ITEM_ADDED = "response.output_item.added"
_EVENT_ITEM_DONE = "response.output_item.done"
_EVENT_COMPLETED = "response.completed"


def _normalize_sql(sql: str) -> str:
    """
    Whitespace-normalized SQL, used as the sub-query identity key.

    Genie re-issues byte-identical SQL when it retries a step (the probe run
    produced 7 function_calls covering only 5 distinct queries), and the
    `metadata.sql` string is matched back to a sub-query by this key. Collapsing
    whitespace makes both comparisons robust to the leading newlines/indentation
    Genie embeds in the generated SQL.
    """
    return " ".join((sql or "").split())


def _columns_from_metadata(raw_columns) -> tuple[list[dict], list[bool]]:
    """
    Convert `metadata.columns` (real names + raw Databricks types) into the
    {"name", "type": "number"|"string"} shape the visualizer expects, plus the
    per-column numeric flags used to coerce the all-string cells.
    """
    columns: list[dict] = []
    numeric_flags: list[bool] = []
    for c in raw_columns or []:
        if not isinstance(c, dict):
            continue
        numeric = _is_numeric_type(c.get("type"))
        numeric_flags.append(numeric)
        columns.append({"name": c.get("name", ""), "type": "number" if numeric else "string"})
    return columns, numeric_flags


def _rows_from_metadata(preview_rows, numeric_flags: list[bool]) -> list[list]:
    """Coerce metadata `preview_rows` (all cells are strings) to typed values."""
    rows: list[list] = []
    for row in preview_rows or []:
        cells = list(row)
        if numeric_flags and len(numeric_flags) == len(cells):
            cells = [_coerce(v, numeric_flags[i]) for i, v in enumerate(cells)]
        rows.append(cells)
    return rows


class SSEFrameParser:
    """
    Incremental SSE line parser: `feed(line)` -> list of completed (event, data) frames.

    Kept line-driven and free of any I/O so the same parser runs against httpx's
    async line iterator in production and against a recorded fixture in tests.
    Follows the SSE framing rules that matter here: `event:`/`data:` fields, an
    optional single leading space after the colon, multiple `data:` lines joining
    with newlines, and a blank line terminating the frame.
    """

    def __init__(self):
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, line: str) -> list[tuple[str, dict]]:
        line = line.rstrip("\r\n")
        if not line:
            return self._flush()
        if line.startswith(":"):
            return []  # comment / keep-alive
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return []

    def close(self) -> list[tuple[str, dict]]:
        """Flush a final frame that was not followed by a blank line."""
        return self._flush()

    def _flush(self) -> list[tuple[str, dict]]:
        if self._event is None and not self._data:
            return []
        event, raw = self._event, "\n".join(self._data)
        self._event, self._data = None, []
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, dict):
            return []
        return [(event or "", data)]


class AgentStreamAccumulator:
    """
    Turns the SSE frame sequence into progressive `thinking` events plus a final result.

    Emission happens as frames arrive, never batched at the end — frames were measured
    landing at +8.6s, +13.2s, +20.1s, +27.0s, +40.8s, so progressive feedback is real.

    `response.output_item.added` and `.done` carry BYTE-IDENTICAL items for reasoning,
    function_call and message (verified on a live run), so we emit on whichever arrives
    first and dedupe by item id. That yields the earliest possible feedback without
    double-emitting. `function_call_output` items are the exception — their `.added`
    frame is a truncated `in_progress` stub — but nothing is emitted from those anyway.
    """

    def __init__(self, emit):
        self._emit = emit
        self._seen_ids: set[str] = set()
        self.reasoning_count = 0
        # Distinct sub-queries, keyed by normalized SQL, in first-seen order.
        self._subqueries: dict[str, dict] = {}
        # metadata blocks off the final message, keyed by normalized SQL.
        self._metadata: dict[str, dict] = {}
        self._narrative_parts: list[str] = []
        self._all_parts: list[str] = []
        self.completed = False
        self.conversation_id: str | None = None
        self.response_id: str | None = None

    def handle_frame(self, event: str, data: dict) -> None:
        if event in (_EVENT_CREATED, _EVENT_COMPLETED):
            resp = data.get("response") or {}
            self.conversation_id = resp.get("conversation_id") or self.conversation_id
            self.response_id = resp.get("id") or self.response_id
            if event == _EVENT_COMPLETED:
                self.completed = True
                # Backstop: the terminal frame replays the full output array. If any
                # item never reached us as its own frame, fold it in now so a dropped
                # frame cannot silently cost us a sub-query.
                for item in resp.get("output") or []:
                    self._handle_item(item)
            return
        if event in (_EVENT_ITEM_ADDED, _EVENT_ITEM_DONE):
            item = data.get("item")
            if isinstance(item, dict):
                self._handle_item(item)

    def _handle_item(self, item: dict) -> None:
        item_id = item.get("id")
        item_type = item.get("type")
        # Dedupe on (id, type): the id is stable across .added/.done, and the
        # replay in response.completed repeats every item a third time.
        key = f"{item_type}:{item_id}"
        if item_id and key in self._seen_ids:
            return
        if item_id:
            self._seen_ids.add(key)

        if item_type == "reasoning":
            self._handle_reasoning(item)
        elif item_type == "function_call":
            self._handle_function_call(item)
        elif item_type == "message":
            self._handle_message(item)
        # function_call_output carries only a markdown rendering (100-row cap), which
        # is deliberately not parsed — structured data comes from metadata or a
        # Statement Execution re-run.

    def _handle_reasoning(self, item: dict) -> None:
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = (part.get("text") or "").strip()
            if not text:
                continue
            self.reasoning_count += 1
            self._emit({"type": "thinking", "text": text})

    def _handle_function_call(self, item: dict) -> None:
        if item.get("name") != "execute_sql":
            return
        try:
            args = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            return
        if not isinstance(args, dict):
            return
        sql = args.get("sql") or ""
        if not sql.strip():
            return
        # `title` is Genie's own label for the step — the clearest thing to show the
        # user, and it doubles as the chart heading downstream.
        title = (args.get("title") or "").strip()
        key = _normalize_sql(sql)
        if key in self._subqueries:
            return  # Genie retried an identical query; one sub-query, not two.
        self._subqueries[key] = {"title": title, "sql": sql}
        self._emit({
            "type": "thinking",
            "text": f"Querying: {title}" if title else "Running a SQL query...",
        })

    def _handle_message(self, item: dict) -> None:
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or ""
            if text:
                self._all_parts.append(text)
            md = part.get("metadata")
            if isinstance(md, dict) and md.get("sql"):
                self._metadata[_normalize_sql(md["sql"])] = md
            elif text:
                # Parts WITHOUT metadata hold the prose; the metadata-bearing parts
                # are just markdown renderings of data we chart separately, so
                # excluding them keeps the narrative from duplicating every table.
                self._narrative_parts.append(text)

    def narrative(self) -> str:
        parts = self._narrative_parts or self._all_parts
        return "\n\n".join(p.strip() for p in parts if p.strip())

    def subqueries(self) -> list[dict]:
        """
        Distinct sub-queries in first-seen order, each resolved against the message
        metadata where one is available.

        A metadata block whose `preview_rows` is short of its own `total_row_count`
        is treated as needing re-execution: charting a partial result silently
        undercounts, which is exactly the failure mode the full-payload injection
        was built to avoid.
        """
        out: list[dict] = []
        for key, sq in self._subqueries.items():
            entry = {
                "title": sq["title"],
                "sql": sq["sql"],
                "columns": [],
                "rows": [],
                "source": "pending",
            }
            md = self._metadata.get(key)
            if md is not None:
                columns, numeric_flags = _columns_from_metadata(md.get("columns"))
                rows = _rows_from_metadata(md.get("preview_rows"), numeric_flags)
                total = md.get("total_row_count")
                complete = not (isinstance(total, int) and total > len(rows))
                if rows and complete:
                    entry.update(columns=columns, rows=rows, source="metadata")
                else:
                    # Keep the schema; rows get refilled by re-execution.
                    entry.update(columns=columns, source="pending")
            out.append(entry)
        return out

    def orphan_metadata(self) -> list[dict]:
        """
        metadata blocks whose SQL matched no function_call.

        Not observed on any probed run (every metadata.sql mapped back to a
        function_call), but the featured result must never be lost if it happens.
        """
        return [
            md for key, md in self._metadata.items() if key not in self._subqueries
        ]


def _execute_statement(sql: str, host: str, warehouse_id: str) -> tuple[list[dict], list[list], bool]:
    """
    Blocking Statement Execution run for one recovered sub-query.

    Returns (columns, rows, truncated). Raises on a failed/cancelled statement so
    the caller can surface the real error text. A fresh WorkspaceClient per call
    keeps the per-request auth pattern the plain path already uses.
    """
    w = WorkspaceClient(host=host)
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        row_limit=REEXEC_ROW_LIMIT,
    )

    # wait_timeout caps at 50s; poll if the statement is still running past it.
    deadline = time.monotonic() + REEXEC_TIMEOUT_SECONDS
    while (
        resp.status
        and resp.status.state in (StatementState.PENDING, StatementState.RUNNING)
        and time.monotonic() < deadline
    ):
        time.sleep(1.0)
        resp = w.statement_execution.get_statement(resp.statement_id)

    state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        err = ""
        if resp.status is not None and resp.status.error is not None:
            err = resp.status.error.message or ""
        raise RuntimeError(err or f"statement ended in state {state}")

    columns, rows = _extract_columns_rows(resp)
    truncated = bool(resp.manifest and resp.manifest.truncated)
    return columns, rows, truncated


async def _reexecute(entry: dict, host: str, warehouse_id: str, emit) -> None:
    """
    Fill one sub-query's columns/rows by re-running its SQL. Mutates `entry`.

    Failures are reported with their real error text and leave the entry empty —
    partial results across the other sub-queries are fine; silent empty data is not.
    """
    label = entry.get("title") or entry["sql"][:60]
    try:
        async with asyncio.timeout(REEXEC_TIMEOUT_SECONDS + 15):
            columns, rows, truncated = await asyncio.to_thread(
                _execute_statement, entry["sql"], host, warehouse_id
            )
    except asyncio.TimeoutError:
        entry["source"] = "error"
        entry["error"] = f"timed out after {REEXEC_TIMEOUT_SECONDS}s"
        emit({"type": "thinking", "text": f"Could not fetch data for '{label}': timed out"})
        return
    except Exception as e:
        entry["source"] = "error"
        entry["error"] = str(e)
        emit({"type": "thinking", "text": f"Could not fetch data for '{label}': {str(e)[:200]}"})
        return

    entry["columns"] = columns or entry.get("columns") or []
    entry["rows"] = rows
    entry["source"] = "reexecuted"
    if truncated:
        emit({
            "type": "thinking",
            "text": f"'{label}' returned more than {REEXEC_ROW_LIMIT} rows; charting the first {len(rows)}.",
        })


async def run_agent_mode(question: str, emit, *, host: str, space_id: str, warehouse_id: str) -> dict:
    """
    Run one deep-research question through Genie Agent mode.

    Streams the SSE response, emitting `thinking` events as reasoning steps and SQL
    queries arrive, then recovers structured data for every sub-query Genie ran.

    Returns the plain path's result shape — {text, sql, columns, rows} for the
    featured result so downstream nodes need no changes — plus `sub_queries`, the
    full list of recovered sub-queries each with its own columns/rows.
    """
    token = await asyncio.to_thread(_bearer_token, host)

    url = f"{host.rstrip('/')}/api/2.0/genie/agents/{space_id}/responses"
    body = {
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": question}],
        }]
    }
    headers = {
        "Authorization": token,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    acc = AgentStreamAccumulator(emit)
    parser = SSEFrameParser()

    # The stream is the only delivery mechanism, so a transport error here means the
    # data is simply gone. Surface the real error text; never fall through to empty.
    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=httpx.Timeout(AGENT_TIMEOUT_SECONDS)) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode("utf-8", "replace")[:400]
                        raise RuntimeError(
                            f"Genie agent returned HTTP {resp.status_code}: {detail}"
                        )
                    async for line in resp.aiter_lines():
                        for event, data in parser.feed(line):
                            acc.handle_frame(event, data)
                    for event, data in parser.close():
                        acc.handle_frame(event, data)
    except asyncio.TimeoutError:
        msg = f"Genie agent mode timed out after {AGENT_TIMEOUT_SECONDS}s"
        emit({"type": "thinking", "text": msg})
        return _error_result(msg, acc)
    except Exception as e:
        msg = f"Genie agent mode failed: {e}"
        emit({"type": "thinking", "text": msg[:300]})
        return _error_result(msg, acc)

    subqueries = acc.subqueries()

    # Any metadata block that matched no function_call still carries a real result;
    # keep it as its own sub-query rather than dropping it.
    for md in acc.orphan_metadata():
        columns, numeric_flags = _columns_from_metadata(md.get("columns"))
        rows = _rows_from_metadata(md.get("preview_rows"), numeric_flags)
        if rows:
            subqueries.append({
                "title": "", "sql": md.get("sql", ""),
                "columns": columns, "rows": rows, "source": "metadata",
            })

    # Re-execute the sub-queries Genie did not hand us structured data for, in
    # parallel. Ones already covered by metadata are skipped (never re-run).
    pending = [e for e in subqueries if e["source"] == "pending"][:MAX_REEXECUTIONS]
    skipped = [e for e in subqueries if e["source"] == "pending"][MAX_REEXECUTIONS:]
    for e in skipped:
        e["source"] = "skipped"
        e["error"] = f"exceeded the {MAX_REEXECUTIONS}-query recovery cap"

    if pending:
        emit({
            "type": "thinking",
            "text": f"Fetching full results for {len(pending)} sub-quer"
                    f"{'y' if len(pending) == 1 else 'ies'}...",
        })
        await asyncio.gather(
            *(_reexecute(e, host, warehouse_id, emit) for e in pending),
            return_exceptions=True,
        )

    if not acc.completed:
        emit({
            "type": "thinking",
            "text": "Genie agent stream ended before signalling completion; "
                    "reporting what was received.",
        })

    featured = next((e for e in subqueries if e["rows"]), None)
    return {
        "text": acc.narrative(),
        "sql": (featured or {}).get("sql", ""),
        "columns": (featured or {}).get("columns", []),
        "rows": (featured or {}).get("rows", []),
        "sub_queries": subqueries,
        "reasoning_count": acc.reasoning_count,
        "conversation_id": acc.conversation_id,
        "response_id": acc.response_id,
    }


def _bearer_token(host: str) -> str:
    """
    Bearer token from the SDK auth chain — the same pattern genie.py/config.py use.

    Returns the full header value ("Bearer <token>") so it can be set directly as
    the Authorization header. Never substitute a placeholder here.
    """
    w = WorkspaceClient(host=host)
    auth = w.config.authenticate()
    token = (auth or {}).get("Authorization", "").strip()
    if not token:
        raise RuntimeError("No Databricks token available. Ensure the workspace is authenticated.")
    return token


def _error_result(message: str, acc: AgentStreamAccumulator) -> dict:
    """Result carrying the real error text, plus whatever the stream did deliver."""
    return {
        "text": acc.narrative(),
        "sql": "",
        "columns": [],
        "rows": [],
        "sub_queries": [],
        "reasoning_count": acc.reasoning_count,
        "conversation_id": acc.conversation_id,
        "response_id": acc.response_id,
        "result_error": message,
    }
