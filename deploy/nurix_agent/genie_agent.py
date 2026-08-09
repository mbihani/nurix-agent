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
captured while the stream is open. That is why every drop path in this module is
counted and surfaced through `emit` instead of being swallowed: a frame we fail to
understand is data the user can never get back, so it has to be visible.

No incremental TEXT deltas (platform limitation)
------------------------------------------------
Reasoning frames DO arrive progressively (measured landing at +8.6s ... +40.8s), which
is what makes the `thinking` events genuinely incremental. The final NARRATIVE does
not: the `message` item arrives on `response.output_item.added` ALREADY
`status=completed` with the entire text in one frame (live probe 2026-08-10: 1270
chars in a single frame at +20.3s), and `.done` repeats it byte-identically. This
surface emits no `response.output_text.delta`-style event — the complete observed
event set is response.created, response.output_item.added/.updated/.done,
response.completed and response.failed.

Consequence: `genie_text_delta` can only be emitted at Genie's own content-PART
granularity (see `narrative_parts`). Chopping the finished narrative into synthetic
chunks is deliberately NOT done — it would add no latency benefit and would present a
non-streaming platform as streaming. If Genie later adds text deltas, they will flow
through this path unchanged.

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
import hashlib
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

# SSE `event:` names observed on this surface. The first four came from the initial
# probe; `response.failed` and `response.output_item.updated` from a second live probe.
# Anything outside this set is counted as an unknown frame and reported, because on a
# sole-delivery stream an unrecognized name is silent data loss.
_EVENT_CREATED = "response.created"
_EVENT_ITEM_ADDED = "response.output_item.added"
_EVENT_ITEM_UPDATED = "response.output_item.updated"
_EVENT_ITEM_DONE = "response.output_item.done"
_EVENT_COMPLETED = "response.completed"
_EVENT_FAILED = "response.failed"

_ITEM_EVENTS = (_EVENT_ITEM_ADDED, _EVENT_ITEM_UPDATED, _EVENT_ITEM_DONE)
_RESPONSE_EVENTS = (_EVENT_CREATED, _EVENT_COMPLETED, _EVENT_FAILED)

# `metadata.status` values that mean "this result set is real and finished".
# 'available' is what the probed runs carry. Anything else (e.g. 'in_progress',
# 'failed') is NOT treated as proof of a complete result — the sub-query is
# re-executed instead of charting data we cannot vouch for.
_METADATA_OK_STATUSES = frozenset({
    "available", "completed", "complete", "succeeded", "success", "ok", "finished",
})

# Quote characters that open a region where whitespace is significant: single-quoted
# string literals, double-quoted and backtick-quoted identifiers.
_QUOTE_CHARS = frozenset("'\"`")


def _sql_segments(sql: str) -> list[tuple[str, str]]:
    """
    Split SQL into (text, kind) segments, kind in {"code", "literal", "comment"}.

    A single left-to-right scanner, used by both SQL keys below so neither of them
    can mistake a `--` inside a string for a comment, or collapse whitespace inside
    a literal. Escapes are handled so the scanner does not mis-detect where a
    literal ends: doubled quotes ('' "" ``) for all three quote styles, plus
    backslash escapes (\\') inside the quote styles Databricks treats as strings.

    Pure string -> list of strings: unit-testable with no warehouse.
    """
    s = sql or ""
    n = len(s)
    out: list[tuple[str, str]] = []
    code: list[str] = []
    i = 0

    def flush_code() -> None:
        if code:
            out.append(("".join(code), "code"))
            code.clear()

    while i < n:
        ch = s[i]

        if ch in _QUOTE_CHARS:
            flush_code()
            quote = ch
            backslash_escapes = quote != "`"  # backticks only ever double-escape
            lit = [quote]
            j = i + 1
            while j < n:
                c = s[j]
                if backslash_escapes and c == "\\" and j + 1 < n:
                    lit.append(s[j:j + 2])
                    j += 2
                    continue
                if c == quote:
                    if j + 1 < n and s[j + 1] == quote:  # doubled -> escaped quote
                        lit.append(quote * 2)
                        j += 2
                        continue
                    lit.append(quote)  # closing quote
                    j += 1
                    break
                lit.append(c)
                j += 1
            out.append(("".join(lit), "literal"))
            i = j
            continue

        if s.startswith("--", i):
            flush_code()
            end = s.find("\n", i)
            end = n if end == -1 else end
            out.append((s[i:end], "comment"))
            i = end
            continue

        if s.startswith("/*", i):
            flush_code()
            end = s.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append((s[i:end], "comment"))
            i = end
            continue

        code.append(ch)
        i += 1

    flush_code()
    return out


def _collapse_ws(text: str) -> str:
    """Collapse internal whitespace runs to one space, keeping edge-space markers."""
    if not text:
        return ""
    core = " ".join(text.split())
    if not core:
        return " "  # whitespace-only segment still separates its neighbours
    lead = " " if text[0].isspace() else ""
    trail = " " if text[-1].isspace() else ""
    return f"{lead}{core}{trail}"


def _join_pieces(pieces: list[str]) -> str:
    """Concatenate segment pieces without doubling the spaces at their seams."""
    out = ""
    for piece in pieces:
        if not piece:
            continue
        if out.endswith(" ") and piece.startswith(" "):
            piece = piece[1:]  # collapsed pieces carry at most one leading space
            if not piece:
                continue
        out += piece
    return out.strip()


def _normalize_sql(sql: str) -> str:
    """
    Whitespace-normalized SQL, used as the sub-query identity key.

    Genie re-issues byte-identical SQL when it retries a step (the probe run
    produced 7 function_calls covering only 5 distinct queries), and the
    `metadata.sql` string is matched back to a sub-query by this key. Collapsing
    whitespace makes both comparisons robust to the leading newlines/indentation
    Genie embeds in the generated SQL.

    Collapsing is **literal-aware**: whitespace inside single-quoted strings and
    quoted identifiers is preserved verbatim, because it changes results.
    `SELECT 'a b'` and `SELECT 'a  b'` are different queries and must not collapse
    onto one key — otherwise a genuine sub-query is silently dropped as a "retry".
    """
    return _join_pieces([
        text if kind == "literal" else _collapse_ws(text)
        for text, kind in _sql_segments(sql)
    ])


def _loose_sql_key(sql: str) -> str:
    """
    Deliberately lenient SQL key, used only to decide whether a `metadata` block is
    genuinely an orphan (matched no function_call) or just a superficially different
    spelling of one we already have.

    Ignores comments, trailing semicolons and the case of keywords/identifiers. Does
    NOT fold the case of string literals — literal text is data, and folding it would
    merge two queries that return different rows.
    """
    pieces: list[str] = []
    for text, kind in _sql_segments(sql):
        if kind == "literal":
            pieces.append(text)
        elif kind == "comment":
            pieces.append(" ")  # comments cannot change the result set
        else:
            pieces.append(_collapse_ws(text).lower())
    return _join_pieces(pieces).rstrip("; ")


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


def _metadata_is_complete(md: dict, rows: list[list]) -> bool:
    """
    Whether a `metadata` block's `preview_rows` can be charted as the whole answer.

    Charting a partial result silently undercounts, which is exactly the failure mode
    this path exists to avoid, so completeness has to be positively established:

      * `total_row_count` known -> the preview must cover all of it;
      * `total_row_count` absent -> NOT proof of completeness. Unknown-total only
        counts as complete when rows are actually present (an empty preview with no
        total tells us nothing, so it goes to re-execution);
      * `status` present but not a success value -> not complete.
    """
    total = md.get("total_row_count")
    if isinstance(total, bool) or not isinstance(total, int):
        complete = bool(rows)
    else:
        complete = total <= len(rows)

    status = md.get("status")
    if isinstance(status, str) and status.strip().lower() not in _METADATA_OK_STATUSES:
        complete = False
    return complete


def _extract_error_text(data: dict) -> str:
    """
    Real error text out of a `response.failed` frame.

    The exact payload shape of this frame is not pinned down by the probes, so this
    walks the plausible locations and, failing those, serializes whatever it was
    handed. It never returns an empty string: a failure whose text we drop becomes
    the misleading generic "stream ended" message, and the caller is left guessing.
    """
    resp = data.get("response")
    holders = [h for h in (resp if isinstance(resp, dict) else None, data) if isinstance(h, dict)]

    for holder in holders:
        err = holder.get("error")
        if isinstance(err, dict):
            for key in ("message", "error_message", "detail", "description", "reason", "code"):
                val = err.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return json.dumps(err, sort_keys=True, default=str)[:600]
        if isinstance(err, str) and err.strip():
            return err.strip()

    for holder in holders:
        for key in ("error_message", "status_message", "message", "detail"):
            val = holder.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    return "unrecognized failure payload: " + json.dumps(data, sort_keys=True, default=str)[:600]


class SSEFrameParser:
    """
    Incremental SSE line parser: `feed(line)` -> list of completed (event, data) frames.

    Kept line-driven and free of any I/O so the same parser runs against httpx's
    async line iterator in production and against a recorded fixture in tests.
    Follows the SSE framing rules that matter here: `event:`/`data:` fields, an
    optional single leading space after the colon, multiple `data:` lines joining
    with newlines, and a blank line terminating the frame.

    A frame whose payload cannot be used (unparseable JSON, or a non-object) is
    still dropped — one bad frame must never kill the stream — but it is counted and
    reported through `on_discard` so the loss is visible rather than silent.
    """

    def __init__(self, on_discard=None):
        self._event: str | None = None
        self._data: list[str] = []
        self._on_discard = on_discard
        self.discarded_frames = 0

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

    def _discard(self, reason: str) -> None:
        self.discarded_frames += 1
        if self._on_discard is not None:
            self._on_discard(reason)

    def _flush(self) -> list[tuple[str, dict]]:
        if self._event is None and not self._data:
            return []
        event, raw = self._event, "\n".join(self._data)
        self._event, self._data = None, []
        if not raw:
            # An `event:` with no payload carries nothing to lose (keep-alives and
            # bare pings look like this), so it is not counted as a discard.
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._discard("unparseable JSON payload")
            return []
        if not isinstance(data, dict):
            self._discard("payload was not a JSON object")
            return []
        return [(event or "", data)]


class AgentStreamAccumulator:
    """
    Turns the SSE frame sequence into progressive `thinking` events plus a final result.

    Emission happens as frames arrive, never batched at the end — frames were measured
    landing at +8.6s, +13.2s, +20.1s, +27.0s, +40.8s, so progressive feedback is real.

    `response.output_item.added`, `.updated` and `.done` carry BYTE-IDENTICAL items for
    reasoning, function_call and message (verified against the recorded streams), so we
    emit on whichever arrives first and dedupe. Dedupe uses the item id when there is
    one and a content hash when there is not — an id-less item would otherwise be
    emitted on `.added`, again on `.done`, and a third time in the `response.completed`
    replay. Because those three frames are byte-identical for exactly the item types we
    emit from, the content hash is a sound key for them.

    `function_call_output` items are the exception — their `.added` frame is a truncated
    `in_progress` stub, so `.added` and `.done` differ and a content hash would not
    collapse them — but nothing is emitted from those items at all, so no double
    emission is possible.
    """

    def __init__(self, emit):
        self._emit = emit
        self._seen_ids: set[str] = set()
        self.reasoning_count = 0
        # Distinct sub-queries, keyed by normalized SQL, in first-seen order.
        self._subqueries: dict[str, dict] = {}
        # metadata blocks off the final message, keyed by normalized SQL.
        self._metadata: dict[str, dict] = {}
        # metadata blocks that carried rows but no SQL — unmatchable and unrunnable,
        # so they can only be reported (see `orphan_metadata`).
        self._sqlless_metadata: list[dict] = []
        self._narrative_parts: list[str] = []
        self._all_parts: list[str] = []
        self.completed = False
        self.failed = False
        self.error_message: str | None = None
        self.conversation_id: str | None = None
        self.response_id: str | None = None
        # Visibility counters for frames that reached us but could not be used.
        self.discarded_frames = 0
        self.unknown_frames = 0
        self.unknown_event_names: set[str] = set()

    # -- loss accounting ---------------------------------------------------

    def note_discarded(self, reason: str | None = None) -> None:
        """A frame was dropped before it could be interpreted (parser callback)."""
        self.discarded_frames += 1

    def note_unknown(self, event: str | None = None) -> None:
        """A frame arrived that this module does not know how to route."""
        self.unknown_frames += 1
        if event:
            self.unknown_event_names.add(event[:60])

    def frame_loss_message(self) -> str | None:
        """One aggregate sentence about unusable frames, or None if there were none."""
        total = self.discarded_frames + self.unknown_frames
        if not total:
            return None
        detail = []
        if self.discarded_frames:
            detail.append(f"{self.discarded_frames} unparseable")
        if self.unknown_frames:
            names = ", ".join(sorted(self.unknown_event_names)) if self.unknown_event_names else ""
            detail.append(f"{self.unknown_frames} unrecognized" + (f" ({names})" if names else ""))
        return (
            f"{total} stream frame{'s' if total != 1 else ''} could not be used "
            f"({'; '.join(detail)}); this answer may be missing part of what Genie sent."
        )

    # -- frame routing -----------------------------------------------------

    def handle_frame(self, event: str, data: dict) -> None:
        if event in _ITEM_EVENTS:
            item = data.get("item")
            if isinstance(item, dict):
                self._handle_item(item)
            else:
                self.note_unknown(f"{event} without an item")
            return

        if event in _RESPONSE_EVENTS:
            self._handle_response(event, data)
            return

        if not event:
            # A `data:`-only frame has no event name to route on, so route on the
            # payload shape instead. Dropping it would be permanent data loss.
            item = data.get("item")
            if isinstance(item, dict):
                self._handle_item(item)
                return
            if isinstance(data.get("response"), dict):
                self._handle_response(None, data)
                return

        self.note_unknown(event)

    def _handle_response(self, event: str | None, data: dict) -> None:
        resp = data.get("response") or {}
        self.conversation_id = resp.get("conversation_id") or self.conversation_id
        self.response_id = resp.get("id") or self.response_id

        # For an event-less frame the response's own `status` says which frame it is.
        status = resp.get("status") if isinstance(resp.get("status"), str) else ""
        failed = event == _EVENT_FAILED or (event is None and status == "failed")
        completed = event == _EVENT_COMPLETED or (event is None and status == "completed")

        if failed:
            self.failed = True
            self.error_message = _extract_error_text(data)
            self._emit({
                "type": "thinking",
                "text": f"Genie agent reported a failure: {self.error_message}"[:600],
            })
        elif completed:
            self.completed = True

        if failed or completed:
            # Terminal frames replay the full output array. If any item never reached
            # us as its own frame, fold it in now so a dropped frame cannot silently
            # cost us a sub-query — and a failure can still carry partial output.
            for item in resp.get("output") or []:
                if isinstance(item, dict):
                    self._handle_item(item)

    def _handle_item(self, item: dict) -> None:
        item_type = item.get("type")
        # Dedupe on (type, id) where an id exists — it is stable across
        # .added/.updated/.done and the replay in response.completed repeats every
        # item again. When the id is MISSING, fall back to a hash of the canonicalized
        # item so an id-less reasoning or message item is still emitted exactly once
        # instead of two or three times.
        item_id = item.get("id") or _content_hash(item)
        key = f"{item_type}:{item_id}"
        if key in self._seen_ids:
            return
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
                continue
            if isinstance(md, dict) and md.get("preview_rows"):
                # Rows but no SQL: nothing to match against and nothing to re-execute,
                # so this can only be reported. Keeping it means the user hears about
                # it; dropping it here is the silent loss this module exists to avoid.
                self._sqlless_metadata.append(md)
            if text:
                # Parts WITHOUT metadata hold the prose; the metadata-bearing parts
                # are just markdown renderings of data we chart separately, so
                # excluding them keeps the narrative from duplicating every table.
                self._narrative_parts.append(text)

    # -- results -----------------------------------------------------------

    def narrative(self) -> str:
        parts = self._narrative_parts or self._all_parts
        return "\n\n".join(p.strip() for p in parts if p.strip())

    def narrative_parts(self) -> list[str]:
        """
        The narrative split at Genie's OWN part boundaries, for progressive emission.

        These are the real content parts of the final `message` item, not arbitrary
        slices of a finished string — this surface emits no text deltas (see
        `run_agent_mode`'s docstring), so part boundaries are the finest genuine
        granularity available.

        Concatenating the returned pieces reproduces `narrative()` exactly, so a client
        that accumulates them ends up with the same text it would have received in one
        piece (modulo the citation stripping the caller applies to the terminal event).
        """
        parts = [p.strip() for p in (self._narrative_parts or self._all_parts) if p.strip()]
        # Carry the joining separator on the pieces themselves so naive concatenation
        # matches narrative() rather than running paragraphs together.
        return [p if i == 0 else "\n\n" + p for i, p in enumerate(parts)]

    def subqueries(self) -> list[dict]:
        """
        Distinct sub-queries in first-seen order, each resolved against the message
        metadata where one is available.

        A metadata block that does not positively establish completeness (see
        `_metadata_is_complete`) is queued for re-execution instead: charting a
        partial result silently undercounts, which is exactly the failure mode the
        full-payload injection was built to avoid.
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
                if _metadata_is_complete(md, rows):
                    entry.update(columns=columns, rows=rows, source="metadata")
                else:
                    # Keep the schema; rows get refilled by re-execution.
                    entry.update(columns=columns, source="pending")
            out.append(entry)
        return out

    def orphan_metadata(self) -> list[dict]:
        """
        metadata blocks that did not map onto a function_call.

        Not observed on any probed run (every metadata.sql mapped back to a
        function_call), but the featured result must never be lost if it happens.

        Matching is checked twice before a block is declared an orphan: on the exact
        normalized key, then on the lenient key (comment-, semicolon- and
        case-insensitive). Without the second check, SQL that differs from its
        function_call only cosmetically would become BOTH a re-executed sub-query and
        an orphan chart — the same data charted twice.

        Blocks that arrived with rows but no SQL are included too. They cannot be
        matched or re-executed, so the caller can only report them — which is still
        strictly better than dropping a result set without a word.
        """
        loose_known = {_loose_sql_key(sq["sql"]) for sq in self._subqueries.values()}
        orphans: list[dict] = []
        for key, md in self._metadata.items():
            if key in self._subqueries:
                continue
            if _loose_sql_key(md.get("sql") or "") in loose_known:
                continue
            orphans.append(md)
        return orphans + self._sqlless_metadata


def _content_hash(item: dict) -> str:
    """Stable fallback dedupe key for an item that arrived without an id."""
    try:
        blob = json.dumps(item, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(sorted(item.items(), key=lambda kv: str(kv[0])))
    return "sha:" + hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


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
    parser = SSEFrameParser(on_discard=acc.note_discarded)

    # The stream is the only delivery mechanism, so a transport error here means
    # anything not already captured is gone. Surface the real error text — and then
    # still finish the recovery step with what WAS captured, because throwing away
    # good sub-queries on a late failure loses data twice over.
    stream_error: str | None = None
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
        stream_error = f"Genie agent mode timed out after {AGENT_TIMEOUT_SECONDS}s"
        emit({"type": "thinking", "text": stream_error})
    except Exception as e:
        stream_error = f"Genie agent mode failed: {e}"
        emit({"type": "thinking", "text": stream_error[:300]})

    return await _finalize(
        acc, emit, host=host, warehouse_id=warehouse_id, stream_error=stream_error
    )


async def _finalize(
    acc: AgentStreamAccumulator,
    emit,
    *,
    host: str,
    warehouse_id: str,
    stream_error: str | None = None,
) -> dict:
    """
    Post-stream recovery and result assembly, shared by the clean and failed paths.

    Runs on a transport failure too: the sub-queries captured before the failure are
    real, and re-executing the pending ones is bounded work, so a late failure
    degrades to "less data plus a real error" rather than to nothing.
    """
    loss = acc.frame_loss_message()
    if loss:
        emit({"type": "thinking", "text": loss})

    subqueries = acc.subqueries()

    # Any metadata block that matched no function_call still carries a real result;
    # keep it as its own sub-query rather than dropping it. It gets the SAME
    # completeness check as the matched ones — an orphan must not be the one place
    # partial rows sneak through to a chart.
    for md in acc.orphan_metadata():
        columns, numeric_flags = _columns_from_metadata(md.get("columns"))
        rows = _rows_from_metadata(md.get("preview_rows"), numeric_flags)
        sql = md.get("sql") or ""
        if _metadata_is_complete(md, rows) and rows:
            subqueries.append({
                "title": "", "sql": sql,
                "columns": columns, "rows": rows, "source": "metadata",
            })
        elif sql.strip():
            # Incomplete: keep the schema and let re-execution fill the rows.
            subqueries.append({
                "title": "", "sql": sql,
                "columns": columns, "rows": [], "source": "pending",
            })
        elif rows:
            emit({
                "type": "thinking",
                "text": "A result set arrived without its query and without a complete "
                        "row count, so it was not charted.",
            })

    # Re-execute the sub-queries Genie did not hand us structured data for, in
    # parallel. Ones already covered by metadata are skipped (never re-run).
    outstanding = [e for e in subqueries if e["source"] == "pending"]
    pending, skipped = outstanding[:MAX_REEXECUTIONS], outstanding[MAX_REEXECUTIONS:]

    for e in skipped:
        e["source"] = "skipped"
        e["error"] = f"exceeded the {MAX_REEXECUTIONS}-query recovery cap"
    if skipped:
        # One aggregate notice, not one per entry.
        emit({
            "type": "thinking",
            "text": f"{len(skipped)} further sub-quer"
                    f"{'y was' if len(skipped) == 1 else 'ies were'} not fetched: the "
                    f"recovery cap is {MAX_REEXECUTIONS} queries per question.",
        })

    if pending:
        emit({
            "type": "thinking",
            "text": f"Fetching full results for {len(pending)} sub-quer"
                    f"{'y' if len(pending) == 1 else 'ies'}...",
        })
        results = await asyncio.gather(
            *(_reexecute(e, host, warehouse_id, emit) for e in pending),
            return_exceptions=True,
        )
        # _reexecute handles its own failures, so anything returned here escaped it.
        # Inspecting the results is the difference between reporting that and losing
        # the sub-query without a word.
        for entry, outcome in zip(pending, results):
            if not isinstance(outcome, BaseException):
                continue
            label = entry.get("title") or entry["sql"][:60]
            entry["source"] = "error"
            entry["error"] = f"{type(outcome).__name__}: {outcome}"
            emit({
                "type": "thinking",
                "text": f"Could not fetch data for '{label}': {entry['error'][:200]}",
            })

    # A real error, when we have one, must never be replaced by the generic message.
    errors = [m for m in (acc.error_message, stream_error) if m]
    if not errors and not acc.completed:
        emit({
            "type": "thinking",
            "text": "Genie agent stream ended before signalling completion; "
                    "reporting what was received.",
        })

    result_error = " | ".join(errors) if errors else None
    if result_error:
        return _error_result(result_error, acc, subqueries)
    return _build_result(acc, subqueries)


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


def _build_result(acc: AgentStreamAccumulator, subqueries: list[dict]) -> dict:
    """Assemble the plain-path result shape plus the recovered sub-queries."""
    featured = next((e for e in subqueries if e["rows"]), None) or {}
    return {
        "text": acc.narrative(),
        # Genie's own part boundaries, for progressive `genie_text_delta` emission.
        # Concatenating these reproduces `text` exactly.
        "text_parts": acc.narrative_parts(),
        "sql": featured.get("sql", ""),
        "columns": featured.get("columns", []),
        "rows": featured.get("rows", []),
        "sub_queries": subqueries,
        "reasoning_count": acc.reasoning_count,
        "conversation_id": acc.conversation_id,
        "response_id": acc.response_id,
    }


def _error_result(
    message: str, acc: AgentStreamAccumulator, subqueries: list[dict] | None = None
) -> dict:
    """
    Result carrying the real error text, plus whatever the stream did deliver.

    "Whatever the stream delivered" is literal: the sub-queries captured before the
    failure are returned, not an empty list. On this sole-delivery surface, dropping
    them means they are gone for good.
    """
    result = _build_result(acc, acc.subqueries() if subqueries is None else subqueries)
    result["result_error"] = message
    return result
