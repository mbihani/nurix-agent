import asyncio
import datetime
import re

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


# Column-name tokens that indicate a MEASURE (something worth plotting on a value
# axis). Used only to rescue a column the type metadata mislabelled as STRING.
_MEASURE_NAME_TOKENS = frozenset({
    "count", "total", "sum", "amount", "rate", "score", "avg", "average",
    "pct", "percent", "percentage", "ratio", "revenue", "price", "cost",
    "qty", "quantity", "num", "n", "median", "min", "max", "delta", "change",
    "growth",
})

# Column-name tokens that indicate an IDENTIFIER or a calendar part — digits that
# are a LABEL, not a magnitude. "2023" and "560001" parse as numbers but plotting
# them on a value axis is nonsense, which is precisely the low-value chart the recon
# filter exists to remove. This guard WINS over measure-name evidence, so
# "order_number" is rejected despite the measure-ish "number"/"num" token.
_IDENTIFIER_NAME_TOKENS = frozenset({
    "id", "ids", "code", "codes", "zip", "zipcode", "postal", "postcode",
    "phone", "telephone", "mobile", "fax", "year", "yr", "month", "day",
    "date", "datetime", "timestamp", "week", "quarter", "uuid", "guid",
    "key", "number", "no", "account", "ssn", "sku", "isbn", "ein", "vat",
})

# Thousands separators / percent / currency in the RAW string. This is the
# formatting evidence that a string column was rendered as a MEASURE: a bare digit
# run like "2023" carries no such marker and is deliberately not matched.
_MEASURE_FORMAT_RE = re.compile(r"[%$£€¥]|\d,\d{3}(?:\D|$)")


def _name_tokens(name: str) -> set[str]:
    """Lowercase word tokens of a column name, splitting on non-alphanumerics."""
    return {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t}


def _looks_like_identifier_name(name: str) -> bool:
    """True if the column name marks it as an identifier or a calendar part."""
    return bool(_name_tokens(name) & _IDENTIFIER_NAME_TOKENS)


def _looks_like_measure_name(name: str) -> bool:
    """
    True if the column name marks it as a measure.

    Also matches the `n_`/`_n` prefix/suffix convention for counts, which
    tokenizes to a bare "n".
    """
    return bool(_name_tokens(name) & _MEASURE_NAME_TOKENS)


def _has_measure_formatting(rows: list, col_index: int) -> bool:
    """
    True if any raw string value carries measure FORMATTING (%, currency, 1,234).

    Deliberately checked on the RAW text, before `_coerce` strips separators: the
    formatting is the evidence, and coercion would erase it.
    """
    for row in rows or []:
        if col_index >= len(row):
            continue
        value = row[col_index]
        if isinstance(value, str) and _MEASURE_FORMAT_RE.search(value):
            return True
    return False


def _has_numeric_column(columns: list, rows: list | None = None) -> bool:
    """
    Whether a result set carries at least one numeric (measure) column.

    Lives next to `_is_numeric_type` so numeric detection has exactly ONE home.
    Callers see columns AFTER `_extract_columns_rows` / `_columns_from_metadata`
    have folded the raw Databricks type through `_is_numeric_type` into the
    normalized {"name", "type": "number"|"string"} shape, so the normalized
    marker is what we read. Re-running `_is_numeric_type("number")` here would
    return False ("NUMBER" is not a Databricks type name) — the exact kind of
    second, inconsistent implementation this helper exists to prevent.

    A raw, un-normalized `type` is still accepted (delegating to
    `_is_numeric_type`) so the helper is correct wherever it is called from.

    When `rows` is supplied, a column the TYPE says is a string may be RESCUED as a
    measure — Genie sometimes types a real measure as STRING, and type metadata alone
    would throw away a chartable result. Types are still consulted FIRST, and value
    sniffing only ever ADDS a numeric column, never removes one.

    The rescue demands STRONG measure evidence, because "every value parses as a
    number" is a statement about STORAGE REPRESENTATION, not measure SEMANTICS:
    customer ids, years, and ZIP codes all parse, and charting them on a value axis
    is exactly the nonsense the recon filter exists to remove. So a rescue needs
    values that all parse AND at least one of:

      1. FORMATTING evidence — %, a currency symbol, or thousands separators in the
         raw text ("42%", "$12.50", "1,234"). A bare digit run is NOT evidence.
      2. NAME evidence — a measure-ish token (count, total, rate, pct, ...).

    An identifier-ish name (id, zip, year, phone, order_number, ...) VETOES the
    rescue outright, outranking name evidence — that is what keeps "order_number"
    and "year" out despite their measure-ish tokens.
    """
    string_typed: list[int] = []
    for i, c in enumerate(columns or []):
        if not isinstance(c, dict):
            continue
        type_text = c.get("type")
        if not isinstance(type_text, str):
            continue
        if type_text.strip().lower() == "number" or _is_numeric_type(type_text):
            return True
        string_typed.append(i)

    if rows:
        for i in string_typed:
            name = columns[i].get("name", "") if isinstance(columns[i], dict) else ""
            # The identifier veto is checked FIRST and wins over name evidence.
            if _looks_like_identifier_name(name):
                continue
            if not (_has_measure_formatting(rows, i) or _looks_like_measure_name(name)):
                continue
            if _column_values_are_numeric(rows, i):
                return True
    return False


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


def _column_values_are_numeric(rows: list, col_index: int) -> bool:
    """
    Whether a column's actual CELL VALUES all parse as numbers.

    PARSEABILITY ONLY — this says nothing about whether the column is a measure.
    Ids, years and ZIP codes all pass. `_has_numeric_column` therefore requires
    separate measure evidence (formatting or name) and applies an identifier veto
    before consulting this; do not use it alone as a chartability test.

    Parsing goes through `_coerce` — the SAME coercion the row pipeline uses — so
    what counts as numeric here cannot drift from what the charts actually receive.
    A trailing unit suffix (%, currency) is tolerated by retrying the stripped text,
    since the underlying magnitude is still plottable.

    Conservative by construction: EVERY non-null value must parse, and there must be
    at least one. One stray label means the column is text, not a measure.
    """
    seen = False
    for row in rows or []:
        if col_index >= len(row):
            return False
        value = row[col_index]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue  # nulls/blanks do not disqualify a measure
        seen = True
        if isinstance(value, bool):
            return False  # a flag is not a measure
        if isinstance(value, (int, float)):
            continue
        if not isinstance(value, str):
            return False
        if isinstance(_coerce(value, True), (int, float)):
            continue
        # Retry without a single leading/trailing unit marker (%, $, £, €).
        stripped = value.strip().strip("%").strip("$£€").strip()
        if stripped and isinstance(_coerce(stripped, True), (int, float)):
            continue
        return False
    return seen


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

    Pulls the SQL/description from the query attachment(s) and the narrative
    from the text attachment, then fetches the tabular result. The message-level
    query-result endpoint was observed to return 0 rows for this space, so we
    fetch the attachment-scoped result (which carries the data) and only fall
    back to the message-level result when no attachment returns rows.

    A message can carry more than one query attachment; we try each in order and
    use the first that returns non-empty rows (aligning the emitted SQL/narrative
    with that attachment) rather than blindly taking the last.

    If a result-retrieval call raises and neither path yields rows, the Databricks
    error text is returned under "result_error" so the caller can surface it as a
    clean event instead of silently emitting an empty dataset.
    """
    if msg.status in (MessageStatus.FAILED, MessageStatus.CANCELLED):
        err = ""
        if msg.error is not None:
            err = getattr(msg.error, "error", None) or str(msg.error)
        return {"text": err or "Genie query failed", "sql": "", "columns": [], "rows": [], "failed": True}

    narrative = ""
    query_attachments: list[tuple[str | None, str, str]] = []
    for att in (msg.attachments or []):
        if att.text is not None and att.text.content:
            narrative = att.text.content
        if att.query is not None:
            query_attachments.append((
                att.attachment_id,
                att.query.query or "",
                att.query.description or "",
            ))

    # Default the SQL/description to the first query attachment so we still emit
    # a sensible SQL/text even when no attachment ultimately returns rows.
    sql = query_attachments[0][1] if query_attachments else ""
    description = query_attachments[0][2] if query_attachments else ""

    columns: list[dict] = []
    rows: list[list] = []
    # First result-retrieval exception seen (attachment path preferred over the
    # message-level fallback); surfaced only if we end up with no rows.
    retrieval_error: Exception | None = None

    # Try each query attachment in order; the first that returns non-empty rows
    # wins, and we align the emitted SQL/description with that attachment.
    for att_id, att_sql, att_desc in query_attachments:
        if not att_id:
            continue
        try:
            res = w.genie.get_message_attachment_query_result(
                space_id, msg.conversation_id, msg.message_id, att_id
            )
            cols_i, rows_i = _extract_columns_rows(res.statement_response)
        except Exception as e:
            retrieval_error = retrieval_error or e
            continue
        if rows_i:
            columns, rows = cols_i, rows_i
            sql = att_sql or sql
            description = att_desc or description
            break
        if cols_i and not columns:
            columns = cols_i  # keep the schema even if this attachment had no rows

    # Fallback: message-level result (rarely populated for this space, but cheap
    # to try) when no attachment returned rows.
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
        except Exception as e:
            retrieval_error = retrieval_error or e

    out = {"text": narrative or description, "sql": sql, "columns": columns, "rows": rows}

    # A retrieval call raised and neither path produced rows: surface the real
    # Databricks error instead of silently degrading to an empty dataset.
    if not rows and retrieval_error is not None:
        out["result_error"] = str(retrieval_error)

    return out


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
        # Narrative/SQL may have been extracted fine while the tabular result
        # retrieval failed; surface that real error rather than an empty chart.
        if result.get("result_error"):
            emit({"type": "thinking", "text": f"Genie result retrieval failed: {result['result_error'][:200]}", "index": index})

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
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            # An unexpected error escaped the per-question handling; surface its
            # text instead of silently substituting an empty dataset.
            q = sub_questions[i] if i < len(sub_questions) else ""
            emit({"type": "thinking", "text": f"Genie query failed for '{q[:60]}': {str(r)[:200]}", "index": i})
            genie_results.append({"text": "", "sql": "", "columns": [], "rows": []})
        else:
            genie_results.append(r)

    return {"genie_results": genie_results}
