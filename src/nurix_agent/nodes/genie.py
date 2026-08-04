import asyncio
import json
import re
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..config import AppConfig
from ..state import AgentState


def _try_parse_genie_result(content) -> tuple[list[dict], list[list]] | None:
    """
    Parse columns and rows from a Genie MCP tool result.

    Handles:
    - List of content blocks: [{"type": "text", "text": "<json>"}]  (Shape A — streamable_http)
    - Direct JSON string (Shape B)
    - Genie native {"content": {"queryAttachments": [{"statement_response": ...}]}} (Shape C)
    - {"columns": [...], "rows": [...]} (Shape D)
    - {"result": {"columns": [...], "rows": [...]}} (Shape E)
    - List of row dicts (Shape F)
    - Markdown table fallback
    """
    # Shape A — list of content blocks; recurse into each text block
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                result = _try_parse_genie_result(block["text"])
                if result:
                    return result
        return None

    if not isinstance(content, str) or len(content) < 10:
        return None

    # Try JSON parse
    try:
        data = json.loads(content)

        # Shape C — Genie native: {"content": {"queryAttachments": [...]}}
        if isinstance(data, dict) and "content" in data:
            attachments = data["content"].get("queryAttachments", [])
            for att in attachments:
                sr = att.get("statement_response", {})
                if sr.get("status", {}).get("state") != "SUCCEEDED":
                    continue
                manifest = sr.get("manifest", {})
                result = sr.get("result", {})
                schema_cols = manifest.get("schema", {}).get("columns", [])
                data_array = result.get("data_array", [])  # JSON_ARRAY format
                if schema_cols and data_array:
                    cols = [{"name": c["name"], "type": c.get("type_text", "string")}
                            for c in schema_cols]
                    rows = []
                    for row in data_array:
                        if isinstance(row, list):
                            rows.append(row)
                        elif isinstance(row, dict):
                            vals = [v.get("string_value") for v in row.get("values", [])]
                            rows.append(vals)
                    return cols, rows

        # Shape D — {"columns": [...], "rows": [...]}
        if isinstance(data, dict) and "columns" in data and "rows" in data:
            cols = [{"name": c if isinstance(c, str) else c.get("name", str(c)), "type": "string"}
                    for c in data["columns"]]
            return cols, data["rows"]

        # Shape E — {"result": {"columns": [...], "rows": [...]}}
        if isinstance(data, dict) and "result" in data:
            inner = data["result"]
            if isinstance(inner, dict) and "columns" in inner:
                cols = [{"name": c.get("name", c) if isinstance(c, dict) else c,
                         "type": c.get("type", "string") if isinstance(c, dict) else "string"}
                        for c in inner["columns"]]
                return cols, inner.get("rows", [])

        # Shape F — list of row dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            cols = [{"name": k, "type": "string"} for k in data[0].keys()]
            rows = [[r.get(c["name"]) for c in cols] for r in data]
            return cols, rows

    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown table fallback
    lines = [l.strip() for l in content.split("\n") if "|" in l]
    if len(lines) >= 2:
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        data_lines = [l for l in lines[2:] if not set(l.replace("|", "").replace("-", "").strip()) <= {""}]
        if headers and data_lines:
            cols = [{"name": h, "type": "string"} for h in headers]
            rows = []
            for line in data_lines[:100]:
                vals = [v.strip() for v in line.split("|")]
                if vals and vals[0] == "":
                    vals = vals[1:]
                if vals and vals[-1] == "":
                    vals = vals[:-1]
                rows.append(vals[:len(headers)])
            return cols, rows

    return None


def _parse_genie_text_response(text: str, emit, index: int) -> dict:
    """
    Parse Genie's final markdown text response into structured data.
    Extracts: SQL, narrative text, and table data (columns + rows).
    """
    if not text or not text.strip():
        return {"text": "", "sql": "", "columns": [], "rows": []}

    # Extract SQL from fenced code block
    sql = ""
    sql_match = re.search(r'```sql\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()

    # Extract table data from embedded query blocks (most reliable)
    # Try <!-- begin-embedded:query_xxx --> ... <!-- end-embedded:query_xxx --> first
    table_text = ""
    embedded_match = re.search(
        r'<!-- begin-embedded:[^>]+ -->\s*\n(.*?)\n<!-- end-embedded:[^>]+ -->',
        text, re.DOTALL
    )
    if embedded_match:
        table_text = embedded_match.group(1).strip()
    else:
        # Try <!-- begin:query_xxx --> ... <!-- end:query_xxx -->
        begin_match = re.search(
            r'<!-- begin:[^>]+ -->\s*\n(.*?)\n<!-- end:[^>]+ -->',
            text, re.DOTALL
        )
        if begin_match:
            table_text = begin_match.group(1).strip()

    # Parse the markdown table
    columns = []
    rows = []
    if table_text:
        lines = [l.strip() for l in table_text.split('\n') if l.strip()]
        # Filter out separator lines (| --- | --- |)
        data_lines = [l for l in lines if l.startswith('|') and not re.match(r'^[\|\s\-:]+$', l)]
        if len(data_lines) >= 2:
            # Header row
            headers = [h.strip() for h in data_lines[0].split('|') if h.strip()]
            columns = [{"name": h, "type": "string"} for h in headers]
            # Data rows
            for line in data_lines[1:]:
                vals = [v.strip() for v in line.split('|') if v.strip() != '' or True]
                # Remove empty leading/trailing from split
                parts = line.split('|')
                parts = [p.strip() for p in parts]
                # Remove first and last empty strings from leading/trailing |
                if parts and parts[0] == '':
                    parts = parts[1:]
                if parts and parts[-1] == '':
                    parts = parts[:-1]
                if len(parts) == len(headers):
                    # Try to cast numeric values
                    row = []
                    for v in parts:
                        try:
                            row.append(int(v.replace(',', '')))
                        except ValueError:
                            try:
                                row.append(float(v.replace(',', '')))
                            except ValueError:
                                row.append(v)
                    rows.append(row)
                    # Update column types based on data
                    for i_col, val in enumerate(row):
                        if isinstance(val, (int, float)) and columns[i_col]["type"] == "string":
                            columns[i_col]["type"] = "number"

    # If no table found, try the general markdown table parser from _try_parse_genie_result
    if not columns:
        parsed = _try_parse_genie_result(text)
        if parsed:
            columns, rows = parsed

    # Extract narrative text (everything that's not SQL, not table markers, not HTML comments)
    narrative = re.sub(r'```sql.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    narrative = re.sub(r'<!--.*?-->', '', narrative, flags=re.DOTALL)
    narrative = re.sub(r'<!-- begin.*?<!-- end[^>]*>', '', narrative, flags=re.DOTALL)
    narrative = re.sub(r'\[.*?\]\(https?://.*?\)', '', narrative)  # remove links
    narrative = re.sub(r'\*\*Status:\*\*.*?\n', '', narrative)
    narrative = re.sub(r'<details>.*?</details>', '', narrative, flags=re.DOTALL)
    narrative = re.sub(r'\n{3,}', '\n\n', narrative).strip()
    # Take first paragraph as the summary
    paragraphs = [p.strip() for p in narrative.split('\n\n') if p.strip() and not p.strip().startswith('#') and len(p.strip()) > 20]
    genie_text = paragraphs[0] if paragraphs else ""

    # Emit events
    if genie_text:
        emit({"type": "genie_text", "text": genie_text, "index": index})
    if sql:
        emit({"type": "sql", "sql": sql, "index": index})

    return {"text": genie_text, "sql": sql, "columns": columns, "rows": rows}


async def _call_genie_for_question(question: str, index: int, cfg: AppConfig, token: str, emit) -> dict:
    """Call workspace-wide Genie MCP for a single sub-question using ask+poll pattern."""
    emit({"type": "thinking", "text": f"Querying Genie for: {question[:60]}...", "index": index})

    mcp_client = MultiServerMCPClient({
        "genie": {
            "url": cfg.genie_mcp_url,
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {token}",
                "X-Databricks-Genie-Space-Id": cfg.genie_space_id,
            }
        }
    })

    try:
        tools = await mcp_client.get_tools()
        if not tools:
            return {"text": "No Genie tools available", "sql": "", "columns": [], "rows": []}

        # Find genie_ask and genie_poll_response tools
        ask_tool = next((t for t in tools if t.name == "genie_ask"), None)
        poll_tool = next((t for t in tools if t.name == "genie_poll_response"), None)

        if not ask_tool:
            # Fallback: try the first tool with 'question' param
            ask_tool = next((t for t in tools if "query" in t.name.lower() or "ask" in t.name.lower()), tools[0])

        # Step 1: Call genie_ask
        ask_result = await ask_tool.ainvoke({"question": question})

        # Extract conversation_id and response_id from ask result
        conversation_id = None
        response_id = None
        ask_status = "unknown"

        raw_text = ""
        if isinstance(ask_result, list):
            for block in ask_result:
                if isinstance(block, dict) and block.get("type") == "text":
                    raw_text += block.get("text", "")
        elif isinstance(ask_result, str):
            raw_text = ask_result

        # Try to parse as JSON to get conversation_id/response_id
        try:
            parsed = json.loads(raw_text.strip())
            conversation_id = parsed.get("conversation_id")
            response_id = parsed.get("response_id")
            ask_status = parsed.get("status", "unknown")
        except (json.JSONDecodeError, ValueError):
            pass

        # If ask returned completed directly (streaming host), parse it now
        if ask_status == "completed" or (not poll_tool) or (not conversation_id):
            return _parse_genie_text_response(raw_text, emit, index)

        # Step 2: Poll until completed (max 20 attempts x 3s = 60s).
        # The poll response is markdown narrative, NOT JSON:
        #   - in-progress turns start with "Still running" and have no data yet
        #   - the completed turn contains "**Status:** completed" and the data
        #     embed blocks (<!-- begin:query_xxx --> / <!-- begin-embedded:... -->)
        #   - a failed turn contains "**Status:** failed"
        final_text = raw_text
        for attempt in range(20):
            await asyncio.sleep(3)
            try:
                poll_result = await poll_tool.ainvoke({
                    "conversation_id": conversation_id,
                    "response_id": response_id,
                })
                poll_text = ""
                if isinstance(poll_result, list):
                    for block in poll_result:
                        if isinstance(block, dict) and block.get("type") == "text":
                            poll_text += block.get("text", "")
                elif isinstance(poll_result, str):
                    poll_text = poll_result

                # Some hosts return a JSON status envelope instead of markdown.
                poll_status = ""
                try:
                    poll_parsed = json.loads(poll_text.strip())
                    if isinstance(poll_parsed, dict):
                        poll_status = poll_parsed.get("status", "")
                except (json.JSONDecodeError, ValueError):
                    pass

                lowered = poll_text.lower()
                is_failed = poll_status == "failed" or "**status:** failed" in lowered
                # Completion is signalled by the status marker or the presence of
                # actual result data (embed blocks / SQL). Guard against the
                # in-progress "Still running" narrative which has neither.
                is_completed = (
                    poll_status == "completed"
                    or "**status:** completed" in lowered
                    or "<!-- begin:" in poll_text
                    or "<!-- begin-embedded:" in poll_text
                )

                if is_failed:
                    emit({"type": "thinking", "text": "Genie query failed", "index": index})
                    return {"text": "", "sql": "", "columns": [], "rows": []}
                if is_completed:
                    final_text = poll_text
                    break
                # else still in_progress, keep polling
            except Exception:
                if attempt >= 5:
                    break

        return _parse_genie_text_response(final_text, emit, index)

    except Exception as e:
        emit({"type": "thinking", "text": f"Genie error: {str(e)[:100]}", "index": index})
        return {"text": "", "sql": "", "columns": [], "rows": []}


async def genie_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]

    # Get token
    from databricks.sdk import WorkspaceClient
    ws = WorkspaceClient(host=cfg.databricks_host)
    auth = ws.config.authenticate()
    token = auth.get("Authorization", "").replace("Bearer ", "")
    if not token:
        emit({"type": "error", "message": "No Databricks token available"})
        return {"genie_results": []}

    sub_questions = state["sub_questions"]

    async def _run_with_timeout(q: str, i: int) -> dict:
        # Bound each sub-question (ask + poll loop ~60s) so one hung task
        # can't block the whole gather.
        try:
            async with asyncio.timeout(90):
                return await _call_genie_for_question(q, i, cfg, token, emit)
        except asyncio.TimeoutError:
            emit({"type": "thinking", "text": f"Genie timed out for: {q[:60]}", "index": i})
            return {"text": "", "sql": "", "columns": [], "rows": []}

    # Run all sub-questions in parallel
    tasks = [
        _run_with_timeout(q, i)
        for i, q in enumerate(sub_questions)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    genie_results = []
    for r in results:
        if isinstance(r, Exception):
            genie_results.append({"text": "", "sql": "", "columns": [], "rows": []})
        else:
            genie_results.append(r)

    return {"genie_results": genie_results}
