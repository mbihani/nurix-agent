import asyncio
import json
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
                    for row_obj in data_array:
                        vals = [v.get("string_value") for v in row_obj.get("values", [])]
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


async def _call_genie_for_question(question: str, index: int, cfg: AppConfig, token: str, emit) -> dict:
    """Call workspace-wide Genie MCP for a single sub-question."""
    emit({"type": "thinking", "text": f"Querying Genie for: {question[:60]}...", "index": index})

    mcp_client = MultiServerMCPClient({
        "genie": {
            "url": cfg.genie_mcp_url,  # workspace-wide: /api/2.0/mcp/genie (NO space_id in URL)
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

        # Find the query tool (first tool available)
        genie_tool = tools[0]

        # Call the tool
        raw_result = await genie_tool.ainvoke({"query": question})

        # Extract Genie's narrative text response
        genie_text = ""
        if isinstance(raw_result, list):
            for block in raw_result:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    # Genie text is the non-JSON part
                    try:
                        json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        if text.strip():
                            genie_text += text.strip() + " "
        elif isinstance(raw_result, str):
            try:
                json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                genie_text = raw_result.strip()

        # Extract SQL
        sql = ""
        if isinstance(raw_result, list):
            for block in raw_result:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict):
                            content = data.get("content", {})
                            attachments = content.get("queryAttachments", []) if isinstance(content, dict) else []
                            for att in attachments:
                                sr = att.get("statement_response", {})
                                sql = sr.get("statement", "") or sql
                    except (json.JSONDecodeError, ValueError):
                        pass

        # Parse columns and rows
        parsed = _try_parse_genie_result(raw_result)
        columns, rows = parsed if parsed else ([], [])

        # Emit events
        if genie_text:
            emit({"type": "genie_text", "text": genie_text.strip(), "index": index})
        if sql:
            emit({"type": "sql", "sql": sql, "index": index})

        return {"text": genie_text, "sql": sql, "columns": columns, "rows": rows}

    except Exception as e:
        emit({"type": "thinking", "text": f"Genie error: {str(e)[:100]}", "index": index})
        return {"text": "", "sql": "", "columns": [], "rows": []}


async def genie_node(state: AgentState, config: dict) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]

    # Get token
    from databricks.sdk import WorkspaceClient
    ws = WorkspaceClient()
    auth = ws.config.authenticate()
    token = auth.get("Authorization", "").replace("Bearer ", "")

    sub_questions = state["sub_questions"]

    # Run all sub-questions in parallel
    tasks = [
        _call_genie_for_question(q, i, cfg, token, emit)
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
