import asyncio
import json
import re
from html.parser import HTMLParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import mlflow
from ..config import AppConfig, get_databricks_token
from ..state import AgentState

VISUALIZATION_GUIDE = (
    "You are a data visualization expert. "
    "\n\nVISUALIZATION SELECTION GUIDE\n\n"
    "Choose the right chart type based on the data pattern:\n"
    "- Trend over time → Line or Area chart\n"
    "- Comparing categories → Bar chart (horizontal if >6 categories)\n"
    "- Part of a whole / proportions → Doughnut/Pie (max 6 slices; group rest as Other)\n"
    "- Distribution / spread / outliers → Histogram\n"
    "- Relationship between two numeric variables → Scatter plot\n"
    "- Flow through sequential stages → Funnel\n"
    "- Single KPI → Counter (large number, no axes)\n"
    "- Detailed data / high cardinality → Table\n"
    "\nAVAILABLE CHART TYPES (Chart.js): line, bar, doughnut, scatter, area (fill:true on line), "
    "histogram (bar with equal bins), counter (custom HTML), table (HTML table)\n"
    "\nANTI-PATTERNS — never do these:\n"
    "- Pie/doughnut with more than 6 slices — use bar instead\n"
    "- Bar chart for time series data — use line\n"
    "- Line chart for categorical (non-temporal) x-axis — use bar\n"
    "- High-cardinality color grouping (>10 unique values) — aggregate to Top-N + Other\n"
    "- Multiple counters when comparison matters — use bar\n"
    "\nCOLOR PALETTE (Databricks brand, use in this order for series):\n"
    "#FF3621, #2272B4, #00A972, #F6A623, #1B3139, #9B59B6, #E74C3C, #3498DB\n"
    "- #00A972 = positive/good, #FF3621 = negative/error, #2272B4 = primary blue\n"
    "- Chart background: transparent, page background: #1B1B1B\n"
    "\nSORTING:\n"
    "- Questions with 'top', 'most', 'highest', 'largest', 'best' → sort descending by metric\n"
    "- Time series → always sort chronologically ascending\n"
    "\nCHART QUALITY:\n"
    "- Always show axis labels and a chart title\n"
    "- Show gridlines on line/area charts\n"
    "- Use horizontal bar if >8 categories\n"
    "- Abbreviate large numbers (K, M, B) on axes\n"
    "- Limit legend to 8 entries max\n"
    "- Output a single H3 heading above the chart — no narrative paragraphs\n"
    "\nOUTPUT REQUIREMENTS:\n"
    "- Output ONLY a complete self-contained HTML document\n"
    "- Use Chart.js via CDN: https://cdn.jsdelivr.net/npm/chart.js\n"
    "- Include a single H3 heading (the question) above the chart\n"
    "- Chart fills full width, height 100%\n"
    "- Use Databricks brand colors: primary #FF3621, blue #2272B4, rest of palette above\n"
    "- NO explanatory text, NO markdown fences, just raw HTML starting with <!DOCTYPE html>\n"
)

# Well-known JS global that the Python node injects the FULL dataset into after
# the LLM returns the chart scaffold. The LLM authors JS that READS from this
# global rather than re-typing data rows inline (which truncates on large
# result sets and undercounts the chart).
CHART_DATA_INJECTION_GUIDE = """
DATA SOURCE — CRITICAL (read carefully):
- The COMPLETE dataset is injected into the page as a JavaScript global BEFORE your script runs:
    window.CHART_DATA = {
      "columns": [ {"name": "<col name>", "type": "number" | "string"}, ... ],
      "rows":    [ [<cell>, <cell>, ...], ... ]   // each row aligned by position to columns
    }
- Build ALL labels, datasets, and aggregations at runtime by iterating over window.CHART_DATA.rows and
  looking up each column's index by matching its name in window.CHART_DATA.columns.
- DO NOT hardcode, inline, or re-type ANY data values, labels, or numbers into the HTML/JS. The sample rows
  shown to you below are ILLUSTRATIVE ONLY and are deliberately truncated — the real dataset lives entirely in
  window.CHART_DATA and may be far larger. Your JS must work for the FULL dataset, not just the sample.
- If the chart needs grouping/summing (e.g. totals per category), compute it in JS over window.CHART_DATA.rows.
- DO NOT declare, reassign, or overwrite window.CHART_DATA — only READ from it. You may alias it locally,
  e.g. `const data = window.CHART_DATA;`.
"""

CHART_SYSTEM_PROMPT = VISUALIZATION_GUIDE + """
Generate a SINGLE self-contained HTML file with:
- One H3 heading from the question (no other text)
- One Chart.js chart
- NO narrative paragraphs, NO analysis text, NO "Here is" phrases
- Inline <meta http-equiv="Content-Security-Policy" content="connect-src 'none'">
- Databricks brand colors: primary #FF3621, blue #2272B4, series palette [#2272B4, #FF8C00, #00A36C, #9467BD, #E15759, #76B7B2]
- window.global = window polyfill not needed (no Plotly)
""" + CHART_DATA_INJECTION_GUIDE

REFINE_SYSTEM_PROMPT = """
You are a chart refinement assistant. The user has an existing Chart.js HTML visualization and wants to modify it.
Apply the instruction to the existing HTML and return the complete updated HTML.
Preserve the H3 heading unless the instruction changes the topic.

DATA SOURCE — CRITICAL:
- The chart's data is supplied at runtime via the JavaScript global window.CHART_DATA
  (an object with "columns" and "rows"). This global is preserved and re-injected automatically;
  you will NOT see it in the HTML given to you and you MUST NOT re-create or inline it.
- Keep building labels/datasets by reading from window.CHART_DATA at runtime. Never hardcode or re-type data rows.

Do NOT add narrative paragraphs. Return ONLY the complete HTML, no markdown.
"""

INSIGHT_SYSTEM_PROMPT = """
You are a data analyst. The user has a visualization based on customer feedback data and wants deeper insight.
Given the chart HTML (which contains the data) and the original SQL, answer their question in 2-4 concise sentences.
Be specific about numbers and trends visible in the data. Do not generate a new chart.
"""


def _embed_json(data) -> str:
    """
    Serialize data to a JS literal safe to embed inside a <script> element.

    Escapes <, >, & (and the JS line-separator code points) to their \\uXXXX
    forms so the payload can never terminate the <script> tag early ("</script>"),
    open an HTML comment, or otherwise break out of the JS context — while still
    parsing back to the exact original characters at runtime.
    """
    return (
        json.dumps(data, ensure_ascii=False, default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


class _FirstScriptFinder(HTMLParser):
    """
    Locate the (line, col) of the FIRST real, EXECUTABLE <script> START tag.

    Subclassing stdlib HTMLParser (no new deps) makes this HTML-context-aware:
    the parser does NOT report a `<script` literal that sits inside an HTML
    comment (`<!-- <script> -->`) or inside a <script>/<style> rawtext element as
    a start tag, so those never false-match the way a bare regex would.

    RCDATA elements (<textarea>, <title>) need explicit handling. Browsers treat
    their contents as raw text, so a `<script></script>` literal inside them is
    inert markup, never an executable script. Some stdlib versions do NOT
    implement that RCDATA semantics for these two tags, so a COMPLETE
    `<script></script>` inside <textarea>/<title> can still fire
    handle_starttag('script') — which would make us inject window.CHART_DATA
    INSIDE the textarea/title (inert text), leaving it undefined when the real
    chart script runs. To be correct on every stdlib version we track RCDATA
    nesting depth ourselves and only record a <script> position while that depth
    is 0. This is a no-op on versions where the parser already suppresses the
    inner <script>, and the needed guard on versions where it does not.
    """

    _RCDATA_TAGS = ("textarea", "title")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pos: tuple[int, int] | None = None  # (line, col) of first <script>
        self._rcdata_depth = 0  # >0 while inside <textarea>/<title>

    def handle_starttag(self, tag, attrs):
        if tag in self._RCDATA_TAGS:
            self._rcdata_depth += 1
            return
        if self.pos is None and tag == "script" and self._rcdata_depth == 0:
            self.pos = self.getpos()

    def handle_endtag(self, tag):
        if tag in self._RCDATA_TAGS and self._rcdata_depth > 0:
            self._rcdata_depth -= 1


def _linecol_to_index(html: str, line: int, col: int) -> int:
    """Convert HTMLParser's 1-based line / 0-based col to an absolute string index."""
    lines = html.split("\n")
    # Sum full prior lines plus their stripped '\n' separators, then add the column.
    idx = sum(len(lines[i]) + 1 for i in range(line - 1)) + col
    return min(idx, len(html))


# Sentinel distinguishing a PARSE FAILURE from a clean parse that found no
# <script>. The two demand opposite fallbacks: on a parse failure we don't know
# where any script sits, so the ONLY way to guarantee window.CHART_DATA is
# defined before anything executes is to PREPEND; a clean no-script result means
# ordering is irrelevant and the tidy <head> placement is fine.
_PARSE_FAILED = object()


def _first_executable_script_index(html: str):
    """
    Locate the first EXECUTABLE <script> start tag.

    Returns one of three distinct signals:
      - int           : absolute index of the first executable <script>.
      - None          : CLEAN parse that found NO executable <script>.
      - _PARSE_FAILED : the parser raised on malformed input (location unknown).

    Uses an HTML-context-aware parse so a `<script` literal inside a comment or a
    RCDATA element (<textarea>, <title>) is ignored rather than false-matching.
    """
    finder = _FirstScriptFinder()
    try:
        finder.feed(html)
        finder.close()
    except Exception:
        return _PARSE_FAILED
    if finder.pos is None:
        return None
    line, col = finder.pos
    return _linecol_to_index(html, line, col)


def _insert_head_script(html: str, script: str) -> str:
    """
    Insert `script` so the global it defines is defined BEFORE any other script runs.

    Ordering is unconditional: we place the data <script> immediately before the
    first EXECUTABLE <script> in the document (found via an HTML-context-aware
    parse, so a `<script` literal inside a comment or RCDATA element does not
    false-match), regardless of where <head> sits — so window.CHART_DATA is
    guaranteed to be defined before the first chart script runs even for a
    malformed scaffold whose <script> precedes its <head>.

    Fallbacks preserve the ordering contract in both no-index cases:
      - PARSE FAILURE: we cannot locate scripts, so PREPEND the data <script> to
        the very front of the document — this still guarantees window.CHART_DATA
        is defined before anything executes. We do NOT use the <head> search
        here, since that could place data AFTER a script that precedes <head>.
      - CLEAN parse with NO <script>: ordering is irrelevant, so use the tidy
        just-after <head> / <html> / <!DOCTYPE html> placement, then prepend.
    """
    idx = _first_executable_script_index(html)
    if idx is _PARSE_FAILED:
        return script + html
    if idx is not None:
        return html[:idx] + script + html[idx:]
    for pattern in (r"<head[^>]*>", r"<html[^>]*>", r"<!DOCTYPE html>"):
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            end = m.end()
            return html[:end] + script + html[end:]
    return script + html


# Matches the data block this node injects. The payload can never contain a
# literal "</script>" (escaped by _embed_json), so the first closer is ours.
_CHART_DATA_RE = re.compile(
    r"<script>\s*window\.CHART_DATA\s*=.*?</script>",
    re.IGNORECASE | re.DOTALL,
)


def _build_data_script(data: dict) -> str:
    return f"<script>window.CHART_DATA = {_embed_json(data)};</script>"


def _inject_chart_data(html: str, data: dict) -> str:
    """Embed the FULL columns+rows payload as a window.CHART_DATA global (no row cap)."""
    return _insert_head_script(html, _build_data_script(data))


def _split_chart_data(html: str) -> tuple[str | None, str]:
    """
    Remove EVERY window.CHART_DATA <script> block from HTML.

    Returns (first_data_script or None, html_without_any_data_script). Stripping
    ALL matches (not just the first) is what guarantees idempotency: if the LLM
    emits its own stray window.CHART_DATA block alongside ours, leaving even one
    behind would let it execute later and OVERWRITE the global with truncated
    data. After this + a single re-injection, exactly one block (ours) remains.

    The first match is returned so the refine path can preserve the ORIGINAL
    full data instead of re-typing it through the LLM.
    """
    matches = list(_CHART_DATA_RE.finditer(html))
    if not matches:
        return None, html
    first = matches[0].group(0)
    cleaned = _CHART_DATA_RE.sub("", html)
    return first, cleaned


def _strip_fences(content) -> str:
    if isinstance(content, list):
        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    html = content.strip()
    if html.startswith("```html"):
        html = html[7:]
    if html.startswith("```"):
        html = html[3:]
    if html.endswith("```"):
        html = html[:-3]
    return html.strip()


async def _generate_chart(sub_question: str, chart_hint: str, genie_result: dict, cfg: AppConfig, index: int, total: int, emit, token: str) -> str:
    llm = ChatOpenAI(base_url=cfg.ai_gateway_url, api_key=token, model=cfg.claude_model)

    # Full structured Genie result. The LLM only sees the schema + a tiny sample
    # (to pick chart type / axis mapping); the ACTUAL data is injected below so it
    # never gets re-typed and truncated in LLM output.
    columns = genie_result.get("columns", [])
    rows = genie_result.get("rows", [])
    sample_rows = rows[:5]
    schema_desc = ", ".join(
        f"{c['name']} ({c['type']})" if isinstance(c, dict) else str(c) for c in columns
    )
    data_summary = (
        f"Columns ({len(columns)}): {schema_desc}\n"
        f"Total rows: {len(rows)}\n"
        f"Sample rows (first {len(sample_rows)} of {len(rows)} — SAMPLE ONLY, "
        f"read the full data from window.CHART_DATA): {sample_rows}"
    )

    user_msg = f"Question: {sub_question}\nChart hint: {chart_hint}\n\nData:\n{data_summary}"

    with mlflow.start_span(name=f"visualizer_chart_{index}") as span:
        span.set_inputs({"question": sub_question, "chart_hint": chart_hint, "row_count": len(rows)})
        async with asyncio.timeout(30):
            response = await llm.ainvoke([
                {"role": "system", "content": CHART_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
        html = _strip_fences(response.content)
        # Defensive: drop any data block the LLM emitted, then inject the real one.
        _, html = _split_chart_data(html)
        html = _inject_chart_data(html, {"columns": columns, "rows": rows})
        span.set_outputs({"html_length": len(html), "row_count": len(rows)})

    emit({"type": "chart", "html": html, "index": index, "total": total})
    return html


async def visualizer_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]
    mode = state.get("mode", "chat")

    token = get_databricks_token(cfg)

    if mode == "refine":
        llm = ChatOpenAI(base_url=cfg.ai_gateway_url, api_key=token, model=cfg.claude_model)
        # Pull the injected data block out so the LLM refines only the scaffold and
        # never has to re-type (and thus truncate) the data; re-inject it afterward.
        existing_html = state["existing_html"] or ""
        data_script, scaffold = _split_chart_data(existing_html)
        async with asyncio.timeout(30):
            response = await llm.ainvoke([
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Instruction: {state['refine_instruction']}\n\nExisting HTML:\n{scaffold}"},
            ])
        html = _strip_fences(response.content)
        if data_script is not None:
            # Preserve the ORIGINAL full data (never round-tripped through the LLM).
            _, html = _split_chart_data(html)
            html = _insert_head_script(html, data_script)
        emit({"type": "chart", "html": html, "index": 0, "total": 1})
        return {"chart_htmls": [html]}

    if mode == "ask_about_viz":
        llm = ChatOpenAI(base_url=cfg.ai_gateway_url, api_key=token, model=cfg.claude_model)
        user_msg = f"Question: {state['question']}\n\nSQL: {state.get('existing_sql', '')}\n\nChart HTML (contains data):\n{state['existing_html'][:3000]}"
        async with asyncio.timeout(30):
            response = await llm.ainvoke([
                {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
        content = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        insight = content.strip()
        emit({"type": "insight", "text": insight})
        return {"insight_text": insight}

    # mode == "chat": parallel chart generation
    genie_results = state.get("genie_results", [])
    sub_questions = state.get("sub_questions", [])
    chart_hints = state.get("chart_hints", [])
    total = len(sub_questions)

    tasks = [
        _generate_chart(sub_questions[i], chart_hints[i] if i < len(chart_hints) else "auto", genie_results[i] if i < len(genie_results) else {}, cfg, i, total, emit, token)
        for i in range(total)
    ]
    htmls = await asyncio.gather(*tasks, return_exceptions=True)

    chart_htmls = []
    for h in htmls:
        if isinstance(h, Exception):
            chart_htmls.append(f"<h3>Chart Error</h3><p>{str(h)}</p>")
        else:
            chart_htmls.append(h)

    return {"chart_htmls": chart_htmls}
