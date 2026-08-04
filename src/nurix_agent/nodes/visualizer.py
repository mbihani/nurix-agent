import asyncio
import json
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

CHART_SYSTEM_PROMPT = VISUALIZATION_GUIDE + """
Generate a SINGLE self-contained HTML file with:
- One H3 heading from the question (no other text)
- One Chart.js chart
- NO narrative paragraphs, NO analysis text, NO "Here is" phrases
- Inline <meta http-equiv="Content-Security-Policy" content="connect-src 'none'">
- Databricks brand colors: primary #FF3621, blue #2272B4, series palette [#2272B4, #FF8C00, #00A36C, #9467BD, #E15759, #76B7B2]
- window.global = window polyfill not needed (no Plotly)
"""

REFINE_SYSTEM_PROMPT = """
You are a chart refinement assistant. The user has an existing Chart.js HTML visualization and wants to modify it.
Apply the instruction to the existing HTML and return the complete updated HTML.
Preserve the H3 heading unless the instruction changes the topic.
Do NOT add narrative paragraphs. Return ONLY the complete HTML, no markdown.
"""

INSIGHT_SYSTEM_PROMPT = """
You are a data analyst. The user has a visualization based on customer feedback data and wants deeper insight.
Given the chart HTML (which contains the data) and the original SQL, answer their question in 2-4 concise sentences.
Be specific about numbers and trends visible in the data. Do not generate a new chart.
"""


async def _generate_chart(sub_question: str, chart_hint: str, genie_result: dict, cfg: AppConfig, index: int, total: int, emit, token: str) -> str:
    llm = ChatOpenAI(base_url=cfg.ai_gateway_url, api_key=token, model=cfg.claude_model)

    # Build data summary for Claude
    columns = genie_result.get("columns", [])
    rows = genie_result.get("rows", [])[:50]  # limit rows
    col_names = [c["name"] if isinstance(c, dict) else str(c) for c in columns]
    data_summary = f"Columns: {col_names}\nRows (first 50): {rows}"

    user_msg = f"Question: {sub_question}\nChart hint: {chart_hint}\n\nData:\n{data_summary}"

    with mlflow.start_span(name=f"visualizer_chart_{index}") as span:
        span.set_inputs({"question": sub_question, "chart_hint": chart_hint})
        async with asyncio.timeout(30):
            response = await llm.ainvoke([
                {"role": "system", "content": CHART_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
        content = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        # Strip markdown fences
        html = content.strip()
        if html.startswith("```html"):
            html = html[7:]
        if html.startswith("```"):
            html = html[3:]
        if html.endswith("```"):
            html = html[:-3]
        html = html.strip()
        span.set_outputs({"html_length": len(html)})

    emit({"type": "chart", "html": html, "index": index, "total": total})
    return html


async def visualizer_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg: AppConfig = config["configurable"]["app_config"]
    emit = state["emit"]
    mode = state.get("mode", "chat")

    token = get_databricks_token(cfg)

    if mode == "refine":
        llm = ChatOpenAI(base_url=cfg.ai_gateway_url, api_key=token, model=cfg.claude_model)
        async with asyncio.timeout(30):
            response = await llm.ainvoke([
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Instruction: {state['refine_instruction']}\n\nExisting HTML:\n{state['existing_html']}"},
            ])
        content = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        html = content.strip().lstrip("```html").lstrip("```").rstrip("```").strip()
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
