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
    "\nCOLOR PALETTE — categorical series, use in EXACTLY this order:\n"
    "#0891B2, #d95926, #199e70, #c98500, #d55181, #9085e9, #e66767\n"
    "- These 7 are the ONLY series colors. They were validated against the dark\n"
    "  surface for lightness, >=3:1 contrast, and perceptual separation between\n"
    "  adjacent slots for normal vision AND common color-vision deficiencies.\n"
    "- Use them in the order given. Do NOT reorder, do NOT substitute other hues,\n"
    "  and do NOT add an 8th color — wrap around if you somehow need more.\n"
    "- There is deliberately NO BLUE in this ramp. Blue was EXCLUDED because it is\n"
    "  not distinguishable from the cyan slot. NEVER add a blue series color.\n"
    "- Semantic colors — use ONLY when the data is genuinely semantic (e.g. sentiment\n"
    "  positive vs negative): positive/good #199e70, negative/error #e66767.\n"
    "  Never use these as generic series colors for non-semantic categories.\n"
    "- #22D3EE (bright cyan) is reserved for UI CHROME in the app shell (buttons,\n"
    "  focus rings). NEVER use #22D3EE for a data mark — it is outside the validated\n"
    "  lightness band, so adjacent values stop being comparable. Data marks use #0891B2.\n"
    "\nDARK THEME — MANDATORY, NOT OPTIONAL:\n"
    "The chart is embedded in a dark dashboard card. Chart.js DEFAULTS to dark-text-on-\n"
    "light, so if you leave any of the options below unset the chart WILL render light on\n"
    "a white box and be unusable. You MUST set every one of them explicitly.\n"
    "- Page/body background: #020617. The card fill AND the outer canvas are both\n"
    "  #020617, so matching it makes the chart look seamless and borderless. Do NOT\n"
    "  use #0F172A (that is the OLD card color and is now wrong), do NOT use #1B1B1B,\n"
    "  and NEVER leave the body white.\n"
    "- Chart.js canvas background: transparent — let the body color show through.\n"
    "- Set ALL of these Chart.js config paths explicitly:\n"
    "    options.plugins.legend.labels.color = '#94A3B8'\n"
    "    options.scales.x.ticks.color        = '#94A3B8'\n"
    "    options.scales.y.ticks.color        = '#94A3B8'\n"
    "    options.scales.x.grid.color         = '#1E293B'\n"
    "    options.scales.y.grid.color         = '#1E293B'\n"
    "    options.scales.x.title.color        = '#94A3B8'   (whenever an axis title shows)\n"
    "    options.scales.y.title.color        = '#94A3B8'\n"
    "- Gridlines: #1E293B, thin (lineWidth 1).\n"
    "- Every other piece of in-chart text (axis titles, data labels, annotations,\n"
    "  tooltips): #94A3B8. ONLY the single headline number of a KPI/counter may be #FFFFFF.\n"
    "- TABLES are a legitimate output and must ALSO be dark: table/cell background\n"
    "  #020617, body text #E2E8F0, header-row text #94A3B8, row borders 1px #1E293B.\n"
    "  NEVER a white table background and never dark text on a light table.\n"
    "\nSIZING — do not fight the app's fitting layer:\n"
    "- The app injects CSS and forces maintainAspectRatio: false so the chart scales to\n"
    "  its card with no overflow. Do NOT set maintainAspectRatio: true.\n"
    "- Do NOT set a fixed pixel width or height on the canvas, and do NOT put a\n"
    "  height= or width= attribute on the <canvas> element.\n"
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
    "- Use ONLY the 7-color categorical ramp above for series, in that order\n"
    "- Apply the dark theme rules above in full: body #020617, transparent canvas,\n"
    "  #94A3B8 tick/legend/title text, #1E293B gridlines\n"
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
- Series palette — ONLY these 7, in this order, no blue, no substitutions:
  [#0891B2, #d95926, #199e70, #c98500, #d55181, #9085e9, #e66767]
- DARK THEME IS REQUIRED: body background #020617, transparent chart canvas,
  legend/tick/axis-title text #94A3B8, gridlines #1E293B. Set
  options.plugins.legend.labels.color, options.scales.*.ticks.color and
  options.scales.*.grid.color EXPLICITLY — Chart.js defaults render dark-on-light
  and would produce a white chart on the dark dashboard.
- Do NOT use #0F172A anywhere: it is the OLD card color and no longer matches the
  dashboard. The surface is #020617.
- Tables must be dark too: background #020617, text #E2E8F0, header text #94A3B8,
  borders #1E293B.
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

DARK THEME — PRESERVE IT:
- The chart lives in a dark dashboard card. Keep the dark theme intact unless the
  instruction explicitly asks to change colors: body background #020617, transparent
  chart canvas, legend/tick/axis-title text #94A3B8, gridlines #1E293B, tables on
  #020617 with #E2E8F0 text. Never return a chart with a white/light background.
- Do NOT use #0F172A: it is the OLD card color and no longer matches the dashboard.
  If the HTML you are given still paints #0F172A, REPLACE it with #020617.
- Series colors come ONLY from [#0891B2, #d95926, #199e70, #c98500, #d55181,
  #9085e9, #e66767]. There is deliberately no blue — never introduce one.
- Keep options.plugins.legend.labels.color, options.scales.*.ticks.color and
  options.scales.*.grid.color set; dropping them makes Chart.js render light again.

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
    chart script runs. To be correct on every stdlib version we track which
    RCDATA elements are open ourselves and only record a <script> position while
    none are. This is a no-op on versions where the parser already suppresses the
    inner <script>, and the needed guard on versions where it does not.

    A SELF-CLOSING `<textarea/>` / `<title/>` is a second, sharper mismatch:
    stdlib routes it to handle_startendtag, whose default fires handle_starttag
    THEN handle_endtag — so the element would open and immediately close. But
    neither tag is a void element in HTML, so a browser leaves `<textarea/>`
    OPEN and treats everything after it as inert text. We therefore run only the
    start-side logic for these tags and never let the paired end-callback close
    them: the element stays open for the rest of the parse, every later <script>
    is treated as inert (matching the browser), and `pos` stays None. The
    unterminated state is reported via `unterminated_rcdata` so the caller can
    choose the conservative PREPEND fallback rather than a <head> placement that
    could land after a script we deliberately ignored.

    Open RCDATA elements are tracked as a STACK OF TAG NAMES rather than a bare
    depth counter: a counter desyncs on malformed interleaving (a stray
    `</textarea>` while only <title> is open would wrongly decrement it), while
    matching by name closes exactly the element that was opened.
    """

    _RCDATA_TAGS = ("textarea", "title")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pos: tuple[int, int] | None = None  # (line, col) of first <script>
        self._rcdata_open: list[str] = []  # names of currently-open RCDATA elements

    @property
    def unterminated_rcdata(self) -> bool:
        """True if the parse ended while still inside an RCDATA element."""
        return bool(self._rcdata_open)

    def handle_starttag(self, tag, attrs):
        if tag in self._RCDATA_TAGS:
            self._rcdata_open.append(tag)
            return
        if self.pos is None and tag == "script" and not self._rcdata_open:
            self.pos = self.getpos()

    def handle_endtag(self, tag):
        if tag in self._RCDATA_TAGS and tag in self._rcdata_open:
            # Close the matching element (and anything nested inside it).
            del self._rcdata_open[self._rcdata_open.index(tag):]

    def handle_startendtag(self, tag, attrs):
        # <textarea/> and <title/> are NOT void elements: a browser leaves them
        # OPEN. Run only the start-side logic so the paired end-callback that
        # HTMLParser's default would fire can never close them.
        if tag in self._RCDATA_TAGS:
            self.handle_starttag(tag, attrs)
            return
        super().handle_startendtag(tag, attrs)


def _linecol_to_index(html: str, line: int, col: int) -> int:
    """Convert HTMLParser's 1-based line / 0-based col to an absolute string index."""
    lines = html.split("\n")
    # Sum full prior lines plus their stripped '\n' separators, then add the column.
    idx = sum(len(lines[i]) + 1 for i in range(line - 1)) + col
    return min(idx, len(html))


# Sentinel distinguishing an UNLOCATABLE first script from a clean parse that
# found no <script>. The two demand opposite fallbacks: when we cannot trust any
# location (parser raised, or the document ended inside an unterminated RCDATA
# element that renders later scripts inert), the ONLY way to guarantee
# window.CHART_DATA is defined before anything executes is to PREPEND; a clean
# no-script result means ordering is irrelevant and tidy <head> placement is fine.
_PARSE_FAILED = object()


def _first_executable_script_index(html: str):
    """
    Locate the first EXECUTABLE <script> start tag.

    Returns one of three distinct signals:
      - int           : absolute index of the first executable <script>.
      - None          : CLEAN parse that found NO executable <script>.
      - _PARSE_FAILED : the parser raised on malformed input (location unknown),
                        OR the parse ended still inside an unterminated RCDATA
                        element having recorded no position — see below.

    Uses an HTML-context-aware parse so a `<script` literal inside a comment or a
    RCDATA element (<textarea>, <title>) is ignored rather than false-matching.

    An unterminated RCDATA element (e.g. a self-closing `<textarea/>`, which a
    browser leaves OPEN) is NOT the same as a clean no-script document: there may
    genuinely be a later <script> that we deliberately ignored as inert. Reporting
    None would route to the tidy <head> fallback, which could place the data block
    AFTER that script. So we report _PARSE_FAILED to force the PREPEND fallback,
    which guarantees window.CHART_DATA is defined before anything executes.
    """
    finder = _FirstScriptFinder()
    try:
        finder.feed(html)
        finder.close()
    except Exception:
        return _PARSE_FAILED
    if finder.pos is None:
        if finder.unterminated_rcdata:
            return _PARSE_FAILED
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
      - UNLOCATABLE first script (parser raised, or the document ended inside an
        unterminated RCDATA element such as a self-closing `<textarea/>` that
        makes every later script inert): PREPEND the data <script> to the very
        front of the document — this still guarantees window.CHART_DATA is defined
        before anything executes. We do NOT use the <head> search here, since that
        could place data AFTER a script we could not or deliberately did not use.
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


def _chart_event(html: str, *, index: int, total: int, sql: str | None) -> dict:
    """
    Build THE chart SSE event — the single construction site for every emitter.

    Centralized deliberately: the refine path used to build its own chart event by
    hand and so missed `chart_index`/`chart_total` entirely when they were added
    here. One builder means that class of drift cannot recur.

    `chart_index`/`chart_total` are the names the nurix-nlviz consumer reads (its
    multi-chart branch is `typeof event.chart_total === 'number' && > 1`).
    `index`/`total` are kept as an INTENTIONAL COMPATIBILITY ALIAS so the already
    deployed nurix-nlviz proxy, which passes events straight through, keeps working
    — no flag-day where one app is redeployed and the other is not. Do not remove
    either pair without redeploying both apps.

    `sql` is a REQUIRED argument (no default) so no caller can silently forget it.
    When it is genuinely absent the key is OMITTED rather than set to "": an empty
    string masquerades as a query the consumer could pin or refine, whereas a
    missing key is an honest "no SQL for this chart". Callers are expected to
    surface that absence rather than pass it along quietly.
    """
    event = {
        "type": "chart",
        "html": html,
        "chart_index": index,
        "chart_total": total,
        "index": index,
        "total": total,
    }
    if sql and sql.strip():
        event["sql"] = sql
    return event


async def _generate_chart(sub_question: str, chart_hint: str, genie_result: dict, cfg: AppConfig, index: int, token: str, on_done=None) -> str:
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

    # Deliberately does NOT emit the chart event. Indices can only be assigned once
    # the full set of SUCCESSES is known (see visualizer_node), so emission happens
    # there. `on_done` reports completion so the caller can keep progress visible
    # while the remaining charts are still generating.
    if on_done is not None:
        on_done()
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
        # Built by the shared helper so this path can never again drift out of sync
        # with the chart-event shape (it previously emitted only index/total).
        emit(_chart_event(html, index=0, total=1, sql=state.get("existing_sql")))
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

    # mode == "chat": parallel chart generation.
    #
    # Generation and EMISSION are deliberately separated. Charts used to be emitted
    # from inside each task as it finished, with `total` fixed to the CANDIDATE count
    # — so a single generation failure (gather(return_exceptions=True) turns it into a
    # non-event) left a GAP in the emitted indices while `chart_total` still claimed
    # the full count. The consumer writes charts into an array by index, so it would
    # wait forever for a chart that was never coming, and sql/chart pairing desynced.
    #
    # Instead: generate everything, keep only the SUCCESSES in their original
    # relative order, and only THEN assign dense indices 0..N-1 with
    # chart_total == number of successes. Emission therefore batches at the end;
    # `on_done` keeps progress visible while generation is still in flight.
    genie_results = state.get("genie_results", [])
    sub_questions = state.get("sub_questions", [])
    chart_hints = state.get("chart_hints", [])
    candidate_count = len(sub_questions)

    # Who owns the `sql` event depends on the path. The PLAIN path's genie_node already
    # emits one per sub-question as it completes (early, useful feedback), so emitting
    # again here would duplicate it and change that path's event shape. Deep research
    # deliberately defers its `sql` events to here so their indices match the charts
    # that actually rendered. Either way the chart event carries its own `sql`, so
    # sql -> chart pairing never depends on which path emitted the sql event.
    emit_sql = bool(state.get("deep_research"))

    completed = 0

    def _note_done() -> None:
        # Perceived-progress heartbeat: charts no longer stream out individually, so
        # without this the UI would sit silent for the whole generation window.
        nonlocal completed
        completed += 1
        emit({
            "type": "thinking",
            "text": f"Rendered chart {completed} of {candidate_count}...",
        })

    tasks = [
        _generate_chart(
            sub_questions[i],
            chart_hints[i] if i < len(chart_hints) else "auto",
            genie_results[i] if i < len(genie_results) else {},
            cfg, i, token, _note_done,
        )
        for i in range(candidate_count)
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # Keep successes in ORIGINAL order; a failure is reported, never silently skipped.
    survivors: list[tuple[int, str]] = []
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, BaseException):
            label = sub_questions[i] if i < len(sub_questions) else f"chart {i}"
            emit({
                "type": "thinking",
                "text": f"Could not render a chart for '{label[:80]}': "
                        f"{type(outcome).__name__}: {str(outcome)[:200]}",
            })
            continue
        survivors.append((i, outcome))

    if len(survivors) < candidate_count:
        emit({
            "type": "thinking",
            "text": f"{candidate_count} chart{'s' if candidate_count != 1 else ''} "
                    f"attempted; {len(survivors)} rendered successfully.",
        })

    # SECOND reducing pass, and it MUST complete before `total` is computed.
    #
    # A deep-research candidate reaching this point was admitted with a non-empty SQL
    # upstream, so an empty one means that invariant broke. Emitting the chart anyway
    # would ship an empty `sql` masquerading as a pinnable query; emitting a
    # `{"sql": ""}` event would do the same. So the entry is DROPPED and reported.
    #
    # Dropping here removes a survivor, so numbering has to happen after this pass —
    # computing `total` before it would reintroduce the gap-plus-inflated-total bug
    # from the other direction.
    emittable: list[tuple[int, str]] = []
    for orig_index, html in survivors:
        result = genie_results[orig_index] if orig_index < len(genie_results) else {}
        sql = (result.get("sql") or "").strip()
        if emit_sql and not sql:
            label = sub_questions[orig_index] if orig_index < len(sub_questions) else f"chart {orig_index}"
            emit({
                "type": "thinking",
                "text": f"Not charting '{label[:80]}': the SQL that produced it is "
                        f"missing, so the chart could not be pinned or refined.",
            })
            continue
        emittable.append((orig_index, html))

    # Numbering is assigned only now, over the entries actually being emitted.
    total = len(emittable)
    chart_htmls: list[str] = []
    for new_index, (orig_index, html) in enumerate(emittable):
        result = genie_results[orig_index] if orig_index < len(genie_results) else {}
        sql = result.get("sql") or ""
        if emit_sql:
            # Deferred from the deep-research node so this index matches the chart
            # that actually rendered, letting a consumer pair sql -> chart.
            emit({"type": "sql", "sql": sql, "chart_index": new_index, "index": new_index})
        emit(_chart_event(html, index=new_index, total=total, sql=sql))
        chart_htmls.append(html)

    return {"chart_htmls": chart_htmls}
