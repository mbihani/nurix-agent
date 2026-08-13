import asyncio
import json
import logging
import re
from html.parser import HTMLParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import mlflow
from .. import tracing
from ..config import AppConfig, get_databricks_token
from ..narrative import clean_genie_narrative
from ..state import AgentState

logger = logging.getLogger(__name__)

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
    "    options.scales.x.grid.color         = '#111A2B'\n"
    "    options.scales.y.grid.color         = '#111A2B'\n"
    "    options.scales.x.grid.borderColor   = '#111A2B'\n"
    "    options.scales.y.grid.borderColor   = '#111A2B'\n"
    "    options.scales.x.border.color       = '#111A2B'\n"
    "    options.scales.y.border.color       = '#111A2B'\n"
    "    options.scales.x.title.color        = '#94A3B8'   (whenever an axis title shows)\n"
    "    options.scales.y.title.color        = '#94A3B8'\n"
    "- Gridlines, tick lines, and scale borders: #111A2B, thin (lineWidth 1).\n"
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
    "- For doughnut/pie charts, keep the arc centred and vertically balanced under\n"
    "  maintainAspectRatio: false: set options.plugins.legend.position = 'bottom',\n"
    "  options.plugins.legend.align = 'center', options.plugins.legend.fullSize = false,\n"
    "  options.plugins.legend.maxHeight = 32, and options.layout.padding =\n"
    "  {top: 40, right: 8, bottom: 8, left: 8}. The extra 32px of top padding\n"
    "  counterbalances the bounded bottom legend; equal side padding centres the arc.\n"
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
    "  #94A3B8 tick/legend/title text, #111A2B gridlines/scale borders\n"
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
  legend/tick/axis-title text #94A3B8, gridlines/scale borders #111A2B. Set
  options.plugins.legend.labels.color, options.scales.*.ticks.color and
  options.scales.x.grid.color, options.scales.y.grid.color,
  options.scales.x.grid.borderColor, options.scales.y.grid.borderColor,
  options.scales.x.border.color, and options.scales.y.border.color EXPLICITLY —
  Chart.js defaults render dark-on-light
  and would produce a white chart on the dark dashboard.
- For doughnut/pie charts under the client's forced maintainAspectRatio: false,
  set options.plugins.legend.position = 'bottom', align = 'center', fullSize = false,
  maxHeight = 32, and options.layout.padding = {top: 40, right: 8, bottom: 8, left: 8}.
  This keeps the arc horizontally centred and counterbalances the bottom legend.
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
  chart canvas, legend/tick/axis-title text #94A3B8, gridlines/scale borders #111A2B, tables on
  #020617 with #E2E8F0 text. Never return a chart with a white/light background.
- Do NOT use #0F172A: it is the OLD card color and no longer matches the dashboard.
  If the HTML you are given still paints #0F172A, REPLACE it with #020617.
- Series colors come ONLY from [#0891B2, #d95926, #199e70, #c98500, #d55181,
  #9085e9, #e66767]. There is deliberately no blue — never introduce one.
- Keep options.plugins.legend.labels.color, options.scales.*.ticks.color,
  options.scales.x.grid.color, options.scales.y.grid.color,
  options.scales.x.grid.borderColor, options.scales.y.grid.borderColor,
  options.scales.x.border.color, and options.scales.y.border.color set to the colors
  above; dropping them makes Chart.js render light again.
- For doughnut/pie charts, preserve bottom-centred legend settings (position = 'bottom',
  align = 'center', fullSize = false, maxHeight = 32) and options.layout.padding =
  {top: 40, right: 8, bottom: 8, left: 8}; never set maintainAspectRatio: true.

Do NOT add narrative paragraphs. Return ONLY the complete HTML, no markdown.
"""

# The GROUNDED prompt: a fresh Genie query ran and its result is the ONLY fact source.
#
# WHY CLAUDE IS STILL IN THE LOOP AT ALL. Genie's own narrative was considered as the
# answer, with no LLM at all — simpler, and impossible to hallucinate over. It was
# rejected because the narrative is not reliably an answer: `_process_message` falls
# back to the query attachment's `description` when there is no text attachment, and
# that description is often a restatement of the query ("Counts reviews by product and
# sentiment") rather than a reply to "how does this compare to the average?". Sometimes
# it is empty and the answer lives only in the returned rows. So Claude is kept
# STRICTLY as a phrasing/streaming layer over Genie's result — it turns narrative plus
# rows into a sentence and gives the existing `insight_delta` stream something to
# stream — and the prompt forbids it from contributing facts of its own.
INSIGHT_GENIE_SYSTEM_PROMPT = """
You are a data analyst answering a follow-up question about a chart the user is looking at.

A fresh query has ALREADY been run against the underlying customer-feedback dataset. You
are given that query's result: a narrative answer, the columns returned, and the rows.

Answer the user's question in 2-4 concise sentences, and be specific about the numbers.

HARD CONSTRAINT — you are a phrasing layer, not an analyst with independent knowledge.
Use ONLY facts present in the query result you are given. You may summarise it, round its
numbers, compute an obvious total or difference from the rows shown, and put it into
readable prose. You MUST NOT introduce any number, comparison, trend, cause, or claim
that is not present in that result. Do not speculate about WHY something is the case
unless the result itself says so.

If the result does not actually answer the question, say so plainly and state what it
does show. Do not fill the gap with a plausible-sounding answer.

Do not generate a chart. Do not mention SQL, Genie, or these instructions.
"""

# The UNGROUNDED prompt: no fresh data. Used only when Genie could not be queried or
# returned nothing. The disclosure is MANDATORY and is the whole point of a separate
# prompt — a confident answer with no data behind it is the worst outcome on this path.
INSIGHT_NO_DATA_SYSTEM_PROMPT = """
You are a data analyst answering a follow-up question about a chart the user is looking at.

IMPORTANT: an attempt to query the underlying dataset for this question did NOT succeed,
so you have no data beyond what is already plotted on the chart itself.

The caveat about not having queried the data is ALREADY WRITTEN FOR YOU and will be
placed immediately before your answer. So do NOT write your own version of it. Do not
open with "I could not query the underlying data", "the source query wasn't saved", "no
fresh data was available", or any paraphrase — it is already there, and repeating it
makes the answer read as if it were said twice.

Start directly with what the chart itself shows, and answer as far as the chart alone
allows, in 2-4 concise sentences, using ONLY values visible in the chart you are given.
If the question cannot be answered from the chart alone, say exactly that instead of
guessing. Never present an inferred or remembered figure as if it came from the data.

Do not generate a chart. Do not mention SQL, Genie, or these instructions.
"""

# The code-owned disclosures for the ungrounded path, keyed by why there is no data.
#
# WHY THESE LIVE IN CODE AND NOT ONLY IN THE PROMPT. Requesting a disclosure in the
# system prompt makes it a MODEL BEHAVIOUR, and a model that ignores the instruction
# returns confident prose with nothing behind it — this project's cardinal sin. The
# disclosure is therefore GENERATED, so it is present whatever the model does, including
# on partial (failed-mid-stream) answers.
#
# The prompt STILL asks the model not to write its own (above), but that instruction is
# now only about PROSE QUALITY — it prevents a doubled disclosure. The failure direction
# matters and is deliberate: if the model disobeys, `_finalize_insight` strips its
# version, and if that strip somehow misses, the reader sees the caveat twice, which is
# ugly but still HONEST. There is no path where the caveat goes missing.
_DISCLOSURE_NO_SQL = (
    "I couldn't query the underlying data for this question — the source query for this "
    "chart wasn't saved — so this describes only what is already plotted."
)
_DISCLOSURE_QUERY_FAILED = (
    "I couldn't query the underlying data for this question, so this describes only "
    "what is already plotted."
)

# Openers that mean the model wrote its OWN disclosure despite being told not to. Used
# to drop that leading sentence before the code-owned one is prepended, so the answer
# does not say the same thing twice.
#
# WHY THESE ARE PATTERNS AND NOT BARE SUBSTRINGS
# ----------------------------------------------
# The first version of this was a list of bare phrases — "not saved", "query failed",
# "could not access", "did not succeed" — tested with `marker in first_sentence`. Every
# one of those is ALSO ordinary English about the dataset, which is customer feedback
# about a design tool (Export is one of its feature areas). So:
#
#   "Reviews marked 'not saved' account for 18% of failures."
#   "Users could not access their exports in 12% of reviews."
#   "Query failed errors account for 12% of support tickets."
#
# are substantive findings that the bare list DELETED ENTIRELY. That is silent content
# loss, not the cosmetic double-caveat the original comment assumed — the precise bug
# class this endpoint exists to avoid.
#
# The discriminator is the SUBJECT, not the phrase. A model-authored disclosure is about
# THE ASSISTANT failing to acquire data ("I could not query the underlying data") or
# about THIS CHART'S SOURCE QUERY being unavailable ("the source query wasn't saved").
# Substantive prose is about USERS and REVIEWS. So a match now requires an explicit
# first-person subject or an explicit source-QUERY reference.
#
# THE FAILURE DIRECTION IS DELIBERATELY ASYMMETRIC, so these patterns UNDER-match on
# purpose. The code-owned disclosure is prepended either way, so:
#   * a MISS  -> the reader sees the caveat twice. Ugly, still honest.
#   * a FALSE POSITIVE -> real analysis is deleted and nobody can tell.
# Only the second is a correctness bug, so anything ambiguous is left unmatched. Phrases
# that cannot carry a subject constraint ("no fresh data", "did not succeed", "query
# failed" standing alone) were DROPPED rather than tightened: a genuine disclosure of a
# failed query is written in the first person and is still caught by the first pattern.
_MODEL_DISCLOSURE_PATTERNS = (
    # (1) FIRST-PERSON inability to ACQUIRE data: "I could not query the underlying
    #     data", "I'm unable to access the data", "we couldn't retrieve fresh data".
    #     The verb must be an ACQUISITION verb. Deliberately NOT "determine",
    #     "calculate" or "answer", because "I cannot determine the average rating from
    #     this chart alone" is a legitimate, substantive sentence that the ungrounded
    #     prompt explicitly ASKS the model to write. Stripping it would lose the one
    #     thing the reader most needs to know.
    re.compile(
        r"\b(?:i|we)\b[^.!?]{0,40}?"
        r"(?:could\s?n[o']?t|can\s?n[o']?t|cannot|can not|unable to|"
        r"was\s?n[o']?t able to|were\s?n[o']?t able to|did\s?n[o']?t)"
        r"[^.!?]{0,30}?"
        r"\b(?:quer(?:y|ied)|access|retrieve|fetch|pull|obtain|run)\b",
        re.I,
    ),
    # (2) THIS CHART'S SOURCE QUERY is unavailable: "the source query for this chart
    #     wasn't saved", "the chart's underlying query was not stored". An explicit
    #     QUERY reference is REQUIRED, which is what keeps prose about reviews "marked
    #     'not saved'" untouched — that sentence contains no query at all.
    re.compile(
        r"\b(?:source|chart|original|originating|underlying|generated|stored)\s+"
        r"quer(?:y|ies)\b[^.!?]{0,60}?"
        r"(?:was\s?n[o']?t|is\s?n[o']?t|not)\s+"
        r"(?:saved|stored|available|kept|recorded)",
        re.I,
    ),
    # (3) The same fact stated the other way round: "no query was saved for this chart".
    re.compile(
        r"\bno\s+(?:source\s+|chart\s+|underlying\s+)?quer(?:y|ies)\s+"
        r"(?:was\s+|were\s+)?(?:saved|stored|available|recorded)\b",
        re.I,
    ),
)


def _log_cleanup_failure(message: str) -> None:
    """
    Log a cleanup failure with NO possibility of raising.

    Called only from the streaming failure handler, in the window between a cleanup step
    failing and the bare `raise` that re-raises the ORIGINAL exception. Everything in
    that window must be incapable of raising, or the caller observes the wrong error:
    a `logger.warning` that raised would surface the LOGGING failure instead of the real
    cause, and could replace an in-flight `CancelledError` — silently breaking
    cooperative cancellation, which is worse than the bug the handler exists to fix.

    In practice `logging` already swallows handler errors (`Handler.handleError`), so the
    trigger is narrow: a monkeypatched logger or a genuinely broken logging setup. This
    exists anyway because the guarantee stated at the `raise` should be TRUE rather than
    usually true — the same correction applied to the terminal-insight guarantee, which
    now names its one exclusion instead of overstating itself. A guarantee with an
    executable statement sitting inside it is not a guarantee.

    The log is KEPT rather than dropped: losing the diagnostic for an unreportable
    cleanup failure would be the worse trade.
    """
    try:
        logger.warning(message, exc_info=True)
    except BaseException:
        # Nothing left to report through, and re-raising here would defeat the whole
        # point. `pass` is the only correct action.
        pass


def _strip_model_disclosure(text: str) -> str:
    """
    Drop a leading model-authored disclosure sentence, if it wrote one.

    Only the FIRST sentence is considered, and only when it matches one of the
    subject-constrained patterns above. Everything else is returned untouched — the goal
    is to avoid a doubled caveat, not to police the model's prose, and a sentence that
    merely mentions saving or querying is far more likely to be a real finding about the
    feedback data than a disclosure. See the patterns for why they under-match.
    """
    stripped = text.lstrip()
    if not stripped:
        return text
    # First sentence boundary: ". " / "! " / "? " / newline, whichever comes first.
    m = re.search(r"(?<=[.!?])\s+|\n", stripped)
    first = stripped[: m.start()] if m else stripped
    rest = stripped[m.end():] if m else ""
    if any(p.search(first) for p in _MODEL_DISCLOSURE_PATTERNS):
        # If the disclosure was the ENTIRE answer, keep nothing — the code-owned
        # disclosure alone is a better answer than the same sentence twice.
        return rest.lstrip()
    return text


def _finalize_insight(text: str, disclosure: str | None) -> str:
    """
    The one place an ungrounded insight gets its disclosure. STRUCTURAL, not prompted.

    `disclosure` is None on the grounded path, where the text passes through untouched —
    a grounded answer must not acquire a caveat it does not deserve.

    Applied to PARTIAL answers too: a stream that died halfway through an ungrounded
    answer must not present its fragment as if data backed it.
    """
    if disclosure is None:
        return text
    body = _strip_model_disclosure(text or "").strip()
    return f"{disclosure} {body}".strip() if body else disclosure

# How much of Genie's tabular result is handed to the phrasing layer. Follow-up answers
# are 2-4 sentences, so a wide result adds prompt cost without changing the wording;
# these bounds keep a pathological result from blowing the budget. The row count is
# always stated truthfully alongside the sample so the model cannot mistake a truncated
# sample for the whole result and total it.
_INSIGHT_MAX_ROWS = 60
_INSIGHT_MAX_CHARS = 12000


def _format_genie_result(result: dict) -> str:
    """
    Render a Genie result as the fact source for the insight phrasing layer.

    Narrative, schema, and rows — never the chart HTML. Truncation is always LABELLED:
    a silently shortened row set invites the model to sum a partial column and present
    it as a total.
    """
    narrative = clean_genie_narrative(result.get("text") or "").strip()
    columns = result.get("columns") or []
    rows = result.get("rows") or []

    col_desc = ", ".join(
        f"{c['name']} ({c.get('type', '?')})" if isinstance(c, dict) else str(c)
        for c in columns
    ) or "(none returned)"

    shown = rows[:_INSIGHT_MAX_ROWS]
    parts = [
        f"Narrative answer from the query: {narrative or '(none returned)'}",
        f"Columns ({len(columns)}): {col_desc}",
        f"Total rows returned: {len(rows)}",
    ]
    if shown:
        label = (
            f"All {len(rows)} rows" if len(shown) == len(rows)
            else f"First {len(shown)} of {len(rows)} rows (TRUNCATED — do not treat as complete)"
        )
        parts.append(f"{label}:\n{shown}")
    else:
        parts.append("Rows: (the query returned no rows)")

    out = "\n\n".join(parts)
    if len(out) > _INSIGHT_MAX_CHARS:
        out = out[:_INSIGHT_MAX_CHARS] + "\n... [result truncated for length]"
    return out


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


def _chunk_text(content) -> str:
    """
    Flatten one streamed chunk's `content` into plain text.

    A chunk carries either a string or a list of content blocks (the same two shapes
    the non-streaming paths already normalize).

    WHY THE LIST BRANCH JOINS WITH " "
    ----------------------------------
    To preserve EXACT parity with the pre-streaming terminal text. The pre-streaming
    `ask_about_viz` branch normalized list content with `" ".join(...)`. Any different
    separator here would change the TERMINAL `insight` event for a list-shaped
    response — visible to a client that ignores deltas — which is precisely the
    additive-contract breach this module promises cannot happen.

    THE REACHABLE PATH IS STRING-ONLY, so the separator is INERT there. Measured
    against the live AI Gateway (`databricks-claude-sonnet-5` via cfg.ai_gateway_url,
    2026-08-10), the only surface this code path talks to:
      * `astream`: 59 of 59 `AIMessageChunk.content` values were plain `str`.
        ZERO list-shaped chunks. (A second run: 42 deltas / 538 chars, same shape.)
      * `ainvoke`: `.content` was `str` (642 chars), not a list.
    langchain_openai builds each streaming chunk's content from the OpenAI-shaped
    `choices[].delta.content`, which is a string field, so `str` is the structural
    outcome for this gateway rather than a lucky sample.

    But the helper explicitly ACCEPTS LangChain's supported list-shaped
    `AIMessageChunk.content`, and Responses-style content blocks, a future gateway
    normalization, or another compatible adapter could reach it. Since the branch
    cannot be reached today, `" "` costs the reachable string-only path exactly
    nothing while restoring parity if it ever is reached — strictly safer than betting
    on a hypothesis about what block semantics such a future shape would carry. Do not
    "optimize" this to `""` on the theory that blocks are always mid-sentence
    fragments: that theory is untestable here, and parity is not.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                # Ignore non-text blocks (e.g. reasoning/tool blocks) — only the
                # visible answer text belongs in a delta.
                if b.get("type") in (None, "text") and isinstance(b.get("text"), str):
                    out.append(b["text"])
            elif isinstance(b, str):
                out.append(b)
        # " " NOT "": parity with the pre-streaming `" ".join(...)` terminal text.
        # See the docstring — this branch is unreachable on the current gateway, so the
        # separator is inert in practice and parity is the only thing it can affect.
        return " ".join(out)
    return str(content)


async def _stream_text_deltas(llm, messages, on_delta) -> str:
    """
    Stream an LLM response, invoking `on_delta` per non-empty text fragment, and
    return the COMPLETE accumulated text.

    THE EVENT CONTRACT (matters to the frontend) — SUCCESS PATH:
      * Progressive fragments go out as an ADDITIVE new event type; the terminal
        event carrying the full text is still emitted by the caller, unchanged.
      * A client that ignores the delta events therefore sees byte-identical
        behaviour to before streaming existed. There is no flag day.
      * Deltas are RAW model output. The caller may post-process the complete text
        (e.g. citation stripping), so the TERMINAL event is AUTHORITATIVE and a
        client rendering deltas is expected to REPLACE its accumulated text with the
        terminal payload rather than append to it.

    THE EVENT CONTRACT — FAILURE PATH:
      * REPLACE-not-append only holds if a terminal event ALWAYS arrives. It used to
        not: when this stream raised after N deltas (30s timeout, or the LLM
        erroring), the exception propagated straight past the caller's terminal emit.
        The client was left rendering partial text with nothing authoritative to
        replace it, and the SSE generator breaks on `error`, so not even `done`
        followed. Partial model output sat on screen looking like a finished answer.
      * The fix does NOT live here. This function deliberately does not catch: the
        most likely mid-stream failure is the caller's `asyncio.timeout`, which works
        by throwing CancelledError INTO this coroutine, and swallowing that here would
        both break the timeout (it needs the CancelledError back to convert it into
        TimeoutError) and break genuine cancellation when the SSE client disconnects.
      * Instead the caller passes an `on_delta` that accumulates, wraps the call in
        try/except, and emits its terminal event from the accumulated text on the way
        out. See the `ask_about_viz` branch in visualizer_node for the authoritative
        statement of the resulting client contract.
      * The rule a client can rely on, stated with its ONE exclusion: every stream that
        emitted at least one delta is followed by a terminal event superseding those
        deltas — succeeded or failed — EXCEPT when the transport itself is what failed.
        No code can deliver an event through a broken emitter, so the guarantee is
        conditional on the emitter working, and it is stated that way on purpose. An
        overstated guarantee is worse than an honest one: a client told the terminal
        event is unconditional would never handle the case where it does not arrive.
      * The concrete instance of that exclusion: on CLIENT DISCONNECT the handler does
        try to queue the partial terminal `insight`, but the SSE generator is already in
        its `finally` by then, so the event is never delivered. That is INHERENT to
        cancellation, not a bug with a fix — the socket is gone. It is written down here
        rather than papered over. (It is also harmless: nothing is left rendering,
        because the thing that would render it is what disconnected.)

    Accumulating here as well (rather than only in the caller) keeps the returned text
    the single source of truth for the SUCCESS-path terminal event.
    """
    parts: list[str] = []
    async for chunk in llm.astream(messages):
        piece = _chunk_text(getattr(chunk, "content", None))
        if not piece:
            continue
        parts.append(piece)
        on_delta(piece)
    return "".join(parts)


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
        # This branch made an LLM call with no span; only the chat path was traced.
        with mlflow.start_span(name="visualizer_refine") as span:
            span.set_inputs({
                "instruction": tracing.truncate(state["refine_instruction"]),
                "scaffold_chars": len(scaffold),
                "had_data_script": data_script is not None,
            })
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
            span.set_outputs({"html_length": len(html), "data_preserved": data_script is not None})
        # Built by the shared helper so this path can never again drift out of sync
        # with the chart-event shape (it previously emitted only index/total).
        emit(_chart_event(html, index=0, total=1, sql=state.get("existing_sql")))
        return {"chart_htmls": [html]}

    if mode == "ask_about_viz":
        llm = ChatOpenAI(base_url=cfg.ai_gateway_url, api_key=token, model=cfg.claude_model)

        # ============================================================
        # GROUNDING: the answer comes from GENIE, not from the chart HTML.
        # ============================================================
        # This path used to run straight here from the router with Genie skipped, so the
        # only fact source was the chart HTML plus the SQL text — meaning the "insight"
        # could restate the chart and nothing more. A question like "how does this
        # compare to the overall average?" was structurally unanswerable, and the model,
        # given no data and a demand for specifics, would answer confidently anyway.
        #
        # Now the graph routes through the genie node first, and `genie_results[0]` is a
        # FRESH query against the underlying dataset. Three cases, and which one we are
        # in is always stated to the user rather than inferred by them:
        #
        #   GROUNDED      Genie answered (narrative and/or rows). Answer from it.
        #   GENIE FAILED  Genie ran but errored / timed out / returned nothing. The REAL
        #                 error text goes out in a `thinking` event, and the insight is
        #                 produced with the mandatory no-data disclosure.
        #   NO SQL        Never asked Genie (nothing to ground a question in — the graph
        #                 skipped the node). Same disclosure, different reason.
        #
        # NO `chart` EVENT IS EMITTED ON THIS PATH, in any of the three cases. The user
        # asked a question about a chart already on their screen; re-sending one would
        # make the client re-render it. This branch emits the insight only.
        genie_results = state.get("genie_results") or []
        genie_result = genie_results[0] if genie_results else {}
        genie_error = genie_result.get("error")
        had_sql = bool((state.get("existing_sql") or "").strip())

        # `grounded` REQUIRES POSITIVE EVIDENCE THAT A QUERY ACTUALLY RAN.
        #
        # It used to accept narrative text alone, which was too loose: Genie can return
        # a text attachment with NO rows and NO generated SQL — `_process_message` even
        # falls back to the query attachment's `description` for the narrative — and
        # prose is not proof that anything was executed. Marking that grounded is the
        # silent-degradation shape in its most dangerous form: a confident answer,
        # flagged as data-backed, with no query behind it.
        #
        # So the evidence must be STRUCTURAL: rows came back, or Genie emitted the SQL
        # it ran. Narrative is then used as the phrasing source, but never as the proof.
        # (Rows OR SQL rather than rows alone because a legitimately empty result set —
        # "no reviews match that filter" — is a real answer, and its generated SQL is
        # the evidence the question was actually put to the data.)
        ran_a_query = bool(genie_result.get("rows")) or bool((genie_result.get("sql") or "").strip())
        has_narrative = bool((genie_result.get("text") or "").strip())
        grounded = bool(genie_results) and ran_a_query and not genie_error

        # None on the grounded path. Otherwise the code-owned disclosure text that
        # `_finalize_insight` puts in front of EVERY terminal insight on this path,
        # including a partial one — see BLOCKER 3 in the docstrings above.
        disclosure: str | None = None

        if grounded:
            system_prompt = INSIGHT_GENIE_SYSTEM_PROMPT
            # The chart HTML is deliberately NOT included here. It is a second,
            # competing fact source (it inlines the chart's own dataset), and the point
            # of this path is that the answer comes from the fresh query. The SQL is
            # included as small, honest context for what the chart depicts.
            user_msg = (
                f"The user's question about the chart: {state['question']}\n\n"
                f"The chart they are looking at was produced by this query:\n"
                f"{(state.get('existing_sql') or '').strip()}\n\n"
                f"Result of the fresh query run to answer their question:\n"
                f"{_format_genie_result(genie_result)}"
            )
        else:
            system_prompt = INSIGHT_NO_DATA_SYSTEM_PROMPT
            # No fresh data, so the chart IS the only source and is included. The real
            # reason is surfaced as a `thinking` event BEFORE the answer streams, so the
            # failure is visible even if the model's wording of the disclosure is weak.
            if not had_sql:
                reason = (
                    "The source query for this chart was not saved, so the underlying "
                    "data could not be queried for this question."
                )
                disclosure = _DISCLOSURE_NO_SQL
            elif genie_error:
                reason = f"Genie could not answer this question: {str(genie_error)[:300]}"
                disclosure = _DISCLOSURE_QUERY_FAILED
            elif has_narrative:
                # Genie replied in prose but produced neither rows nor SQL, so nothing
                # proves a query ran. The narrative is NOT presented as data.
                reason = (
                    "Genie replied but did not run a query for this question (no rows "
                    "and no SQL), so its answer is not backed by the data."
                )
                disclosure = _DISCLOSURE_QUERY_FAILED
            else:
                reason = (
                    "Genie returned no data for this question, so there is nothing "
                    "beyond the chart itself to answer from."
                )
                disclosure = _DISCLOSURE_QUERY_FAILED
            emit({"type": "thinking", "text": reason})
            user_msg = (
                f"The user's question about the chart: {state['question']}\n\n"
                f"Why no fresh data is available: {reason}\n\n"
                f"The query that produced the chart:\n{(state.get('existing_sql') or '(not saved)').strip()}\n\n"
                f"Chart HTML (contains the plotted values):\n{(state.get('existing_html') or '')[:3000]}"
            )

        # This branch made an LLM call with no span. It also blocked on a full
        # ainvoke, so the user saw nothing until the whole insight was written —
        # hence the switch to astream below.
        with mlflow.start_span(name="visualizer_insight") as span:
            span.set_inputs({
                "question": tracing.truncate(state["question"]),
                # Genie-generated SQL can embed customer text as a literal
                # (WHERE verbatim LIKE '%...%'), so it goes through the same bound as
                # any other free text instead of being recorded unbounded.
                "sql": tracing.truncate(state.get("existing_sql") or ""),
                "html_chars": len(state.get("existing_html") or ""),
                # Whether the answer is backed by a fresh query is the single most
                # important thing to know when auditing one of these traces.
                "grounded_in_genie": grounded,
                "genie_row_count": len(genie_result.get("rows") or []),
                "genie_error": tracing.truncate(genie_error) or None,
                "had_sql": had_sql,
            })
            # STREAMED, not ainvoke: `insight_delta` events carry the text out as the
            # model writes it.
            #
            # ============================================================
            # THE `insight` / `insight_delta` CLIENT CONTRACT — AUTHORITATIVE
            # ============================================================
            # Kept in ONE comment block, success and failure together, so the two
            # halves cannot drift apart. `_stream_text_deltas`'s docstring points here.
            #
            # RULE: `insight_delta` is a progressive PREVIEW and is never the final
            # word. The terminal `insight` event REPLACES the client's accumulated
            # delta text — it is never appended to. A terminal `insight` follows at
            # least one `insight_delta` on the success path AND on the failure path, so
            # a client may render deltas immediately and rely on being told what the
            # real text is.
            #
            # THE ONE EXCLUSION, stated rather than glossed: this holds only while the
            # TRANSPORT works. No code can deliver an event through a failed emitter, so
            # "a terminal event always arrives" is conditional on the emitter, not
            # absolute. Concretely, on CLIENT DISCONNECT the handler does attempt the
            # partial terminal emit, but the SSE generator is already in its `finally`
            # and the event is never delivered — inherent to cancellation, not fixable,
            # and harmless (whatever would have rendered it is what disconnected). An
            # overstated guarantee would be worse than this honest one: a client told the
            # terminal event is unconditional would never handle its absence.
            #
            # REPLACE-not-append is also load-bearing for a reason beyond post-
            # processing: on the UNGROUNDED path the terminal text carries a code-owned
            # disclosure prefix that the deltas do not have. A client that appended would
            # silently drop the caveat.
            #
            # The three sequences a client must handle:
            #
            #  (a) SUCCESS
            #        insight_delta × N  →  insight{text, partial:false}  →  done
            #      Replace accumulated text with `insight.text`. As before streaming.
            #
            #  (b) FAILURE AFTER N ≥ 1 DELTAS (30s timeout, or the LLM erroring)
            #        insight_delta × N  →  insight{text, partial:true, error}  →  error
            #      `insight.text` is the PARTIAL text that actually arrived, flagged
            #      `partial: true` with `error` carrying the reason. Replace the
            #      accumulated text with it and present it as incomplete (or discard
            #      it) — do NOT render it as a finished answer. This event is the fix
            #      for the original hole: previously the exception propagated past this
            #      emit, the SSE generator broke on `error`, and the client was left
            #      showing partial text with no correction and not even a `done`.
            #      NOTE the `error` event still terminates the stream, so `done` does
            #      NOT follow it — treat `error` as terminal, exactly as before.
            #
            #  (c) FAILURE BEFORE ANY DELTA
            #        error
            #      NO `insight` event at all. Nothing was rendered, so there is
            #      nothing to correct and inventing an empty terminal event would just
            #      make a client clear a pane it never filled. Identical to the
            #      pre-streaming failure behaviour.
            #
            # `partial` is present on EVERY terminal `insight` (explicitly false on
            # success) so a client can branch on the field itself rather than on its
            # absence — an absent key and a false key must not mean different things.
            deltas: list[str] = []

            def _on_delta(piece: str) -> None:
                deltas.append(piece)
                emit({"type": "insight_delta", "text": piece})

            try:
                async with asyncio.timeout(30):
                    insight = await _stream_text_deltas(
                        llm,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        _on_delta,
                    )
            except BaseException as e:
                # BaseException on purpose: asyncio.timeout surfaces as TimeoutError,
                # but a client disconnect cancels this task and raises CancelledError,
                # which is NOT an Exception. Both leave the same orphaned partial on a
                # client that was rendering deltas, so both must emit the terminal
                # correction.
                #
                # THE ORIGINAL EXCEPTION IS PRESERVED UNCONDITIONALLY. Every cleanup
                # step below is individually guarded and its own failure is SWALLOWED,
                # because the alternative is worse in both directions:
                #   * an `emit` that raises would otherwise mask the real cause AND
                #     still leave the client's partial text uncorrected;
                #   * a `span.set_outputs` that raises would otherwise replace the
                #     in-flight exception before the bare `raise` — turning a genuine
                #     CancelledError into an unrelated error and breaking cooperative
                #     cancellation, which is worse than the bug this handler fixes.
                # Best-effort cleanup, then an unconditional bare `raise`. Nothing
                # between here and that `raise` is allowed to substitute an exception.
                partial = "".join(deltas).strip()
                if partial:
                    try:
                        emit({
                            "type": "insight",
                            # Disclosed even on a partial: a fragment of an ungrounded
                            # answer must not look data-backed either.
                            "text": _finalize_insight(partial, disclosure),
                            "partial": True,
                            "grounded": grounded,
                            "error": str(e) or type(e).__name__,
                        })
                    except BaseException:
                        # The transport itself is broken. There is no channel left to
                        # report anything on, so the only correct action is to let the
                        # ORIGINAL failure propagate untouched. Deliberately not logged
                        # through `emit` for the same reason. Logged via the helper
                        # because even `logger.warning` must not be able to raise here.
                        _log_cleanup_failure(
                            "ask_about_viz: could not emit the partial terminal insight; "
                            "the original error is being re-raised"
                        )
                try:
                    span.set_outputs({
                        "insight_chars": len(partial),
                        "delta_chunks": len(deltas),
                        "partial": True,
                        "grounded_in_genie": grounded,
                        "error": tracing.truncate(str(e) or type(e).__name__),
                    })
                except BaseException:
                    # Tracing is observability, never control flow. A span that cannot
                    # record its outputs must not change what the caller sees.
                    _log_cleanup_failure(
                        "ask_about_viz: could not record failure outputs on the span"
                    )
                # Bare `raise`: re-raises the ORIGINAL exception. This is now
                # UNCONDITIONAL rather than merely usual — every statement between the
                # `except BaseException` above and this line is either inside its own
                # BaseException guard or is `_log_cleanup_failure`, which cannot raise.
                # Nothing in this window can substitute a different exception, so an
                # in-flight CancelledError still reaches the caller and cooperative
                # cancellation keeps working.
                raise
            # The disclosure is attached HERE, in code, not requested from the model.
            # On the grounded path `disclosure` is None and this is a no-op.
            insight = _finalize_insight(insight.strip(), disclosure)
            span.set_outputs({
                "insight_chars": len(insight),
                "delta_chunks": len(deltas),
                "partial": False,
                "grounded_in_genie": grounded,
                "disclosed": disclosure is not None,
            })
        # TERMINAL event, unchanged in shape (`partial: false` is additive) and still
        # carrying the FULL text. A client that ignores `insight_delta` sees exactly
        # the old sequence. See the contract block above for the failure sequences.
        #
        # NOTE the terminal text can now differ from the concatenated deltas by the
        # code-owned disclosure prefix. That is exactly what REPLACE-not-append is for,
        # and it is the reason the terminal event is authoritative: a client that renders
        # deltas live sees the model's words, then gets the disclosed version as the
        # final answer. Appending instead of replacing would drop the caveat.
        #
        # `grounded` is ADDITIVE and advisory: it lets a client mark an answer that has
        # no fresh data behind it without parsing the prose. The disclosure is ALSO in
        # the text itself — and, since BLOCKER 3, put there by CODE rather than by the
        # prompt — so a client that ignores this field still shows an honest answer.
        emit({"type": "insight", "text": insight, "partial": False, "grounded": grounded})
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
