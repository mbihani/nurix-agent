"""
Lightweight, LLM-free tests for the visualizer data-injection helpers.

Run:  uv run python tests/test_chart_injection.py
Exits non-zero on any failed assertion. Imports the REAL helpers from the
module so it exercises shipping code, not a copy.
"""
import json
import re

import nurix_agent.nodes.visualizer as viz
from nurix_agent.nodes.visualizer import (
    _split_chart_data,
    _inject_chart_data,
    _insert_head_script,
    _build_data_script,
)

_BLOCK_RE = re.compile(r"window\.CHART_DATA\s*=", re.IGNORECASE)


def _extract_payload(html: str) -> dict:
    """Parse the single injected window.CHART_DATA JS literal back to a dict."""
    m = re.search(r"window\.CHART_DATA\s*=\s*(\{.*?\});</script>", html, re.DOTALL)
    assert m, "no window.CHART_DATA block found"
    return json.loads(m.group(1))  # JSON understands the \\uXXXX escapes we add


def test_stray_block_yields_exactly_one_trusted_block():
    """(a) A stray LLM window.CHART_DATA block must be stripped; only ours survives."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    stray = '{"columns":[{"name":"x","type":"string"}],"rows":[["STALE_TRUNCATED"]]}'
    # LLM scaffold that (against instructions) emitted its own data block.
    llm_html = (
        "<!DOCTYPE html><html><head>"
        f"<script>window.CHART_DATA = {stray};</script>"
        "</head><body><canvas></canvas>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
        "<script>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    # _generate_chart path: strip all, then inject the single trusted block.
    _, cleaned = _split_chart_data(llm_html)
    assert "STALE_TRUNCATED" not in cleaned, "stray block not stripped from scaffold"
    final = _inject_chart_data(cleaned, trusted)

    count = len(_BLOCK_RE.findall(final))
    assert count == 1, f"expected exactly one CHART_DATA block, found {count}"
    payload = _extract_payload(final)
    assert payload == trusted, f"surviving block is not the trusted one: {payload}"
    assert "STALE_TRUNCATED" not in final
    print("PASS (a) stray block stripped -> exactly one trusted block")


def test_multiple_stray_blocks_all_stripped():
    """(a-extra) Even multiple stray blocks are all removed before injection."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    llm_html = (
        "<!DOCTYPE html><html><head>"
        '<script>window.CHART_DATA = {"a":1};</script>'
        "</head><body>"
        '<script>window.CHART_DATA = {"b":2};</script>'
        "<script>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    _, cleaned = _split_chart_data(llm_html)
    final = _inject_chart_data(cleaned, trusted)
    count = len(_BLOCK_RE.findall(final))
    assert count == 1, f"expected exactly one block, found {count}"
    assert _extract_payload(final) == trusted
    print("PASS (a-extra) multiple stray blocks all stripped -> one trusted block")


def test_script_before_head_ordering():
    """(b) Scaffold with a <script> before <head>: injected data must precede it."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    # Malformed: a <script> appears BEFORE <head>.
    malformed = (
        "<script>console.log('early');</script>"
        "<!DOCTYPE html><html><head></head><body>"
        "<script>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(malformed, trusted)
    data_pos = final.index("window.CHART_DATA =")
    first_script_pos = re.search(r"<script\b", final, re.IGNORECASE).start()
    # Our injected data <script> must BE the first <script> in the document.
    assert data_pos - len("<script>") <= first_script_pos, "data not before first script"
    early_pos = final.index("console.log('early')")
    assert data_pos < early_pos, "data block must precede the pre-head <script>"
    assert len(_BLOCK_RE.findall(final)) == 1
    print("PASS (b) data <script> precedes a pre-<head> <script>")


def test_full_dataset_aggregation_totals():
    """(c) Full 72-row sentiment dataset round-trips and aggregates to correct totals."""
    products = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    features = ["Search", "Checkout", "Onboarding", "Reports"]
    sentiments = ["negative", "neutral", "positive"]
    # Deterministic counts whose per-sentiment totals match the live smoke case.
    targets = {"negative": 2394, "neutral": 2298, "positive": 2760}
    ncells = len(products) * len(features)  # 12 (product, feature) combos per sentiment
    rows = []
    per = {}
    for s in sentiments:
        base = targets[s] // ncells
        rem = targets[s] - base * ncells
        per[s] = [base + (1 if i < rem else 0) for i in range(ncells)]
    ci = 0
    for p in products:
        for f in features:
            for s in sentiments:
                rows.append([p, f, s, per[s][ci]])
            ci += 1
    assert len(rows) == 72, f"expected 72 rows, built {len(rows)}"
    data = {
        "columns": [
            {"name": "product", "type": "string"},
            {"name": "feature_area", "type": "string"},
            {"name": "sentiment_label", "type": "string"},
            {"name": "review_count", "type": "number"},
        ],
        "rows": rows,
    }
    # Inject into a minimal scaffold and read the payload back out.
    html = _inject_chart_data(
        "<!DOCTYPE html><html><head></head><body>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
        "<script>const d=window.CHART_DATA;</script></body></html>",
        data,
    )
    assert len(_BLOCK_RE.findall(html)) == 1
    payload = _extract_payload(html)
    assert len(payload["rows"]) == 72, f"lost rows: {len(payload['rows'])}"
    totals = {}
    for r in payload["rows"]:
        totals[r[2]] = totals.get(r[2], 0) + r[3]
    assert totals == targets, f"aggregation mismatch: {totals} != {targets}"
    print(f"PASS (c) full 72-row aggregation totals correct: {totals}")


def test_script_literal_in_comment_ignored():
    """(d) A `<script` literal inside an HTML comment before the real chart script
    must NOT be treated as the first script: data lands before the REAL script."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head>"
        "<!-- example usage: <script>foo()</script> -->"
        "</head><body><canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    data_pos = final.index("window.CHART_DATA =")
    real_pos = final.index("<script id='chart'>")
    comment_end = final.index("-->")
    # Injected data must come AFTER the comment (not inside it) and BEFORE the real script.
    assert data_pos > comment_end, "data injected inside/before the comment"
    assert data_pos < real_pos, "data not before the real chart script"
    print("PASS (d) `<script` inside HTML comment ignored; data before real script")


def test_script_literal_in_textarea_ignored():
    """(e) A `<script` literal inside <textarea> rawtext before the real script
    is ignored; data lands before the REAL executable script."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head></head><body>"
        "<textarea>paste like <script>evil()</script> here</textarea>"
        "<canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    data_pos = final.index("window.CHART_DATA =")
    textarea_close = final.index("</textarea>")
    real_pos = final.index("<script id='chart'>")
    assert data_pos > textarea_close, "data injected inside/before the textarea"
    assert data_pos < real_pos, "data not before the real chart script"
    print("PASS (e) `<script` inside <textarea> ignored; data before real script")


def test_no_script_uses_head_fallback():
    """(f) A document with NO <script> falls back to just-after <head>."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = "<!DOCTYPE html><html><head><title>t</title></head><body><canvas></canvas></body></html>"
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    head_open_end = final.lower().index("<head>") + len("<head>")
    data_pos = final.index("<script>window.CHART_DATA")
    # Data script sits immediately after the opening <head> tag.
    assert data_pos == head_open_end, f"expected data right after <head>, at {data_pos} vs {head_open_end}"
    print("PASS (f) no <script> -> data injected just after <head> (fallback)")


def test_complete_script_in_title_ignored():
    """(h) A COMPLETE `<script>evil()</script>` literal inside <title> BEFORE the
    real chart script must NOT be treated as the first executable script. Browsers
    treat <title> as RCDATA, so the inner script is inert text. Our trusted data
    block must be injected BEFORE the real chart <script>, NOT inside <title>."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head>"
        "<title>Chart <script>evil()</script> title</title>"
        "</head><body><canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    data_pos = final.index("window.CHART_DATA =")
    title_close = final.index("</title>")
    real_pos = final.index("<script id='chart'>")
    # Data must land AFTER the closing </title> (not inside it) and BEFORE the real script.
    assert data_pos > title_close, "data injected inside/before the <title> RCDATA"
    assert data_pos < real_pos, "data not before the real chart script"
    print("PASS (h) complete <script> inside <title> ignored; data before real script")


def test_complete_script_in_textarea_ignored():
    """(i) A COMPLETE `<script>evil()</script>` literal inside <textarea> BEFORE
    the real chart script must NOT be treated as the first executable script.
    <textarea> is RCDATA, so the inner script is inert text; data lands BEFORE the
    real chart <script>, NOT inside the textarea."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head></head><body>"
        "<textarea>type <script>evil()</script> stuff</textarea>"
        "<canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    data_pos = final.index("window.CHART_DATA =")
    textarea_close = final.index("</textarea>")
    real_pos = final.index("<script id='chart'>")
    assert data_pos > textarea_close, "data injected inside/before the <textarea> RCDATA"
    assert data_pos < real_pos, "data not before the real chart script"
    print("PASS (i) complete <script> inside <textarea> ignored; data before real script")


def test_parser_exception_prepends_data(monkeypatch):
    """(j) On a PARSER EXCEPTION we cannot locate scripts, so the data block must
    be PREPENDED to the very front of the document (index 0) — guaranteeing
    window.CHART_DATA is defined before ANY other content/script executes. We must
    NOT fall back to the <head> search, which could place data AFTER a pre-<head>
    script and reintroduce the ordering bug."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    # A scaffold whose FIRST script precedes <head> — the exact shape where a
    # <head>-based fallback would be WRONG.
    scaffold = (
        "<script>console.log('early');</script>"
        "<!DOCTYPE html><html><head></head><body>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )

    def boom(self, *a, **k):
        raise ValueError("simulated parser failure")

    # Force the finder to raise on parse.
    monkeypatch.setattr(viz._FirstScriptFinder, "feed", boom)
    monkeypatch.setattr(viz._FirstScriptFinder, "close", boom)

    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    # Data block is at the very front — before the original document.
    assert final.startswith("<script>window.CHART_DATA"), (
        f"data not prepended on parse failure: starts with {final[:40]!r}"
    )
    data_pos = final.index("window.CHART_DATA =")
    early_pos = final.index("console.log('early')")
    real_pos = final.index("<script id='chart'>")
    assert data_pos == len("<script>"), f"data not at index 0 region: {data_pos}"
    assert data_pos < early_pos < real_pos, "data must precede ALL other scripts on exception"
    print("PASS (j) parser exception -> data PREPENDED before all content")


def test_no_regression_all_prior_assertions():
    """(k) Re-run ALL prior scenarios end-to-end to confirm no regression, and
    re-assert the full 72-row aggregation totals (negative 2394 / neutral 2298 /
    positive 2760)."""
    test_stray_block_yields_exactly_one_trusted_block()
    test_multiple_stray_blocks_all_stripped()
    test_script_before_head_ordering()
    test_full_dataset_aggregation_totals()
    test_script_literal_in_comment_ignored()
    test_script_literal_in_textarea_ignored()
    test_no_script_uses_head_fallback()

    # Explicit re-assertion of the smoke totals through a fresh injection.
    products = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    features = ["Search", "Checkout", "Onboarding", "Reports"]
    sentiments = ["negative", "neutral", "positive"]
    targets = {"negative": 2394, "neutral": 2298, "positive": 2760}
    ncells = len(products) * len(features)
    per = {}
    for s in sentiments:
        base = targets[s] // ncells
        rem = targets[s] - base * ncells
        per[s] = [base + (1 if i < rem else 0) for i in range(ncells)]
    rows = []
    ci = 0
    for p in products:
        for f in features:
            for s in sentiments:
                rows.append([p, f, s, per[s][ci]])
            ci += 1
    data = {
        "columns": [
            {"name": "product", "type": "string"},
            {"name": "feature_area", "type": "string"},
            {"name": "sentiment_label", "type": "string"},
            {"name": "review_count", "type": "number"},
        ],
        "rows": rows,
    }
    html = _inject_chart_data(
        "<!DOCTYPE html><html><head></head><body>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
        "<script>const d=window.CHART_DATA;</script></body></html>",
        data,
    )
    payload = _extract_payload(html)
    assert len(payload["rows"]) == 72
    totals = {}
    for r in payload["rows"]:
        totals[r[2]] = totals.get(r[2], 0) + r[3]
    assert totals == targets, f"regression in totals: {totals} != {targets}"
    print(f"PASS (k) no regression across all prior tests; totals still {totals}")


def _assert_data_is_first_script(final: str, label: str):
    """Assert the injected data block is PREPENDED — before ANY script in the doc."""
    assert final.startswith("<script>window.CHART_DATA"), (
        f"{label}: data not prepended; document starts with {final[:48]!r}"
    )
    # No other <script> may precede ours anywhere in the document.
    starts = [m.start() for m in re.finditer(r"<script\b", final, re.IGNORECASE)]
    assert starts and starts[0] == 0, f"{label}: another <script> precedes ours at {starts[:2]}"


def test_self_closing_textarea_forces_prepend():
    """(l) A SELF-CLOSING `<textarea/>` before the real chart <script>. Browsers do
    NOT treat <textarea> as a void element, so `<textarea/>` leaves the element OPEN
    and every later script is inert text. We must therefore NOT record that script
    as our insertion point, and must NOT degrade to the <head> fallback (which could
    land AFTER it) — the data block must be PREPENDED to the very front so
    window.CHART_DATA is defined before ANY script in the document."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head></head><body>"
        "<textarea/>"
        "<canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    _assert_data_is_first_script(final, "(l)")
    data_pos = final.index("window.CHART_DATA =")
    assert data_pos < final.index("<textarea/>"), "data must precede the open <textarea/>"
    assert data_pos < final.index("<script id='chart'>"), "data must precede the chart script"
    assert _extract_payload(final) == trusted
    print("PASS (l) self-closing <textarea/> -> data PREPENDED before all scripts")


def test_self_closing_title_forces_prepend():
    """(m) Same as (l) for a SELF-CLOSING `<title/>`: not a void element, so it stays
    OPEN in a browser and the later chart <script> is inert. Data must be PREPENDED."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head><title/></head><body><canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    _assert_data_is_first_script(final, "(m)")
    data_pos = final.index("window.CHART_DATA =")
    assert data_pos < final.index("<title/>"), "data must precede the open <title/>"
    assert data_pos < final.index("<script id='chart'>"), "data must precede the chart script"
    assert _extract_payload(final) == trusted
    print("PASS (m) self-closing <title/> -> data PREPENDED before all scripts")


def test_properly_closed_textarea_still_injects_before_real_script():
    """(n) REGRESSION GUARD — the conservative prepend must NOT over-trigger. A
    NORMAL well-formed document whose <textarea> is properly CLOSED before the chart
    script must still get the precise placement: data injected immediately BEFORE
    the REAL chart <script>, NOT degraded to a front-of-document prepend."""
    trusted = {"columns": [{"name": "x", "type": "string"}], "rows": [["real"]]}
    scaffold = (
        "<!DOCTYPE html><html><head><title>Chart</title></head><body>"
        "<textarea>notes go here</textarea>"
        "<canvas></canvas>"
        "<script id='chart'>const d = window.CHART_DATA;</script>"
        "</body></html>"
    )
    final = _inject_chart_data(scaffold, trusted)
    assert len(_BLOCK_RE.findall(final)) == 1
    # NOT prepended: the healthy document keeps its original prefix.
    assert final.startswith("<!DOCTYPE html>"), (
        f"healthy document was degraded to a prepend: starts with {final[:48]!r}"
    )
    data_pos = final.index("window.CHART_DATA =")
    real_pos = final.index("<script id='chart'>")
    textarea_close = final.index("</textarea>")
    title_close = final.index("</title>")
    # Data sits after BOTH closed RCDATA elements and immediately before the real script.
    assert data_pos > title_close, "data injected inside/before the closed <title>"
    assert data_pos > textarea_close, "data injected inside/before the closed <textarea>"
    assert data_pos < real_pos, "data not before the real chart script"
    assert final.index("<script>window.CHART_DATA") == real_pos - len(
        _build_data_script(trusted)
    ), "data block not immediately adjacent to the real chart script"
    assert _extract_payload(final) == trusted
    print("PASS (n) properly closed <textarea>/<title> -> precise pre-chart-script injection (no over-trigger)")


def test_full_dataset_aggregation_totals_rerun():
    """(o) Re-run the full-dataset aggregation check after the RCDATA fix: the 72-row
    product x feature_area x sentiment fixture must still total negative 2394 /
    neutral 2298 / positive 2760."""
    test_full_dataset_aggregation_totals()
    print("PASS (o) full-dataset aggregation re-verified: 2394 / 2298 / 2760")


def _run_with_monkeypatch(fn):
    """Minimal monkeypatch shim so tests run under plain `python` (no pytest)."""
    import contextlib

    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value):
            old = getattr(target, name)
            self._undo.append((target, name, old))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._undo):
                setattr(target, name, old)

    mp = _MP()
    try:
        fn(mp)
    finally:
        mp.undo()


def main():
    test_stray_block_yields_exactly_one_trusted_block()
    test_multiple_stray_blocks_all_stripped()
    test_script_before_head_ordering()
    test_full_dataset_aggregation_totals()
    test_script_literal_in_comment_ignored()
    test_script_literal_in_textarea_ignored()
    test_no_script_uses_head_fallback()
    test_complete_script_in_title_ignored()
    test_complete_script_in_textarea_ignored()
    _run_with_monkeypatch(test_parser_exception_prepends_data)
    test_no_regression_all_prior_assertions()
    test_self_closing_textarea_forces_prepend()
    test_self_closing_title_forces_prepend()
    test_properly_closed_textarea_still_injects_before_real_script()
    test_full_dataset_aggregation_totals_rerun()
    print("\nALL CHART-INJECTION TESTS PASSED")


if __name__ == "__main__":
    main()
