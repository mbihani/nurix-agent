"""
Lightweight, LLM-free tests for the visualizer data-injection helpers.

Run:  uv run python tests/test_chart_injection.py
Exits non-zero on any failed assertion. Imports the REAL helpers from the
module so it exercises shipping code, not a copy.
"""
import json
import re

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


def main():
    test_stray_block_yields_exactly_one_trusted_block()
    test_multiple_stray_blocks_all_stripped()
    test_script_before_head_ordering()
    test_full_dataset_aggregation_totals()
    test_script_literal_in_comment_ignored()
    test_script_literal_in_textarea_ignored()
    test_no_script_uses_head_fallback()
    print("\nALL CHART-INJECTION TESTS PASSED")


if __name__ == "__main__":
    main()
