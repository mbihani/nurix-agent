"""
Deterministic, LLM-free and network-free tests for the Genie Agent mode SSE parser.

Fixtures are REAL recorded streams from the fevm-stable space:
  genie_agent_multistep.sse  — "top feature areas driving negative sentiment, and how
                                has that changed over time?" (40.9s, 38 frames,
                                5 reasoning items, 7 function_calls / 5 distinct SQL,
                                final message with 5 parts, 2 carrying metadata)
  genie_agent_singlestep.sse — a single-query run whose final message carries NO
                               metadata at all (every sub-query needs re-execution)

Run:  uv run python tests/test_genie_agent_parser.py
Exits non-zero on any failed assertion. Imports the REAL helpers so it exercises
shipping code, not a copy.
"""
import json
import sys
from pathlib import Path

from nurix_agent.genie_agent import (
    AgentStreamAccumulator,
    SSEFrameParser,
    _normalize_sql,
    _columns_from_metadata,
    _rows_from_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures"
MULTISTEP = FIXTURES / "genie_agent_multistep.sse"
SINGLESTEP = FIXTURES / "genie_agent_singlestep.sse"


def _replay(path: Path):
    """Feed a recorded stream through the real parser + accumulator, line by line."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    parser = SSEFrameParser()
    frames = 0
    for line in path.read_text().splitlines():
        for event, data in parser.feed(line):
            frames += 1
            acc.handle_frame(event, data)
    for event, data in parser.close():
        frames += 1
        acc.handle_frame(event, data)
    return acc, events, frames


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------

def test_parser_frame_count_matches_recorded_stream():
    _, _, frames = _replay(MULTISTEP)
    # 1 created + 18 added + 18 done + 1 completed
    assert frames == 38, f"expected 38 frames, got {frames}"


def test_parser_handles_sse_framing_details():
    """Optional space after the colon, multi-line data, comments, and a final
    frame with no trailing blank line."""
    p = SSEFrameParser()
    out = []
    for line in [
        ": keep-alive",
        "event: response.created",
        'data: {"response":',
        'data: {"id":"r1","conversation_id":"c1"}}',
        "",
    ]:
        out += p.feed(line)
    assert len(out) == 1, out
    event, data = out[0]
    assert event == "response.created"
    assert data["response"]["id"] == "r1"

    # No trailing blank line -> close() must still flush.
    p2 = SSEFrameParser()
    assert p2.feed("event:response.completed") == []
    assert p2.feed('data:{"response":{"status":"completed"}}') == []
    flushed = p2.close()
    assert len(flushed) == 1 and flushed[0][0] == "response.completed"


def test_parser_ignores_malformed_data():
    p = SSEFrameParser()
    assert p.feed("event:response.created") == []
    assert p.feed("data:{not json") == []
    assert p.feed("") == []  # malformed frame dropped, no exception


# --------------------------------------------------------------------------
# reasoning -> thinking mapping
# --------------------------------------------------------------------------

def test_reasoning_maps_to_thinking_once_per_item():
    """5 reasoning items, each arriving as .added AND .done AND replayed in
    response.completed — must yield exactly 5 thinking events, in order."""
    acc, events, _ = _replay(MULTISTEP)
    thinking = [e for e in events if e["type"] == "thinking"]
    reasoning_texts = [
        "I'll analyze the top feature areas driving negative sentiment",
        "Let me retry those queries with the correct task numbers",
        "I see all the data is from March 2026",
        "Now let me analyze the daily breakdown",
        "Completing checklist item",
    ]
    assert acc.reasoning_count == 5, f"expected 5 reasoning items, got {acc.reasoning_count}"
    matched = [t["text"] for t in thinking if any(r in t["text"] for r in reasoning_texts)]
    assert len(matched) == 5, f"expected 5 reasoning thinking events, got {len(matched)}: {matched}"
    # Order preserved.
    for expected, actual in zip(reasoning_texts, matched):
        assert expected in actual, f"order broke: {expected!r} vs {actual!r}"


def test_reasoning_is_emitted_progressively_not_batched():
    """A reasoning thinking event must be emitted BEFORE the terminal frame is
    ever fed — i.e. progressive, not accumulated and flushed at the end."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    parser = SSEFrameParser()
    for line in MULTISTEP.read_text().splitlines():
        for event, data in parser.feed(line):
            if event == "response.completed":
                # Everything the stream can tell the user about progress — all 5
                # reasoning steps and all 5 distinct queries — is already out the
                # door before the terminal frame is seen.
                assert acc.reasoning_count == 5, (
                    f"only {acc.reasoning_count}/5 reasoning items emitted before the "
                    "terminal frame — emission is batched, not progressive"
                )
                assert len([e for e in events if e["text"].startswith("Querying: ")]) == 5
                assert len(events) == 10, f"expected 10 progressive events, got {len(events)}"
                return
            acc.handle_frame(event, data)
    raise AssertionError("no response.completed frame in fixture")


def test_function_call_emits_thinking_naming_the_query():
    acc, events, _ = _replay(MULTISTEP)
    querying = [e for e in events if e["type"] == "thinking" and e["text"].startswith("Querying: ")]
    titles = [e["text"][len("Querying: "):] for e in querying]
    # 7 function_calls collapse to 5 distinct SQL -> 5 announcements.
    assert titles == [
        "Top feature areas by negative review count",
        "Monthly negative reviews by feature area",
        "Date range of review data",
        "Weekly negative reviews by feature area",
        "Daily negative reviews by feature area with percentages",
    ], titles


# --------------------------------------------------------------------------
# Sub-query SQL recovery
# --------------------------------------------------------------------------

def test_all_subquery_sql_recovered_and_deduped():
    """7 function_call items, 5 distinct SQL statements (Genie retried 2 verbatim)."""
    acc, _, _ = _replay(MULTISTEP)
    subs = acc.subqueries()
    assert len(subs) == 5, f"expected 5 distinct sub-queries, got {len(subs)}"
    for sq in subs:
        assert sq["sql"].strip(), "recovered a sub-query with empty SQL"
        assert "SELECT" in sq["sql"].upper()
    # Every distinct title is preserved.
    assert len({sq["title"] for sq in subs}) == 5


def test_dedup_is_whitespace_insensitive():
    assert _normalize_sql("SELECT\n  a\nFROM t") == _normalize_sql("SELECT a FROM t")
    events = []
    acc = AgentStreamAccumulator(events.append)
    for args in ('{"title":"A","sql":"SELECT 1"}', '{"title":"A","sql":"SELECT\\n   1"}'):
        acc.handle_frame("response.output_item.added", {
            "item": {"type": "function_call", "id": f"id{args[-3]}", "name": "execute_sql",
                     "arguments": args},
        })
    assert len(acc.subqueries()) == 1, "whitespace-only SQL variants were not deduped"


def test_non_execute_sql_function_calls_ignored():
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.added", {
        "item": {"type": "function_call", "id": "x", "name": "something_else",
                 "arguments": '{"sql":"SELECT 1"}'},
    })
    assert acc.subqueries() == []


# --------------------------------------------------------------------------
# metadata extraction
# --------------------------------------------------------------------------

def test_metadata_extracted_for_the_two_featured_results():
    """The recorded multi-step run carries metadata on 2 of 5 message parts (NOT 1)."""
    acc, _, _ = _replay(MULTISTEP)
    subs = acc.subqueries()
    from_metadata = [s for s in subs if s["source"] == "metadata"]
    pending = [s for s in subs if s["source"] == "pending"]
    assert len(from_metadata) == 2, f"expected 2 metadata-backed sub-queries, got {len(from_metadata)}"
    assert len(pending) == 3, f"expected 3 sub-queries needing re-execution, got {len(pending)}"

    by_title = {s["title"]: s for s in from_metadata}
    top = by_title["Top feature areas by negative review count"]
    assert [c["name"] for c in top["columns"]] == [
        "feature_area", "negative_review_count", "pct_of_negative_reviews"
    ]
    assert [c["type"] for c in top["columns"]] == ["string", "number", "number"]
    assert len(top["rows"]) == 8, f"expected 8 rows, got {len(top['rows'])}"
    assert top["rows"][0] == ["AI Tools", 311, 12.99], top["rows"][0]

    daily = by_title["Daily negative reviews by feature area with percentages"]
    assert len(daily["rows"]) == 16
    assert daily["rows"][0] == ["2026-03-16", "AI Tools", 303, 13.13], daily["rows"][0]


def test_metadata_sql_matches_back_to_a_recovered_subquery():
    """No metadata block should be orphaned — each maps onto a function_call SQL."""
    acc, _, _ = _replay(MULTISTEP)
    assert acc.orphan_metadata() == [], "a metadata block failed to match any function_call"


def test_databricks_types_coerced_via_shared_helpers():
    """BIGINT/DECIMAL -> number and coerced from strings; STRING/DATE stay strings."""
    columns, flags = _columns_from_metadata([
        {"name": "s", "type": "STRING"},
        {"name": "n", "type": "BIGINT"},
        {"name": "d", "type": "DECIMAL(27,2)"},
        {"name": "dt", "type": "DATE"},
    ])
    assert [c["type"] for c in columns] == ["string", "number", "number", "string"]
    assert flags == [False, True, True, False]
    rows = _rows_from_metadata([["a", "311", "12.99", "2026-03-16"]], flags)
    assert rows == [["a", 311, 12.99, "2026-03-16"]], rows


def test_partial_metadata_is_marked_pending_not_charted():
    """preview_rows short of total_row_count must NOT be charted as-is — charting a
    partial result silently undercounts."""
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.added", {
        "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                 "arguments": json.dumps({"title": "T", "sql": "SELECT a FROM t"})},
    })
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m1", "content": [{
            "type": "output_text", "text": "table",
            "metadata": {
                "sql": "SELECT a FROM t",
                "columns": [{"name": "a", "type": "BIGINT"}],
                "preview_rows": [["1"], ["2"]],
                "status": "available",
                "total_row_count": 500,
            },
        }]},
    })
    sq = acc.subqueries()[0]
    assert sq["source"] == "pending", f"partial metadata was accepted: {sq}"
    assert [c["name"] for c in sq["columns"]] == ["a"], "schema should be retained"


def test_orphan_metadata_is_reported_not_dropped():
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m1", "content": [{
            "type": "output_text", "text": "t",
            "metadata": {"sql": "SELECT unmatched", "columns": [{"name": "a", "type": "STRING"}],
                         "preview_rows": [["x"]], "total_row_count": 1},
        }]},
    })
    orphans = acc.orphan_metadata()
    assert len(orphans) == 1 and orphans[0]["sql"] == "SELECT unmatched"


# --------------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------------

def test_narrative_excludes_metadata_table_parts():
    acc, _, _ = _replay(MULTISTEP)
    narrative = acc.narrative()
    assert "Top Feature Areas Driving Negative Sentiment" in narrative
    assert "Changes Over Time" in narrative
    # The markdown table renderings live on the metadata-bearing parts and are
    # charted separately, so they must not be duplicated into the prose.
    assert "| --- | --- |" not in narrative, "narrative duplicated a metadata table"
    assert acc.completed is True


def test_narrative_falls_back_to_all_parts_when_every_part_has_metadata():
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m", "content": [{
            "type": "output_text", "text": "only text",
            "metadata": {"sql": "SELECT 1", "columns": [], "preview_rows": [], "total_row_count": 0},
        }]},
    })
    assert acc.narrative() == "only text"


# --------------------------------------------------------------------------
# Single-step fixture: the no-metadata-at-all case
# --------------------------------------------------------------------------

def test_singlestep_stream_has_no_metadata_so_all_subqueries_pending():
    acc, events, _ = _replay(SINGLESTEP)
    subs = acc.subqueries()
    assert len(subs) == 1, f"expected 1 sub-query, got {len(subs)}"
    assert subs[0]["source"] == "pending", (
        "single-step fixture carries no metadata; its sub-query must be queued for "
        f"re-execution, got source={subs[0]['source']!r}"
    )
    assert subs[0]["title"] == "Average sentiment score by product tier"
    assert "avg_sentiment_score" in subs[0]["sql"]
    assert acc.reasoning_count == 2
    assert acc.completed is True
    assert "nearly identical across all plan tiers" in acc.narrative()
    # Zero rows recovered from the stream alone -> the re-execution path is the ONLY
    # way this run produces a chart.
    assert subs[0]["rows"] == []


def test_conversation_and_response_ids_captured():
    acc, _, _ = _replay(MULTISTEP)
    assert acc.conversation_id, "conversation_id not captured"
    assert acc.response_id, "response_id not captured"


def test_completed_frame_backfills_items_that_never_arrived_individually():
    """If a per-item frame is dropped, the replay in response.completed must still
    recover the sub-query."""
    completed = None
    parser = SSEFrameParser()
    for line in MULTISTEP.read_text().splitlines():
        for event, data in parser.feed(line):
            if event == "response.completed":
                completed = data
    assert completed is not None
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.completed", completed)  # terminal frame ONLY
    assert len(acc.subqueries()) == 5, "completed-frame backfill lost sub-queries"
    assert acc.reasoning_count == 5
    assert len([s for s in acc.subqueries() if s["source"] == "metadata"]) == 2


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
