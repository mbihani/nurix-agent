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
import asyncio
import json
import sys
from pathlib import Path

from nurix_agent.genie_agent import (
    MAX_REEXECUTIONS,
    AgentStreamAccumulator,
    SSEFrameParser,
    _build_result,
    _error_result,
    _finalize,
    _loose_sql_key,
    _normalize_sql,
    _columns_from_metadata,
    _rows_from_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures"
MULTISTEP = FIXTURES / "genie_agent_multistep.sse"
SINGLESTEP = FIXTURES / "genie_agent_singlestep.sse"


def _finalize_offline(acc, emit, reexecute=None, stream_error=None):
    """
    Run the real `_finalize` with the Statement Execution step stubbed out.

    `_finalize` re-executes every metadata-less sub-query, so calling it unpatched
    would hit the network (and hang). Every test goes through here so the suite stays
    deterministic and offline; `reexecute` lets a test decide what recovery does.
    """
    import nurix_agent.genie_agent as ga

    async def default_reexecute(entry, host, warehouse_id, emit):
        entry.update(
            columns=entry.get("columns") or [{"name": "a", "type": "number"}],
            rows=[["REEXECUTED"]],
            source="reexecuted",
        )

    original = ga._reexecute
    ga._reexecute = reexecute or default_reexecute
    try:
        return asyncio.run(_finalize(
            acc, emit, host="h", warehouse_id="w", stream_error=stream_error
        ))
    finally:
        ga._reexecute = original


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


# --------------------------------------------------------------------------
# BLOCKER 1 — response.failed must surface the REAL error text, and must not be
# masked by the generic "stream ended before signalling completion" message.
# --------------------------------------------------------------------------

def test_response_failed_surfaces_real_error_text():
    """A response.failed frame must set failed/error_message and emit the REAL text."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.failed", {
        "response": {
            "id": "r1", "conversation_id": "c1", "status": "failed",
            "error": {"message": "WAREHOUSE_UNAVAILABLE: warehouse is stopped"},
        },
    })
    assert acc.failed is True, "response.failed did not mark the run as failed"
    assert acc.completed is False
    assert "WAREHOUSE_UNAVAILABLE" in acc.error_message, acc.error_message
    texts = [e["text"] for e in events]
    assert any("WAREHOUSE_UNAVAILABLE: warehouse is stopped" in t for t in texts), texts
    # Captured ids survive a failure — they are how the run is traced afterwards.
    assert acc.conversation_id == "c1" and acc.response_id == "r1"


def test_response_failed_never_drops_the_error_whatever_the_shape():
    """The payload shape isn't pinned down by the probes, so every shape must yield
    non-empty text — an unrecognized shape is serialized, never dropped."""
    shapes = [
        ({"response": {"error": {"message": "real message"}}}, "real message"),
        ({"response": {"error": "flat string error"}}, "flat string error"),
        ({"error": {"error_message": "top-level nested"}}, "top-level nested"),
        ({"response": {"error": {"code": "ONLY_A_CODE"}}}, "ONLY_A_CODE"),
        ({"response": {"status_message": "status message text"}}, "status message text"),
        # No error-ish key anywhere: must serialize the payload rather than drop it.
        ({"response": {"status": "failed", "weird": {"nested": "clue"}}}, "clue"),
    ]
    for payload, expected in shapes:
        events = []
        acc = AgentStreamAccumulator(events.append)
        acc.handle_frame("response.failed", payload)
        assert acc.error_message, f"error text was dropped for payload {payload!r}"
        assert expected in acc.error_message, (
            f"expected {expected!r} in {acc.error_message!r} for payload {payload!r}"
        )
        assert events and expected in events[0]["text"], events


def test_response_failed_carries_partial_output_forward():
    """A failure frame replays whatever output it has; those sub-queries are real and
    must be captured rather than discarded with the failure."""
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.failed", {
        "response": {
            "status": "failed",
            "error": {"message": "boom"},
            "output": [
                {"type": "reasoning", "id": "r", "content": [{"text": "thought"}]},
                {"type": "function_call", "id": "f", "name": "execute_sql",
                 "arguments": json.dumps({"title": "T", "sql": "SELECT 1"})},
            ],
        },
    })
    assert acc.reasoning_count == 1
    assert len(acc.subqueries()) == 1, "failure frame threw away its replayed output"


def test_real_error_suppresses_the_generic_stream_ended_message():
    """BLOCKER 1's core symptom: with a real error present, the generic
    'ended before signalling completion' line must NOT be emitted."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.failed", {
        "response": {"status": "failed", "error": {"message": "GENIE_INTERNAL_ERROR x7"}},
    })
    result = _finalize_offline(acc, events.append)
    texts = [e["text"] for e in events]
    assert not any("ended before signalling completion" in t for t in texts), texts
    assert result["result_error"] and "GENIE_INTERNAL_ERROR x7" in result["result_error"]


def test_generic_message_still_emitted_when_there_is_no_real_error():
    """The generic message is still the right thing to say when the stream simply
    stopped with no completion and no error — that case must not regress."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.created", {"response": {"id": "r", "status": "in_progress"}})
    result = _finalize_offline(acc, events.append)
    texts = [e["text"] for e in events]
    assert any("ended before signalling completion" in t for t in texts), texts
    assert result.get("result_error") is None


def test_transport_error_is_reported_and_keeps_captured_subqueries():
    """A stream_error (transport failure) must surface as result_error AND still carry
    the sub-queries captured before it — BLOCKER 1 + BLOCKER 6 together."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.output_item.added", {
        "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                 "arguments": json.dumps({"title": "kept", "sql": "SELECT 1"})},
    })
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m1", "content": [{
            "text": "table", "metadata": {
                "sql": "SELECT 1", "columns": [{"name": "a", "type": "BIGINT"}],
                "preview_rows": [["1"]], "status": "available", "total_row_count": 1,
            },
        }]},
    })
    result = _finalize_offline(
        acc, events.append,
        stream_error="Genie agent mode failed: connection reset by peer",
    )
    assert "connection reset by peer" in result["result_error"]
    assert len(result["sub_queries"]) == 1, "transport failure discarded captured data"
    assert result["sub_queries"][0]["rows"] == [[1]]
    assert not any("ended before signalling completion" in e["text"] for e in events)


# --------------------------------------------------------------------------
# BLOCKER 1b — response.output_item.updated must be ingested like .added/.done
# --------------------------------------------------------------------------

def test_output_item_updated_is_ingested():
    """`.updated` was observed live but unhandled: an item that arrives ONLY on
    .updated must still produce its sub-query and its reasoning."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.output_item.updated", {
        "item": {"type": "reasoning", "id": "r1", "content": [{"text": "thinking out loud"}]},
    })
    acc.handle_frame("response.output_item.updated", {
        "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                 "arguments": json.dumps({"title": "Updated only", "sql": "SELECT 42"})},
    })
    assert acc.reasoning_count == 1, "reasoning on .updated was dropped"
    subs = acc.subqueries()
    assert len(subs) == 1 and subs[0]["title"] == "Updated only", subs
    assert acc.unknown_frames == 0, "a known event name was counted as unknown"


def test_updated_frame_does_not_double_emit_with_added_and_done():
    """.added/.updated/.done for the same item is still ONE emission."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    item = {"type": "reasoning", "id": "r1", "content": [{"text": "once"}]}
    for event in ("response.output_item.added",
                  "response.output_item.updated",
                  "response.output_item.done"):
        acc.handle_frame(event, {"item": dict(item)})
    assert acc.reasoning_count == 1, f"emitted {acc.reasoning_count} times, want 1"
    assert len(events) == 1, events


# --------------------------------------------------------------------------
# BLOCKER 2 — dedupe must not depend on an item id
# --------------------------------------------------------------------------

def test_id_less_reasoning_emits_exactly_once_across_added_done_and_replay():
    """An id-less reasoning item arrives on .added, .done and again in the
    response.completed replay. Before the content-hash fallback that was THREE
    thinking events for one thought."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    item = {"type": "reasoning", "content": [{"text": "no id on this one"}]}  # NO id
    acc.handle_frame("response.output_item.added", {"item": dict(item)})
    acc.handle_frame("response.output_item.done", {"item": dict(item)})
    acc.handle_frame("response.completed", {
        "response": {"status": "completed", "output": [dict(item)]},
    })
    assert acc.reasoning_count == 1, f"id-less reasoning emitted {acc.reasoning_count}x, want 1"
    matching = [e for e in events if e["text"] == "no id on this one"]
    assert len(matching) == 1, matching


def test_id_less_message_does_not_triple_the_narrative():
    """The same failure mode on a message item tripled the narrative text."""
    acc = AgentStreamAccumulator(lambda e: None)
    item = {"type": "message", "content": [{"text": "the whole answer"}]}  # NO id
    acc.handle_frame("response.output_item.added", {"item": dict(item)})
    acc.handle_frame("response.output_item.done", {"item": dict(item)})
    acc.handle_frame("response.completed", {
        "response": {"status": "completed", "output": [dict(item)]},
    })
    assert acc.narrative() == "the whole answer", repr(acc.narrative())


def test_id_less_function_call_yields_one_subquery_and_one_announcement():
    events = []
    acc = AgentStreamAccumulator(events.append)
    item = {"type": "function_call", "name": "execute_sql",
            "arguments": json.dumps({"title": "Anonymous step", "sql": "SELECT 7"})}
    for event in ("response.output_item.added",
                  "response.output_item.updated",
                  "response.output_item.done"):
        acc.handle_frame(event, {"item": dict(item)})
    assert len(acc.subqueries()) == 1
    assert len([e for e in events if e["text"] == "Querying: Anonymous step"]) == 1, events


def test_content_hash_still_separates_genuinely_different_id_less_items():
    """The fallback key must not over-collapse: two DIFFERENT id-less reasoning items
    are two thoughts, not one."""
    acc = AgentStreamAccumulator(lambda e: None)
    for text in ("first thought", "second thought"):
        acc.handle_frame("response.output_item.added", {
            "item": {"type": "reasoning", "content": [{"text": text}]},
        })
    assert acc.reasoning_count == 2


def test_function_call_output_is_never_emitted_from():
    """BLOCKER 2's soundness argument depends on this: function_call_output .added and
    .done DIFFER, so a content hash would not collapse them — safe only because
    nothing is emitted from that item type at all."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.output_item.added", {
        "item": {"type": "function_call_output", "output": "|a|b|\n|-|-|", "status": "in_progress"},
    })
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "function_call_output", "output": "|a|b|\n|-|-|\n|1|2|", "status": "completed"},
    })
    assert events == [], f"function_call_output produced output: {events}"
    assert acc.subqueries() == [] and acc.narrative() == ""


def test_added_and_done_are_byte_identical_for_the_emitted_item_types():
    """The recorded stream is the evidence for the content-hash fallback being sound:
    reasoning/function_call/message repeat verbatim; only function_call_output differs."""
    parser = SSEFrameParser()
    added, done = {}, {}
    for line in MULTISTEP.read_text().splitlines():
        for event, data in parser.feed(line):
            item = data.get("item")
            if not isinstance(item, dict):
                continue
            key = (item.get("type"), item.get("id"))
            if event.endswith(".added"):
                added[key] = item
            elif event.endswith(".done"):
                done[key] = item
    checked = 0
    for key, item in added.items():
        if key not in done:
            continue
        identical = json.dumps(item, sort_keys=True) == json.dumps(done[key], sort_keys=True)
        if key[0] == "function_call_output":
            assert not identical, f"{key} unexpectedly identical — premise changed"
        else:
            assert identical, f"{key} differs between .added and .done"
            checked += 1
    assert checked >= 3, f"only checked {checked} emitted-type items"


# --------------------------------------------------------------------------
# BLOCKER 3 — SQL normalization must be literal-aware
# --------------------------------------------------------------------------

def test_whitespace_inside_string_literals_is_significant():
    """`SELECT 'a b'` and `SELECT 'a  b'` are DIFFERENT queries. Collapsing all
    whitespace merged them onto one key and silently dropped a real sub-query."""
    assert _normalize_sql("SELECT 'a b'") != _normalize_sql("SELECT 'a  b'")
    assert _loose_sql_key("SELECT 'a b'") != _loose_sql_key("SELECT 'a  b'")


def test_whitespace_outside_literals_is_still_collapsed():
    """The other direction: Genie's leading newlines/indentation MUST still dedupe."""
    assert _normalize_sql("\n  SELECT\n    a,\n    b\n  FROM t\n") == _normalize_sql(
        "SELECT a, b FROM t"
    )
    assert _normalize_sql("SELECT\t a  FROM\n\n t") == _normalize_sql("SELECT a FROM t")


def test_literal_aware_dedup_in_both_directions_end_to_end():
    """Through the accumulator, not just the helper: differing-literal queries stay
    two sub-queries; differing-indentation queries collapse to one."""
    acc = AgentStreamAccumulator(lambda e: None)
    for i, sql in enumerate(["SELECT 'a b' FROM t", "SELECT 'a  b' FROM t"]):
        acc.handle_frame("response.output_item.added", {
            "item": {"type": "function_call", "id": f"f{i}", "name": "execute_sql",
                     "arguments": json.dumps({"title": f"t{i}", "sql": sql})},
        })
    assert len(acc.subqueries()) == 2, "two distinct literals were wrongly deduped"

    acc2 = AgentStreamAccumulator(lambda e: None)
    for i, sql in enumerate(["\n\n    SELECT a\n    FROM t\n", "SELECT a FROM t"]):
        acc2.handle_frame("response.output_item.added", {
            "item": {"type": "function_call", "id": f"g{i}", "name": "execute_sql",
                     "arguments": json.dumps({"title": f"t{i}", "sql": sql})},
        })
    assert len(acc2.subqueries()) == 1, "indentation-only variants were not deduped"


def test_normalization_preserves_quoted_identifiers_and_handles_escapes():
    # Double-quoted and backtick identifiers are whitespace-significant too.
    assert _normalize_sql('SELECT "a b" FROM t') != _normalize_sql('SELECT "a  b" FROM t')
    assert _normalize_sql("SELECT `a b` FROM t") != _normalize_sql("SELECT `a  b` FROM t")
    # Doubled quotes ('') and backslash escapes (\') must not end the literal early —
    # if they did, the scanner would treat following whitespace as collapsible.
    assert _normalize_sql("SELECT 'it''s  here'  FROM t") == "SELECT 'it''s  here' FROM t"
    assert _normalize_sql("SELECT 'it\\'s  here'  FROM t") == "SELECT 'it\\'s  here' FROM t"
    # An unterminated literal must not throw or swallow the rest of the string.
    assert "abc" in _normalize_sql("SELECT 'abc")


def test_loose_key_strips_comments_and_semicolons_but_not_inside_literals():
    assert _loose_sql_key("SELECT a FROM t;") == _loose_sql_key("select A from T")
    assert _loose_sql_key("SELECT a -- why\nFROM t") == _loose_sql_key("SELECT a FROM t")
    assert _loose_sql_key("SELECT a /* note */ FROM t") == _loose_sql_key("SELECT a FROM t")
    # A `--` INSIDE a string is data, not a comment: it must survive.
    assert "--not a comment" in _loose_sql_key("SELECT '--not a comment' FROM t")
    # Literal case is data and must NOT be folded.
    assert _loose_sql_key("SELECT 'Ab'") != _loose_sql_key("SELECT 'ab'")


def test_normalize_is_a_pure_function_of_its_input():
    sql = "SELECT 'x  y' /* c */ FROM t"
    assert _normalize_sql(sql) == _normalize_sql(sql)
    assert _normalize_sql("") == "" and _normalize_sql(None) == ""


# --------------------------------------------------------------------------
# BLOCKER 4 — cap-skipped and gather-exception sub-queries must be surfaced
# --------------------------------------------------------------------------

def _acc_with_pending(n: int):
    """An accumulator holding `n` distinct metadata-less (=pending) sub-queries."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    for i in range(n):
        acc.handle_frame("response.output_item.added", {
            "item": {"type": "function_call", "id": f"f{i}", "name": "execute_sql",
                     "arguments": json.dumps({"title": f"q{i}", "sql": f"SELECT {i} FROM t"})},
        })
    return acc, events


def test_cap_skipped_subqueries_emit_ONE_aggregate_notice():
    """Entries beyond MAX_REEXECUTIONS got an internal `error` field and NO emit —
    invisible loss. Now: marked skipped and announced once, not once per entry."""
    over = 3
    acc, _ = _acc_with_pending(MAX_REEXECUTIONS + over)
    events = []

    async def fake_reexecute(entry, host, warehouse_id, emit):
        entry.update(columns=[{"name": "a", "type": "number"}], rows=[[1]], source="reexecuted")

    result = _finalize_offline(acc, events.append, reexecute=fake_reexecute)

    skipped = [s for s in result["sub_queries"] if s["source"] == "skipped"]
    assert len(skipped) == over, f"expected {over} skipped, got {len(skipped)}"
    assert all(s["error"] for s in skipped), "skipped entries lost their reason"
    cap_notices = [e for e in events if "recovery cap" in e["text"]]
    assert len(cap_notices) == 1, f"expected ONE aggregate cap notice, got {cap_notices}"
    assert str(over) in cap_notices[0]["text"], cap_notices[0]["text"]
    assert cap_notices[0]["type"] == "thinking"
    # And the ones within the cap were actually recovered.
    assert len([s for s in result["sub_queries"] if s["source"] == "reexecuted"]) == MAX_REEXECUTIONS


def test_cap_notice_is_singular_for_exactly_one_skipped_entry():
    acc, _ = _acc_with_pending(MAX_REEXECUTIONS + 1)
    events = []
    _finalize_offline(acc, events.append)
    notice = [e["text"] for e in events if "recovery cap" in e["text"]]
    assert len(notice) == 1 and "1 further sub-query was not fetched" in notice[0], notice


def test_gather_exception_marks_the_entry_and_emits_real_text():
    """asyncio.gather(..., return_exceptions=True) results were never inspected, so an
    exception escaping _reexecute vanished silently."""
    acc, _ = _acc_with_pending(2)
    events = []

    async def exploding_reexecute(entry, host, warehouse_id, emit):
        if entry["title"] == "q0":
            raise MemoryError("native allocator gave up")
        entry.update(columns=[{"name": "a", "type": "number"}], rows=[[1]], source="reexecuted")

    result = _finalize_offline(acc, events.append, reexecute=exploding_reexecute)

    by_title = {s["title"]: s for s in result["sub_queries"]}
    assert by_title["q0"]["source"] == "error", by_title["q0"]
    assert "native allocator gave up" in by_title["q0"]["error"], by_title["q0"]["error"]
    assert "MemoryError" in by_title["q0"]["error"]
    texts = [e["text"] for e in events]
    assert any("native allocator gave up" in t and "q0" in t for t in texts), texts
    # The sibling sub-query still succeeded — one failure must not sink the batch.
    assert by_title["q1"]["source"] == "reexecuted" and by_title["q1"]["rows"] == [[1]]


# --------------------------------------------------------------------------
# BLOCKER 5 — malformed / unknown / data-only frames must be counted and reported
# --------------------------------------------------------------------------

def test_discarded_frames_are_counted_not_silently_dropped():
    counted = []
    p = SSEFrameParser(on_discard=counted.append)
    p.feed("event:response.created")
    p.feed("data:{not json")
    p.feed("")
    p.feed("event:response.created")
    p.feed('data:"a bare string, not an object"')
    p.feed("")
    assert p.discarded_frames == 2, p.discarded_frames
    assert len(counted) == 2, counted


def test_malformed_frame_does_not_kill_the_stream():
    """A bad frame in the middle must not stop the good frames after it."""
    p = SSEFrameParser()
    out = []
    for line in [
        "event:response.output_item.added", "data:{broken", "",
        "event:response.output_item.added",
        'data:{"item":{"type":"reasoning","id":"r","content":[{"text":"survived"}]}}', "",
    ]:
        out += p.feed(line)
    assert len(out) == 1 and out[0][1]["item"]["content"][0]["text"] == "survived"
    assert p.discarded_frames == 1


def test_unknown_event_names_are_counted_with_their_names():
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.something.brand_new", {"response": {}})
    acc.handle_frame("response.output_item.added", {"item": "not a dict"})
    assert acc.unknown_frames == 2, acc.unknown_frames
    assert any("brand_new" in n for n in acc.unknown_event_names), acc.unknown_event_names


def test_frame_loss_produces_ONE_aggregate_thinking_event_after_the_stream():
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.note_discarded("unparseable JSON payload")
    acc.note_discarded("unparseable JSON payload")
    acc.handle_frame("response.mystery", {"response": {}})
    acc.completed = True

    loss = acc.frame_loss_message()
    assert loss and "3 stream frames could not be used" in loss, loss
    assert "2 unparseable" in loss and "1 unrecognized" in loss and "response.mystery" in loss

    _finalize_offline(acc, events.append)
    notices = [e for e in events if "could not be used" in e["text"]]
    assert len(notices) == 1, f"expected ONE aggregate loss notice, got {notices}"
    assert notices[0]["type"] == "thinking"


def test_no_loss_means_no_loss_notice():
    acc, events = _replay(MULTISTEP)[0], []
    assert acc.frame_loss_message() is None, "clean stream reported phantom frame loss"
    _finalize_offline(acc, events.append)
    assert not any("could not be used" in e["text"] for e in events)


def test_data_only_frame_is_routed_by_payload_shape():
    """A frame with `data:` but no `event:` must be routed on its shape, not dropped."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    # Shape 1: carries an item -> item handler.
    acc.handle_frame("", {
        "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                 "arguments": json.dumps({"title": "shape routed", "sql": "SELECT 1"})},
    })
    # Shape 2: carries a response -> response handler, status says which frame it is.
    acc.handle_frame("", {"response": {"id": "r9", "conversation_id": "c9", "status": "completed"}})
    subs = acc.subqueries()
    assert len(subs) == 1 and subs[0]["title"] == "shape routed", subs
    assert acc.completed is True, "data-only completed frame was not recognized"
    assert acc.response_id == "r9" and acc.conversation_id == "c9"
    assert acc.unknown_frames == 0, "shape-routed frames were counted as unknown"


def test_data_only_failed_frame_is_recognized_by_status():
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("", {
        "response": {"status": "failed", "error": {"message": "data-only failure"}},
    })
    assert acc.failed is True and "data-only failure" in acc.error_message
    assert acc.unknown_frames == 0


def test_unroutable_data_only_frame_is_counted_as_unknown():
    """No event name AND no recognizable shape: nothing to do but count it."""
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("", {"totally": "unfamiliar"})
    assert acc.unknown_frames == 1


def test_parser_does_not_count_payloadless_event_frames_as_loss():
    """A bare `event:` keep-alive carries nothing to lose and must not be reported."""
    p = SSEFrameParser()
    assert p.feed("event:ping") == []
    assert p.feed("") == []
    assert p.discarded_frames == 0


# --------------------------------------------------------------------------
# BLOCKER 6 — late failures keep captured data; orphan metadata neither
# duplicates nor charts partial rows
# --------------------------------------------------------------------------

def test_error_result_preserves_captured_subqueries():
    """`_error_result` forced sub_queries=[] despite its docstring — on a
    sole-delivery stream that data is then gone for good."""
    acc, _, _ = _replay(MULTISTEP)
    result = _error_result("late failure", acc)
    assert result["result_error"] == "late failure"
    assert len(result["sub_queries"]) == 5, (
        f"error result dropped captured sub-queries: {len(result['sub_queries'])}"
    )
    assert any(s["rows"] for s in result["sub_queries"]), "kept the queries but lost the rows"
    # The narrative and the featured result survive too.
    assert result["text"], "error result dropped the narrative"
    assert result["columns"] and result["rows"]


def test_error_result_uses_the_resolved_subqueries_it_is_handed():
    """The re-executed list must win over a re-derived one, so recovery work done
    before a late failure is not thrown away."""
    acc, _, _ = _replay(MULTISTEP)
    resolved = acc.subqueries()
    for entry in resolved:
        if entry["source"] == "pending":
            entry.update(columns=[{"name": "x", "type": "number"}], rows=[[9]],
                         source="reexecuted")
    result = _error_result("late failure", acc, resolved)
    assert all(s["rows"] for s in result["sub_queries"]), "re-executed rows were discarded"
    assert len([s for s in result["sub_queries"] if s["source"] == "reexecuted"]) == 3


def test_orphan_metadata_is_not_duplicated_when_it_matches_cosmetically():
    """An orphan is declared only after a lenient match fails. Otherwise SQL that
    differs from its function_call by a semicolon/case/comment becomes BOTH a
    re-executed sub-query AND an orphan chart — the same data charted twice."""
    for variant in (
        "select a from t",              # case only
        "SELECT a FROM t;",             # trailing semicolon
        "SELECT a FROM t  ",            # trailing whitespace
        "SELECT a /* hi */ FROM t",     # comment
        "SELECT a -- hi\nFROM t",       # line comment
    ):
        acc = AgentStreamAccumulator(lambda e: None)
        acc.handle_frame("response.output_item.added", {
            "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                     "arguments": json.dumps({"title": "T", "sql": "SELECT a FROM t"})},
        })
        acc.handle_frame("response.output_item.done", {
            "item": {"type": "message", "id": "m1", "content": [{
                "text": "table", "metadata": {
                    "sql": variant, "columns": [{"name": "a", "type": "BIGINT"}],
                    "preview_rows": [["1"]], "status": "available", "total_row_count": 1,
                },
            }]},
        })
        assert acc.orphan_metadata() == [], f"variant {variant!r} wrongly became an orphan"

    # A genuinely different query IS still an orphan — the lenient match must not
    # over-match and swallow a real extra result set.
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.added", {
        "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                 "arguments": json.dumps({"title": "T", "sql": "SELECT a FROM t"})},
    })
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m1", "content": [{
            "text": "table", "metadata": {
                "sql": "SELECT b FROM other", "columns": [{"name": "b", "type": "BIGINT"}],
                "preview_rows": [["1"]], "status": "available", "total_row_count": 1,
            },
        }]},
    })
    assert len(acc.orphan_metadata()) == 1


def _finalize_with_orphan(md: dict):
    """Run _finalize over an accumulator whose ONLY content is one orphan metadata
    block, with re-execution stubbed out. Returns (result, events)."""
    events = []
    acc = AgentStreamAccumulator(events.append)
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m1", "content": [{"text": "t", "metadata": md}]},
    })
    acc.completed = True
    return _finalize_offline(acc, events.append), events


def test_orphan_metadata_with_partial_rows_is_not_charted_as_complete():
    """Orphans must get the SAME completeness check subqueries() applies — an orphan
    must not be the one place partial rows sneak through to a chart."""
    result, _ = _finalize_with_orphan({
        "sql": "SELECT a FROM orphan",
        "columns": [{"name": "a", "type": "BIGINT"}],
        "preview_rows": [["1"], ["2"]],
        "status": "available",
        "total_row_count": 900,          # preview covers 2 of 900 -> NOT complete
    })
    orphan = [s for s in result["sub_queries"] if "orphan" in s["sql"]]
    assert len(orphan) == 1, f"orphan lost or duplicated: {result['sub_queries']}"
    assert orphan[0]["source"] == "reexecuted", (
        f"partial orphan rows were charted as complete: {orphan[0]}"
    )
    assert orphan[0]["rows"] == [["REEXECUTED"]], "partial preview rows leaked into the chart"


def test_orphan_metadata_that_is_complete_is_charted_directly():
    result, _ = _finalize_with_orphan({
        "sql": "SELECT a FROM orphan",
        "columns": [{"name": "a", "type": "BIGINT"}],
        "preview_rows": [["1"], ["2"]],
        "status": "available",
        "total_row_count": 2,            # preview covers all of it
    })
    orphan = [s for s in result["sub_queries"] if "orphan" in s["sql"]]
    assert len(orphan) == 1 and orphan[0]["source"] == "metadata", orphan
    assert orphan[0]["rows"] == [[1], [2]], orphan[0]["rows"]


def test_orphan_metadata_appears_exactly_once_in_the_result():
    """Guards the duplication half of BLOCKER 6 end-to-end: one orphan block must
    yield one sub-query, never one re-executed AND one charted copy."""
    result, _ = _finalize_with_orphan({
        "sql": "SELECT a FROM orphan",
        "columns": [{"name": "a", "type": "BIGINT"}],
        "preview_rows": [["1"]],
        "status": "available",
        "total_row_count": 1,
    })
    assert len(result["sub_queries"]) == 1, result["sub_queries"]


def test_rowless_orphan_without_sql_is_reported_rather_than_silently_dropped():
    result, events = _finalize_with_orphan({
        "sql": "",                       # no SQL -> cannot be re-executed
        "columns": [{"name": "a", "type": "BIGINT"}],
        "preview_rows": [["1"], ["2"]],
        "status": "available",
        "total_row_count": 40,           # incomplete AND unrunnable
    })
    assert result["sub_queries"] == [], result["sub_queries"]
    assert any("was not charted" in e["text"] for e in events), [e["text"] for e in events]


# --------------------------------------------------------------------------
# Non-blocking: unknown total_row_count is not proof of completeness; a
# non-success `status` is respected.
# --------------------------------------------------------------------------

def _one_metadata_subquery(md_extra: dict, rows=(["1"],)):
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.added", {
        "item": {"type": "function_call", "id": "f1", "name": "execute_sql",
                 "arguments": json.dumps({"title": "T", "sql": "SELECT a FROM t"})},
    })
    md = {"sql": "SELECT a FROM t", "columns": [{"name": "a", "type": "BIGINT"}],
          "preview_rows": [list(r) for r in rows]}
    md.update(md_extra)
    acc.handle_frame("response.output_item.done", {
        "item": {"type": "message", "id": "m1", "content": [{"text": "t", "metadata": md}]},
    })
    return acc.subqueries()[0]


def test_unknown_total_row_count_is_complete_only_when_rows_are_present():
    # Unknown total + rows present -> chartable.
    assert _one_metadata_subquery({"status": "available"})["source"] == "metadata"
    # Unknown total + NO rows -> tells us nothing, so re-execute rather than chart empty.
    assert _one_metadata_subquery({"status": "available"}, rows=())["source"] == "pending"
    # A non-int total is not a count, so it falls back to the rows-present rule. A bool
    # is an int in Python and must NOT be mistaken for a row count of 1.
    assert _one_metadata_subquery({"total_row_count": True})["source"] == "metadata"
    assert _one_metadata_subquery({"total_row_count": True}, rows=())["source"] == "pending"
    assert _one_metadata_subquery({"total_row_count": "40"}, rows=())["source"] == "pending"
    assert _one_metadata_subquery({"total_row_count": "40"})["source"] == "metadata"


def test_non_success_metadata_status_blocks_charting():
    """`status` was ignored entirely; anything that is not a success value must not be
    charted as a finished result set."""
    for status in ("in_progress", "failed", "cancelled", "PENDING"):
        sq = _one_metadata_subquery({"status": status, "total_row_count": 1})
        assert sq["source"] == "pending", f"status={status!r} was charted anyway: {sq}"
    for status in ("available", "AVAILABLE", " succeeded ", "completed"):
        sq = _one_metadata_subquery({"status": status, "total_row_count": 1})
        assert sq["source"] == "metadata", f"status={status!r} was rejected: {sq}"


def test_featured_result_is_the_first_subquery_with_rows():
    """_build_result's contract with the plain path: {text, sql, columns, rows}."""
    acc = AgentStreamAccumulator(lambda e: None)
    subs = [
        {"title": "empty", "sql": "SELECT 0", "columns": [], "rows": [], "source": "error"},
        {"title": "has data", "sql": "SELECT 1",
         "columns": [{"name": "a", "type": "number"}], "rows": [[1]], "source": "metadata"},
    ]
    result = _build_result(acc, subs)
    assert result["sql"] == "SELECT 1" and result["rows"] == [[1]]
    assert result["sub_queries"] == subs
    assert set(result) >= {"text", "sql", "columns", "rows", "sub_queries",
                           "reasoning_count", "conversation_id", "response_id"}


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
