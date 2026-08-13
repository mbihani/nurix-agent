"""
Plain-assert checks for the ADDITIVE streaming event contract.

The frontend is wired independently, so the contract these tests pin down is:
a client that IGNORES the new `*_delta` events must see byte-identical behaviour
to before streaming existed. Every test here exists to catch a regression that
would break that promise.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nurix_agent.genie_agent import AgentStreamAccumulator
from nurix_agent.nodes.visualizer import _chunk_text, _stream_text_deltas


class _FakeChunk:
    def __init__(self, content):
        self.content = content


class _FakeStreamingLLM:
    """Minimal ChatOpenAI stand-in exposing only `astream`."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.astream_calls = 0

    async def astream(self, messages):
        self.astream_calls += 1
        for c in self._chunks:
            yield _FakeChunk(c)


def test_chunk_text_handles_string_and_block_shapes():
    assert _chunk_text("hello") == "hello"
    assert _chunk_text(None) == ""
    # Non-text blocks (reasoning/tool) are not narrative and must not leak in.
    assert _chunk_text([{"type": "reasoning", "text": "hidden"}, {"type": "text", "text": "shown"}]) == "shown"
    # A block dict without an explicit type is still treated as text.
    assert _chunk_text([{"text": "bare"}]) == "bare"
    print("PASS _chunk_text normalizes string / block / mixed content shapes")


def test_chunk_text_joins_blocks_with_a_space_for_terminal_parity():
    """
    The list branch MUST join with " ", matching the pre-streaming `" ".join(...)`.

    The reachable path is string-only (measured: 59/59 chunks were `str`), so this
    separator is inert in practice — which is exactly why parity wins. If the supported
    list shape is ever reached, the TERMINAL `insight` text must not change from what
    the pre-streaming code would have produced; that terminal event is what a client
    ignoring deltas sees, and changing it breaks the additive contract.
    """
    assert _chunk_text([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]) == "first second"
    assert _chunk_text(["first", "second"]) == "first second"
    # A single block must not gain a stray separator.
    assert _chunk_text([{"type": "text", "text": "only"}]) == "only"
    # Skipped non-text blocks must not leave a doubled separator behind.
    assert _chunk_text([
        {"type": "text", "text": "a"},
        {"type": "reasoning", "text": "drop me"},
        {"type": "text", "text": "b"},
    ]) == "a b"
    print("PASS _chunk_text joins list blocks with ' ' (pre-streaming terminal parity)")


def test_stream_text_deltas_emits_pieces_and_returns_full_text():
    llm = _FakeStreamingLLM(["The ", "top ", "issue ", "is latency."])
    seen: list[str] = []
    full = asyncio.run(_stream_text_deltas(llm, [{"role": "user", "content": "q"}], seen.append))

    assert llm.astream_calls == 1, "must stream, not invoke"
    assert seen == ["The ", "top ", "issue ", "is latency."], seen
    # The accumulated return value is what the TERMINAL event carries, and it must
    # equal the concatenation of the deltas.
    assert full == "The top issue is latency."
    assert "".join(seen) == full
    print("PASS _stream_text_deltas streams per-chunk and returns the full text")


def test_stream_text_deltas_skips_empty_chunks():
    """Empty/None chunks must not produce empty delta events (noise on the wire)."""
    llm = _FakeStreamingLLM(["a", "", None, [], "b"])
    seen: list[str] = []
    full = asyncio.run(_stream_text_deltas(llm, [], seen.append))
    assert seen == ["a", "b"], seen
    assert full == "ab"
    print("PASS _stream_text_deltas suppresses empty chunks")


def _message_item(parts, item_id="msg-1"):
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "content": [{"type": "output_text", "text": t} for t in parts],
    }


def test_narrative_parts_concatenate_to_narrative():
    """
    The delta pieces must reproduce narrative() exactly when concatenated.

    This is the invariant that lets a delta-rendering client end up with the same
    text as a client that only reads the terminal event.
    """
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.done", {"item": _message_item(["First para.", "Second para."])})

    assert acc.narrative() == "First para.\n\nSecond para."
    parts = acc.narrative_parts()
    assert parts == ["First para.", "\n\nSecond para."], parts
    assert "".join(parts) == acc.narrative(), "deltas must rebuild the narrative exactly"
    print("PASS narrative_parts() concatenates back to narrative() exactly")


def test_narrative_parts_empty_when_no_narrative():
    acc = AgentStreamAccumulator(lambda e: None)
    assert acc.narrative_parts() == []
    assert acc.narrative() == ""
    print("PASS narrative_parts() is empty when no narrative arrived")


def test_narrative_parts_skip_blank_parts():
    acc = AgentStreamAccumulator(lambda e: None)
    acc.handle_frame("response.output_item.done", {"item": _message_item(["Real text.", "   "])})
    parts = acc.narrative_parts()
    assert parts == ["Real text."], parts
    assert "".join(parts) == acc.narrative()
    print("PASS narrative_parts() drops whitespace-only parts")


def test_genie_stream_carries_no_text_delta_events():
    """
    Documents the PLATFORM LIMITATION verified against the live surface: the
    recorded streams contain no text-delta event, and the final message item
    arrives already complete on `.added`.

    If Genie ever starts emitting deltas this test will still pass (it asserts on
    the recorded fixtures), but the fixture refresh that adds them is exactly the
    moment to revisit forwarding them.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    for name in ("genie_agent_singlestep.sse", "genie_agent_multistep.sse"):
        raw = (fixtures / name).read_text()
        event_names = {
            line.split(":", 1)[1].strip()
            for line in raw.splitlines()
            if line.startswith("event:")
        }
        assert event_names, f"{name}: no event names parsed"
        deltas = [e for e in event_names if "delta" in e.lower()]
        assert not deltas, f"{name}: unexpected delta events {deltas}"
        print(f"PASS {name}: no text-delta events (names: {sorted(event_names)})")


if __name__ == "__main__":
    test_chunk_text_handles_string_and_block_shapes()
    test_chunk_text_joins_blocks_with_a_space_for_terminal_parity()
    test_stream_text_deltas_emits_pieces_and_returns_full_text()
    test_stream_text_deltas_skips_empty_chunks()
    test_narrative_parts_concatenate_to_narrative()
    test_narrative_parts_empty_when_no_narrative()
    test_narrative_parts_skip_blank_parts()
    test_genie_stream_carries_no_text_delta_events()
    print("\n8 streaming-contract tests passed")
