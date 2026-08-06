"""Plain-assert checks for deterministic Genie narrative cleanup."""

from nurix_agent.narrative import clean_genie_narrative


CITE_1 = (
    r"\[[1](https://fevm-stable-classic-7ppxjq.cloud.databricks.com/"
    r"genie/rooms/room-id/chats/chat-id?o=7474660648944264&gra_focus=focus-1)\]"
)
CITE_2 = (
    r"\[[2](https://fevm-stable-classic-7ppxjq.cloud.databricks.com/"
    r"genie/rooms/room-id/chats/chat-id?o=7474660648944264&gra_focus=focus-2)\]"
)


def test_removes_observed_inline_citations_and_cleans_spacing():
    text = f"Finding {CITE_1}{CITE_2} . Next sentence."
    assert clean_genie_narrative(text) == "Finding. Next sentence."
    assert " \n" not in clean_genie_narrative(f"Finding {CITE_1}\n\nNext")


def test_removes_stranded_trailing_sources_section():
    text = f"A sourced finding.\n\n### Sources\n{CITE_1}{CITE_2}"
    assert clean_genie_narrative(text) == "A sourced finding."


def test_preserves_legitimate_bracketed_content_and_links():
    text = (
        "Keep [1] because it is data, [draft] as prose, and "
        "[the report](https://example.com/report)."
    )
    assert clean_genie_narrative(text) == text


def test_preserves_brackets_and_citation_shape_inside_code_spans():
    text = f"Keep `[1]`, `[...]`, and `{CITE_1}` but remove this {CITE_2}."
    expected = f"Keep `[1]`, `[...]`, and `{CITE_1}` but remove this."
    assert clean_genie_narrative(text) == expected


if __name__ == "__main__":
    test_removes_observed_inline_citations_and_cleans_spacing()
    test_removes_stranded_trailing_sources_section()
    test_preserves_legitimate_bracketed_content_and_links()
    test_preserves_brackets_and_citation_shape_inside_code_spans()
    print("4 narrative cleanup tests passed")
