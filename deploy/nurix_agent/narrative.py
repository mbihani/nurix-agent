"""Deterministic cleanup for Genie narrative text shown in chat."""

import re


# Observed in the deployed Agent response on 2026-08-06 as, for example:
# \[[1](https://.../genie/rooms/.../chats/...?o=...&gra_focus=...)\]
# Keep this deliberately narrow: ordinary Markdown links and bracketed numbers may
# be meaningful narrative content and must not be treated as citations.
_GENIE_CITATION = re.compile(
    r"[ \t]*\\\[\[\d+\]\("
    r"https://[^\s)]+/genie/rooms/[^\s/)]+/chats/[^\s?)]+"
    r"\?[^\s)]*\bgra_focus=[^\s)&]+[^\s)]*"
    r"\)\\\]"
)

_TRAILING_REFERENCE_HEADING = re.compile(
    r"\n{2,}(?:#{1,6}[ \t]+)?(?:sources|references|footnotes):?[ \t]*\Z",
    re.IGNORECASE,
)


def _clean_non_code(text: str) -> str:
    """Remove observed Genie citations from a span known not to be code."""
    cleaned = _GENIE_CITATION.sub("", text)
    # Citation removal can expose whitespace immediately before punctuation.
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    return cleaned


def clean_genie_narrative(text: str) -> str:
    """Strip observed Genie citation links without altering inline code spans."""
    if not text:
        return text

    # Backtick code spans are opaque. Splitting (instead of one broad regex over
    # the narrative) also protects bracket-like examples shown as code.
    parts = re.split(r"(`+[^\n]*?`+)", text)
    for index in range(0, len(parts), 2):
        parts[index] = _clean_non_code(parts[index])
    cleaned = "".join(parts).strip()
    # A sources heading whose entire body consisted of citations is stranded once
    # those links are removed. Only remove it at the very end of the narrative.
    return _TRAILING_REFERENCE_HEADING.sub("", cleaned).rstrip()
