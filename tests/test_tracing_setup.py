"""
Plain-assert checks for MLflow tracing setup and span-attribute redaction.

Two classes of bug are pinned down here:

  1. The ORIGINAL BUG — no tracking URI plus a bare experiment name, wrapped in
     `except Exception: pass`, which sent every trace to a local ./mlruns directory
     and reported nothing. `init()` must now refuse a bare name loudly and record a
     readable `last_error`.
  2. CUSTOMER DATA — the traced payloads are customer feedback verbatims, so free
     text entering a span must be truncated and row data must never be recorded.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nurix_agent import tracing
from nurix_agent.config import AppConfig


class _Cfg:
    def __init__(self, uri, experiment):
        self.mlflow_tracking_uri = uri
        self.mlflow_experiment = experiment


def test_default_config_is_absolute_path_and_databricks_uri():
    """The defaults themselves must not reproduce the original misconfiguration."""
    cfg = AppConfig()
    assert cfg.mlflow_experiment.startswith("/"), (
        f"experiment default {cfg.mlflow_experiment!r} must be an ABSOLUTE workspace path; "
        "a bare name is what sent traces to a local ./mlruns directory"
    )
    assert cfg.mlflow_tracking_uri == "databricks", cfg.mlflow_tracking_uri
    print(f"PASS defaults are safe: uri={cfg.mlflow_tracking_uri} experiment={cfg.mlflow_experiment}")


def test_bare_experiment_name_is_refused_with_a_readable_error():
    """The exact original bug: a bare name must NOT silently 'work'."""
    status = tracing.init(_Cfg("databricks", "nurix-agent-traces"))
    assert status["enabled"] is False
    assert status["last_error"], "a refusal with no error text is the silent failure we removed"
    assert "absolute" in status["last_error"].lower(), status["last_error"]
    print(f"PASS bare experiment name refused: {status['last_error'][:80]}...")


def test_explicitly_disabled_uri_reports_why():
    for uri in ("", "none", "off", "disabled"):
        status = tracing.init(_Cfg(uri, "/Shared/whatever"))
        assert status["enabled"] is False, uri
        assert "disabled by configuration" in (status["last_error"] or ""), status
    print("PASS empty/none/off/disabled tracking URIs disable tracing with a stated reason")


def test_status_reports_all_documented_keys():
    """GET /mlflow_status's response shape is a contract; keep every key present."""
    status = tracing.init(_Cfg("databricks", "bare-name"))
    for key in ("enabled", "tracking_uri", "experiment", "experiment_id", "last_error"):
        assert key in status, f"missing {key}"
    # status() must be a COPY: a caller mutating the response must not corrupt state.
    status["enabled"] = "tampered"
    assert tracing.status()["enabled"] != "tampered", "status() leaked its internal dict"
    print("PASS status() exposes all five documented keys and returns a copy")


def test_truncate_caps_text_and_marks_the_cut():
    assert tracing.truncate("") == ""
    assert tracing.truncate(None) == ""
    assert tracing.truncate("short") == "short"

    long_text = "x" * 900
    out = tracing.truncate(long_text)
    assert len(out) < len(long_text)
    assert out.startswith("x" * tracing.MAX_SPAN_TEXT)
    # The marker matters: a silently clipped excerpt reads like a short model answer.
    assert "truncated" in out and "900" in out, out
    assert tracing.MAX_SPAN_TEXT <= 500, "customer verbatims must stay capped at <=500 chars"
    print(f"PASS truncate() caps at {tracing.MAX_SPAN_TEXT} chars and marks the cut")


def test_truncate_coerces_non_strings():
    assert tracing.truncate(1234) == "1234"
    print("PASS truncate() coerces non-string values")


def test_column_names_records_names_only_never_rows():
    columns = [{"name": "sentiment", "type": "string"}, {"name": "n", "type": "number"}]
    names = tracing.column_names(columns)
    assert names == ["sentiment", "n"], names
    # Types/rows are deliberately NOT part of the recorded attribute.
    assert "string" not in names
    assert tracing.column_names(None) == []
    assert tracing.column_names(["raw_name"]) == ["raw_name"]
    print("PASS column_names() records schema names only")


if __name__ == "__main__":
    test_default_config_is_absolute_path_and_databricks_uri()
    test_bare_experiment_name_is_refused_with_a_readable_error()
    test_explicitly_disabled_uri_reports_why()
    test_status_reports_all_documented_keys()
    test_truncate_caps_text_and_marks_the_cut()
    test_truncate_coerces_non_strings()
    test_column_names_records_names_only_never_rows()
    print("\n7 tracing-setup tests passed")
