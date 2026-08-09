"""
MLflow tracing setup and the redaction rules for what may enter a span.

Why this module exists
----------------------
Tracing used to be two `try: ... except Exception: pass` lines in api.py with no
tracking URI at all. That combination is the worst of both worlds: MLflow silently
defaults to a LOCAL `./mlruns` directory on the app container's ephemeral disk, so
every trace was written somewhere nobody could read and vanished on restart — and
because both calls swallowed their exception, there was nothing in the app log to
say so.

So the setup here is explicit about three things:

  1. The tracking URI is SET (default "databricks"), never left to default to a
     local directory. It stays configurable so a local run can opt out.
  2. The experiment name is an ABSOLUTE WORKSPACE PATH. Databricks rejects a bare
     name like "nurix-agent-traces"; it must look like "/Shared/nurix-agent-traces".
  3. Every failure is LOGGED with the real error and recorded in `status()`.
     Tracing is observability, not a hard dependency, so the app still starts — but
     the failure is visible in the app log and over GET /mlflow_status instead of
     being swallowed.

Customer data
-------------
The traced payloads are customer feedback. Span attributes therefore carry COUNTS,
column names, and SQL — never full verbatim text and never row data. `truncate()`
is the single chokepoint for any free text that does go in; use it rather than
slicing at call sites, so the cap cannot drift between nodes.
"""
import logging

import mlflow

logger = logging.getLogger(__name__)

# Free text entering a span attribute is capped at this many characters. The
# payloads are customer feedback verbatims, so a span is allowed a recognizable
# excerpt for debugging and nothing more.
MAX_SPAN_TEXT = 500

# Tracking-URI values that mean "do not trace at all". Lets a local run opt out
# via MLFLOW_TRACKING_URI="" (or "none"/"off"/"disabled") without code changes,
# instead of silently filling ./mlruns.
_DISABLED_URIS = frozenset({"", "none", "off", "disabled", "false"})

_state: dict = {
    "enabled": False,
    "tracking_uri": None,
    "experiment": None,
    "experiment_id": None,
    "last_error": None,
}


def truncate(text, limit: int = MAX_SPAN_TEXT) -> str:
    """
    Span-safe rendering of free text: coerced to str, capped, and marked when cut.

    The ellipsis suffix matters — an excerpt that silently ends mid-sentence reads
    like the model returned a short answer, which sends whoever is reading the
    trace after the wrong bug.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


def column_names(columns) -> list[str]:
    """
    Column NAMES only, for span attributes.

    Names are schema, not customer data, so they are safe to record in full and are
    what makes a trace useful. Rows are deliberately never recorded.
    """
    out: list[str] = []
    for c in columns or []:
        if isinstance(c, dict):
            out.append(str(c.get("name", "")))
        else:
            out.append(str(c))
    return out


def init(cfg) -> dict:
    """
    Point MLflow at the workspace and select the experiment.

    Order is deliberate: tracking URI first (so `set_experiment` resolves against
    the workspace rather than a local directory), then the experiment, and
    `mlflow.langchain.autolog()` LAST so autologged traces are already bound to the
    right destination.

    Never raises. Every failure mode is logged with the real error text and left
    readable through `status()`.
    """
    uri = (cfg.mlflow_tracking_uri or "").strip()
    experiment = (cfg.mlflow_experiment or "").strip()
    _state.update(tracking_uri=uri, experiment=experiment, enabled=False,
                  experiment_id=None, last_error=None)

    if uri.lower() in _DISABLED_URIS:
        msg = f"MLflow tracing disabled by configuration (MLFLOW_TRACKING_URI={uri!r})"
        logger.warning(msg)
        _state["last_error"] = msg
        return status()

    # A bare experiment name is the exact misconfiguration that sent every trace to
    # a local ./mlruns directory, so it is refused loudly rather than "working".
    if uri == "databricks" and not experiment.startswith("/"):
        msg = (
            f"MLflow tracing disabled: experiment {experiment!r} is not an absolute "
            f"workspace path. Databricks requires a path like '/Shared/nurix-agent-traces'."
        )
        logger.warning(msg)
        _state["last_error"] = msg
        return status()

    try:
        mlflow.set_tracking_uri(uri)
    except Exception as e:
        logger.warning("MLflow tracing disabled: could not set tracking URI %r: %s", uri, e)
        _state["last_error"] = f"set_tracking_uri({uri!r}) failed: {e}"
        return status()

    try:
        exp = mlflow.set_experiment(experiment)
        _state["experiment_id"] = getattr(exp, "experiment_id", None)
    except Exception as e:
        logger.warning("MLflow tracing disabled: %s", e)
        _state["last_error"] = f"set_experiment({experiment!r}) failed: {e}"
        return status()

    # autolog is a genuinely separate failure: the destination above may be fine
    # while LangChain instrumentation is unavailable. Losing it costs the automatic
    # LLM spans, not the hand-rolled ones, so tracing stays ENABLED and the reason
    # is recorded rather than silently dropped.
    try:
        mlflow.langchain.autolog()
    except Exception as e:
        logger.warning("MLflow LangChain autolog unavailable (hand-rolled spans still trace): %s", e)
        _state["last_error"] = f"langchain.autolog() failed: {e}"

    _state["enabled"] = True
    logger.info(
        "MLflow tracing enabled: uri=%s experiment=%s id=%s",
        uri, experiment, _state["experiment_id"],
    )
    return status()


def status() -> dict:
    """Current tracing health, as served by GET /mlflow_status."""
    return dict(_state)
