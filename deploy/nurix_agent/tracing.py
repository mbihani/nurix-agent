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

Customer data — what this module DOES and DOES NOT protect against
------------------------------------------------------------------
Read this before adding a span attribute. An earlier version of this docstring
claimed spans carry "never full verbatim text", which was FALSE: `truncate()` caps
at MAX_SPAN_TEXT characters, so anything SHORTER than the cap is recorded IN FULL.

What is actually true:

  * ROW DATA is never recorded. Tabular results contribute only counts
    (`row_count`, `column_count`) and `column_names()` — schema, not content.
    This is the one real content guarantee here.
  * Free text (questions, refine instructions, error strings, generated SQL) IS
    recorded, bounded to MAX_SPAN_TEXT characters by `truncate()`. A customer
    verbatim shorter than that cap lands in the span verbatim and complete.
  * `truncate()` is therefore a VOLUME control, not a privacy control. It keeps a
    span attribute from ballooning; it does not redact anything.
  * It is nonetheless the single chokepoint for every free-text attribute, so the
    bound cannot drift per node. Route text through it rather than slicing at the
    call site, and never pass unbounded free text to `set_inputs`/`set_outputs`.

KNOWN LIMITATION (deliberate, not an oversight): there is NO PII detection or
scrubbing. The highest-exposure attribute is Genie-generated `sql`, which can embed
customer content as a literal (`WHERE verbatim LIKE '%...%'`) — bounded here, but
not redacted. Anyone pointing this app at real customer feedback should treat the
trace experiment as carrying customer data and control access to it accordingly.
Adding real redaction is a design decision with its own failure modes (false
negatives read as a guarantee); it was consciously left out rather than faked.
"""
import logging

import mlflow

logger = logging.getLogger(__name__)

# Free text entering a span attribute is capped at this many characters. This bounds
# attribute SIZE; it does not redact. Text shorter than the cap is recorded in full —
# see the "Customer data" section above before assuming otherwise.
MAX_SPAN_TEXT = 500

# Tracking-URI values that mean "do not trace at all". Lets a local run opt out
# via MLFLOW_TRACKING_URI="" (or "none"/"off"/"disabled") without code changes,
# instead of silently filling ./mlruns.
_DISABLED_URIS = frozenset({"", "none", "off", "disabled", "false"})

# Process-local, and deliberately so — but see the note in status(): the app runs
# under multiple uvicorn workers, so this reflects ONE worker's view.
#
# Three separate booleans rather than one `enabled`, because the failure modes are
# genuinely independent and an operator conflating them chases the wrong bug:
#   destination_ok — tracking URI + experiment resolved; traces have a real home.
#   autolog_ok     — LangChain auto-instrumentation active (the `LangGraph` span).
#   manual_spans   — whether `mlflow.start_span` calls are still being made.
_state: dict = {
    "enabled": False,
    "destination_ok": False,
    "autolog_ok": False,
    "tracking_uri": None,
    "experiment": None,
    "experiment_id": None,
    "last_error": None,
}


def truncate(text, limit: int = MAX_SPAN_TEXT) -> str:
    """
    Bound free text for a span attribute: coerced to str, capped, marked when cut.

    This is a SIZE bound, NOT redaction — text under `limit` is returned unchanged.
    Do not treat a call to this function as making an attribute safe to record; it
    only makes it small. See the module docstring's "Customer data" section.

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
                  destination_ok=False, autolog_ok=False,
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
        _state["autolog_ok"] = True
    except Exception as e:
        logger.warning("MLflow LangChain autolog unavailable (hand-rolled spans still trace): %s", e)
        _state["last_error"] = f"langchain.autolog() failed: {e}"

    _state["destination_ok"] = True
    _state["enabled"] = True
    logger.info(
        "MLflow tracing enabled: uri=%s experiment=%s id=%s",
        uri, experiment, _state["experiment_id"],
    )
    return status()


def status() -> dict:
    """
    Current tracing health, as served by GET /mlflow_status.

    Reading the flags
    -----------------
      destination_ok — tracking URI and experiment resolved. Traces have a real home.
      autolog_ok     — LangChain autolog active, so the `LangGraph` span appears.
      manual_spans   — whether hand-rolled `mlflow.start_span` calls are being made.
      degraded       — destination is fine but something else failed (today: autolog).
                       `last_error` says which. This is the state that used to read
                       ambiguously as `enabled: true` WITH a non-null `last_error`.
      enabled        — kept as `destination_ok` for compatibility with existing
                       callers; prefer the specific flags above.

    manual_spans is currently ALWAYS true, and that is a real caveat rather than a
    formality: nothing in this app gates `mlflow.start_span`, so when init is disabled
    or fails, every node still opens spans. MLflow then resolves them against its own
    default destination — i.e. a local `./mlruns` on the container's ephemeral disk,
    which is the exact failure this module was written to eliminate. It is reported
    here rather than silently gated because gating spans app-wide would also hide the
    calls from a local `mlflow ui` run, where writing locally is the desired outcome.
    Treat `manual_spans: true` with `destination_ok: false` as "spans are being
    written somewhere you are probably not reading".

    `_state` is PROCESS-LOCAL and the app serves under multiple uvicorn workers, so
    this reflects only the worker that happened to answer this request. Workers all
    run the same `init()` against the same config, so they should agree — but a
    transient per-worker failure (one worker's `set_experiment` racing a permission
    change) shows up in only some responses. Poll a few times before concluding the
    whole app is healthy, and do not read one 200 as fleet-wide.
    """
    out = dict(_state)
    # Always attempted — see the docstring. Reported, not inferred.
    out["manual_spans"] = True
    out["degraded"] = bool(_state["destination_ok"] and _state["last_error"])
    return out
