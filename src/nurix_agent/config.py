from pathlib import Path
from typing import ClassVar
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)

class AppConfig(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=env_file, extra="ignore"
    )
    databricks_host: str = Field(default="https://fevm-stable-classic-7ppxjq.cloud.databricks.com", validation_alias="DATABRICKS_HOST")
    genie_space_id: str = Field(default="01f11dcb53181defb69ee49bd73bca10", validation_alias="GENIE_SPACE_ID")
    # Used only by deep-research (agent) mode, to re-execute the sub-queries Genie
    # reports SQL for but returns no structured data for.
    warehouse_id: str = Field(default="24b0352e1b0dca66", validation_alias="WAREHOUSE_ID")
    ai_gateway_url: str = Field(default="https://7474660648944264.ai-gateway.cloud.databricks.com/mlflow/v1", validation_alias="AI_GATEWAY_URL")
    claude_model: str = Field(default="databricks-claude-sonnet-5", validation_alias="CLAUDE_MODEL")
    # MUST be an ABSOLUTE workspace path. Databricks rejects a bare experiment name,
    # and a bare name silently sent every trace to a local ./mlruns directory on the
    # app container's ephemeral disk — invisible to the user and gone on restart.
    mlflow_experiment: str = Field(default="/Shared/nurix-agent-traces", validation_alias="MLFLOW_EXPERIMENT")
    # Without this set, MLflow defaults to a LOCAL ./mlruns directory. Set to
    # "databricks" so traces land in the workspace; set to "" (or "none"/"off") to
    # turn tracing off for a local run.
    mlflow_tracking_uri: str = Field(default="databricks", validation_alias="MLFLOW_TRACKING_URI")

    # Whether `ask_about_viz` may CONTINUE the Genie conversation named by the request's
    # `conversation_id` instead of starting a fresh one.
    #
    # DEFAULTS OFF, and must stay off until conversation OWNERSHIP CAN BE VERIFIED.
    # The gap is not in the SDK call, it is in the trust model: nothing binds a submitted
    # `conversation_id` to the submitting user, to `session_id`, or to the chart being
    # asked about, and in production every Genie call runs as ONE app service principal.
    # So a caller who supplies a conversation ID belonging to a DIFFERENT app session
    # could reach that conversation's prior context, because the SP — not the end user —
    # is what Genie authorizes. The IDs are high entropy and the deployed nurix-nlviz
    # never sends the field, so this is latent rather than live; gating it costs nothing
    # today and removes the latent cross-session exposure.
    #
    # BEFORE ENABLING THIS, one of the following must exist:
    #   * server-side provenance: the app stores which conversation produced which pin,
    #     for which user/session, and the request is checked against that record rather
    #     than trusting the id in the payload; or
    #   * a Genie-side ownership check that the caller (not just the SP) may read the
    #     conversation.
    # With the flag off the field is ACCEPTED AND IGNORED — a fresh conversation is
    # started with the chart's SQL as context, which is the behaviour every current
    # client already gets.
    enable_conversation_continuation: bool = Field(
        default=False, validation_alias="ENABLE_CONVERSATION_CONTINUATION"
    )


def get_databricks_token(cfg: 'AppConfig') -> str:
    """Fetch a fresh Databricks bearer token from the SDK."""
    from databricks.sdk import WorkspaceClient
    ws = WorkspaceClient(host=cfg.databricks_host)
    auth = ws.config.authenticate()
    token = auth.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise RuntimeError("No Databricks token available. Ensure the workspace is authenticated.")
    return token
