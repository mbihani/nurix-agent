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
    ai_gateway_url: str = Field(default="https://7474660648944264.ai-gateway.cloud.databricks.com/mlflow/v1", validation_alias="AI_GATEWAY_URL")
    claude_model: str = Field(default="databricks-claude-sonnet-5", validation_alias="CLAUDE_MODEL")
    mlflow_experiment: str = Field(default="nurix-agent-traces", validation_alias="MLFLOW_EXPERIMENT")


def get_databricks_token(cfg: 'AppConfig') -> str:
    """Fetch a fresh Databricks bearer token from the SDK."""
    from databricks.sdk import WorkspaceClient
    ws = WorkspaceClient(host=cfg.databricks_host)
    auth = ws.config.authenticate()
    token = auth.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise RuntimeError("No Databricks token available. Ensure the workspace is authenticated.")
    return token
