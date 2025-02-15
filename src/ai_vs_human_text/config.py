from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_VS_HUMAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    model_path: Path = Field(default=Path("artifacts/model.joblib"))
    metrics_path: Path = Field(default=Path("artifacts/metrics.json"))
    max_input_chars: int = Field(default=50_000)
    min_input_chars: int = Field(default=1)

    http_timeout_s: float = Field(default=30.0)
    http_max_retries: int = Field(default=3)

    @field_validator("model_path", "metrics_path", mode="before")
    @classmethod
    def resolve_paths(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            root = _default_project_root()
            p = (root / p).resolve()
        return p


class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    model_primary: str = Field(
        default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL_PRIMARY"
    )
    model_fallback: str = Field(
        default="llama-3.1-8b-instant", validation_alias="GROQ_MODEL_FALLBACK"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_groq_settings() -> GroqSettings:
    return GroqSettings()
