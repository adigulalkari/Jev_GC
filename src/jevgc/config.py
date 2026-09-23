"""JevGCConfig: loads jev-gc's configuration from YAML, environment, or kwargs.

Config load is the one place this library fails fast (§2 principle 2 is about
runtime Jev calls, not startup) -- a missing api_key means every later Jev
call would fail anyway, so we raise `ConfigurationError` immediately instead
of deferring to the first fail-open at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from jevgc.archive import DEFAULT_MAX_CONTENT_BYTES
from jevgc.exceptions import ConfigurationError
from jevgc.models import ErrorTreatment


class JevSettings(BaseModel):
    api_key: SecretStr
    base_url: str = "https://api.typesafe.ai/v1"
    timeout_seconds: float = 2.0
    max_retries: int = 2


class PrefilterSettings(BaseModel):
    keep_last_n_turns: int = 3
    drop_zero_output_after_turns: int = 10


class ScorerSettings(BaseModel):
    batch_max_size: int = 50
    batch_max_wait_ms: int = 200


class PolicySettings(BaseModel):
    relevance_keep_threshold: float = 0.35
    min_confidence_to_drop: float = 0.6
    error_default_treatment: ErrorTreatment = ErrorTreatment.KEEP_ERROR_SUMMARY_ONLY


class StoreSettings(BaseModel):
    backend: str = "memory"
    sqlite_path: str | None = None


class ArchiveSettings(BaseModel):
    """Ceiling on the memory the content archive may hold. Past it, the
    least-recently-used snapshots release their content and keep their
    keywords -- evicted spans stay discoverable, but can no longer be
    rehydrated verbatim. See `jevgc.archive.Archive`."""

    max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES


class TelemetrySettings(BaseModel):
    emit_self_metrics: bool = True


class JevGCConfig(BaseSettings):
    """Root configuration object.

    Loadable three ways: `JevGCConfig.from_yaml(path)` (env vars are still
    interpolated into `${VAR}` placeholders in the YAML), directly from
    environment variables prefixed `JEVGC_` with `__` as the nested
    delimiter (e.g. `JEVGC_JEV__API_KEY`), or via explicit kwargs for tests.
    """

    model_config = SettingsConfigDict(
        env_prefix="JEVGC_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    jev: JevSettings
    prefilter: PrefilterSettings = Field(default_factory=PrefilterSettings)
    scorer: ScorerSettings = Field(default_factory=ScorerSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    archive: ArchiveSettings = Field(default_factory=ArchiveSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> JevGCConfig:
        """Load config from a YAML file, interpolating `${ENV_VAR}` placeholders
        against `os.environ` before parsing (e.g. `api_key: ${JEV_API_KEY}`)."""
        yaml_path = Path(path)
        try:
            raw_text = yaml_path.read_text()
        except OSError as exc:
            raise ConfigurationError(f"Could not read config file {yaml_path}: {exc}") from exc

        interpolated = _interpolate_env(raw_text)

        try:
            data: dict[str, Any] = yaml.safe_load(interpolated) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in {yaml_path}: {exc}") from exc

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JevGCConfig:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid jev-gc configuration: {exc}") from exc

    # Alias matching the quickstart snippet in the spec/README.
    @classmethod
    def from_config(cls, path: str | Path) -> JevGCConfig:
        return cls.from_yaml(path)


def _interpolate_env(text: str) -> str:
    """Replace `${VAR_NAME}` with `os.environ["VAR_NAME"]`, leaving the
    placeholder untouched (so YAML parsing surfaces a clear error) if unset."""
    import re

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, text)
