from __future__ import annotations

import pytest

from jevgc.config import JevGCConfig
from jevgc.exceptions import ConfigurationError

SECRET = "sk-super-secret-jev-key-do-not-leak"


def test_missing_api_key_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        JevGCConfig.from_dict({"jev": {}})


def test_defaults_are_applied():
    config = JevGCConfig.from_dict({"jev": {"api_key": SECRET}})
    assert config.jev.base_url == "https://api.typesafe.ai/v1"
    assert config.prefilter.keep_last_n_turns == 3
    assert config.policy.relevance_keep_threshold == 0.35


def test_api_key_never_leaks_in_str_or_repr():
    config = JevGCConfig.from_dict({"jev": {"api_key": SECRET}})
    assert SECRET not in str(config)
    assert SECRET not in repr(config)
    assert SECRET not in str(config.jev)
    assert SECRET not in repr(config.jev)


def test_api_key_never_leaks_in_raised_exception_message():
    try:
        JevGCConfig.from_dict({"jev": {"api_key": SECRET, "timeout_seconds": "not-a-number"}})
    except ConfigurationError as exc:
        assert SECRET not in str(exc)
    else:
        pytest.fail("expected ConfigurationError for invalid timeout_seconds")


def test_from_yaml_interpolates_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    yaml_path = tmp_path / "jevgc.yaml"
    yaml_path.write_text("jev:\n  api_key: ${JEV_API_KEY}\n")

    config = JevGCConfig.from_yaml(yaml_path)

    assert config.jev.api_key.get_secret_value() == SECRET


def test_from_yaml_missing_file_raises_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError):
        JevGCConfig.from_yaml(tmp_path / "does_not_exist.yaml")


def test_from_yaml_invalid_yaml_raises_configuration_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("jev: [this is not, valid: yaml")
    with pytest.raises(ConfigurationError):
        JevGCConfig.from_yaml(bad)
