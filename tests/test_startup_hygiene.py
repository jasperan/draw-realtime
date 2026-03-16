"""Tests for startup warning cleanup helpers."""

from __future__ import annotations

import logging

from app.startup_hygiene import (
    WarningTextFilter,
    install_warning_filters,
    normalize_hf_telemetry_env,
)


def test_normalize_hf_telemetry_env_removes_legacy_false_value():
    env = {"DISABLE_TELEMETRY": "0"}

    normalize_hf_telemetry_env(env)

    assert "DISABLE_TELEMETRY" not in env
    assert "HF_HUB_DISABLE_TELEMETRY" not in env


def test_normalize_hf_telemetry_env_maps_truthy_value_to_new_name():
    env = {"DISABLE_TELEMETRY": "true"}

    normalize_hf_telemetry_env(env)

    assert "DISABLE_TELEMETRY" not in env
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_warning_text_filter_blocks_known_startup_noise():
    filter_ = WarningTextFilter(("skip_prk_steps",))
    record = logging.LogRecord(
        name="diffusers.configuration_utils",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="The config attributes {'skip_prk_steps': True} were passed to LCMScheduler",
        args=(),
        exc_info=None,
    )

    assert filter_.filter(record) is False


def test_warning_text_filter_keeps_unrelated_messages():
    filter_ = WarningTextFilter(("skip_prk_steps",))
    record = logging.LogRecord(
        name="diffusers.configuration_utils",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="A real warning we still want to see",
        args=(),
        exc_info=None,
    )

    assert filter_.filter(record) is True


def test_install_warning_filters_is_idempotent():
    logger = logging.getLogger("diffusers.configuration_utils")
    original_filters = list(logger.filters)
    try:
        logger.filters[:] = []

        install_warning_filters()
        first_count = len(logger.filters)
        install_warning_filters()
        second_count = len(logger.filters)

        assert first_count == 1
        assert second_count == 1
    finally:
        logger.filters[:] = original_filters
