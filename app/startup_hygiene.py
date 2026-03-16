"""Startup hygiene helpers for draw-realtime.

These are deliberately small and boring. They run before heavy ML imports so the
server starts with fewer noisy, known-harmless warnings.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping, Sequence

_TRUE_VALUES = {"1", "true", "yes", "on"}

# Known warning texts we intentionally suppress during startup because they are
# harmless with this pinned stack.
_IGNORED_WARNING_SNIPPETS = (
    "You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>",
    "The config attributes {'skip_prk_steps': True} were passed to LCMScheduler",
    "The config attributes {'shift_factor': 0.0, 'upsample_fn': 'nearest'} were passed to AutoencoderTiny",
)

_FILTER_LOGGERS = (
    "diffusers.configuration_utils",
    "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
)


class WarningTextFilter(logging.Filter):
    """Drop log records that contain a known harmless warning substring."""

    def __init__(self, ignored_snippets: Sequence[str]):
        super().__init__()
        self.ignored_snippets = tuple(ignored_snippets)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(snippet in message for snippet in self.ignored_snippets)


def normalize_hf_telemetry_env(env: MutableMapping[str, str] | None = None) -> None:
    """Replace the deprecated DISABLE_TELEMETRY env var with the new HF name.

    transformers emits a FutureWarning whenever DISABLE_TELEMETRY is present,
    even when it is set to a falsy value like "0". We normalize it once before
    transformers/diffusers import so startup stays quiet.
    """

    target_env = os.environ if env is None else env
    legacy_value = target_env.pop("DISABLE_TELEMETRY", None)

    if legacy_value is None:
        return

    if "HF_HUB_DISABLE_TELEMETRY" in target_env:
        return

    if str(legacy_value).strip().lower() in _TRUE_VALUES:
        target_env["HF_HUB_DISABLE_TELEMETRY"] = "1"


def install_warning_filters() -> None:
    """Install targeted filters for noisy, known-harmless startup warnings."""

    for logger_name in _FILTER_LOGGERS:
        logger = logging.getLogger(logger_name)
        if any(isinstance(existing, WarningTextFilter) for existing in logger.filters):
            continue
        logger.addFilter(WarningTextFilter(_IGNORED_WARNING_SNIPPETS))
