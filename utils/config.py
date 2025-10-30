# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from pathlib import Path

from src.pfl_benchmark.config import ExperimentConfig, RuntimeConfig


class Config:
    """Backwards compatible loader for legacy utilities.

    New code should rely on :class:`src.pfl_benchmark.config.ExperimentConfig`
    and :class:`src.pfl_benchmark.config.RuntimeConfig` directly. This shim only
    implements the minimal API surface used by the evaluation helpers.
    """

    @staticmethod
    def cargar(path: str | Path) -> RuntimeConfig:
        directory = Path(path)
        try:
            return RuntimeConfig.from_metadata(directory)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Unable to locate metadata.yaml in {directory}. Did you run the new CLI?"
            ) from exc

    def __init__(self, *_args, **_kwargs) -> None:
        warnings.warn(
            "Direct Config() construction is deprecated. Use the Typer CLI with"
            " YAML configuration files instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "Config() is no longer instantiable. Use ExperimentConfig.from_file(...)."
        )


__all__ = ["Config", "ExperimentConfig", "RuntimeConfig"]
