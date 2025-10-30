# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import subprocess
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from data.d5_dataset import dict_act_red


def _coerce_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if lowered.startswith("0x"):
            return int(lowered, 16)
        if lowered.startswith("0b"):
            return int(lowered, 2)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_overrides(data: MutableMapping[str, Any], overrides: Dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor: MutableMapping[str, Any] = data
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], MutableMapping):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value


class DatasetConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(default="digitfive", description="Dataset identifier")
    collection: str = Field(default="d5", description="Logical collection identifier")
    data_type: Optional[str] = Field(default=None, description="Primary data domain for honest clients")
    extra_data_type: Optional[str] = Field(default=None, description="Data domain assigned to adversarial clients")
    batch_size: int = Field(default=64, gt=0)
    test_batch_size: int = Field(default=256, gt=0)
    path_to_models_zero: str = Field(default="models/d5_net_zeros.pt")
    root: str = Field(default="./data/d5/")
    nbyz: Optional[int] = Field(default=None, ge=0)
    attack_type: str = Field(
        default="mean",
        description="Type of Byzantine attack used by adversarial clients",
    )

    @field_validator("attack_type")
    def _attack_type_supported(cls, value: str) -> str:
        normalised = str(value).lower()
        allowed = {"mean", "backdoor", "label_flip", "none"}
        if normalised not in allowed:
            raise ValueError(
                "dataset.attack_type must be one of: " + ", ".join(sorted(allowed))
            )
        return normalised

    @field_validator("collection")
    def _collection_supported(cls, value: str) -> str:
        if value not in {"d5", "5b"}:
            raise ValueError(
                "dataset.collection must be either 'd5' or '5b' for DigitFive experiments"
            )
        return value

    @model_validator(mode="after")
    def _validate_requirements(cls, model: "DatasetConfig") -> "DatasetConfig":
        # For 'd5' collection these fields are required
        if model.collection == "d5":
            if model.data_type is None:
                raise ValueError("dataset.data_type must be provided when collection is 'd5'")
            if model.extra_data_type is None:
                raise ValueError("dataset.extra_data_type must be provided when collection is 'd5'")
            if model.nbyz is None:
                return model.model_copy(update={"nbyz": 1})
            if model.attack_type != "none" and model.data_type != model.extra_data_type:
                raise ValueError(
                    "Byzantine attacks are only supported when dataset.data_type and dataset.extra_data_type are the same"
                )
        else:
            # For '5b' collection these fields are not required; ensure sensible default for nbyz
            if model.nbyz is None:
                return model.model_copy(update={"nbyz": 0})
        return model


class OptimizerConfig(BaseModel):
    name: str = Field(description="Optimizer name (adam or sgd)")
    lr: float = Field(description="Learning rate", gt=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: Optional[float] = Field(default=None, ge=0.0, description="Only for SGD")

    @field_validator("name", mode="before")
    def _normalise_name(cls, v: str) -> str:
        return str(v).lower()

    @model_validator(mode="after")
    def _momentum_for_sgd(self):
        if self.name == "sgd":
            if self.momentum is None:
                self.momentum = 0.0
        else:  # adam
            if self.momentum is not None:
                raise ValueError("momentum is only applicable when using SGD")
        return self


class AlgorithmConfig(BaseModel):
    name: str = Field(description="Federated algorithm to execute")
    num_clients: int = Field(gt=0)
    num_epochs: int = Field(gt=0)
    eval_interval: int = Field(gt=0)
    delta_personalization: bool = Field(default=False)
    fl_freq: int = Field(default=5, gt=0)
    mu: Optional[float] = Field(default=None)
    k_active: Optional[int] = Field(default=None, gt=0)
    epoch_encoder: int = Field(default=4, gt=0)
    ablation: str = Field(default="baseline")

    @field_validator("name")
    def _normalise_algorithm(cls, value: str) -> str:
        normalised = value.lower()
        allowed = {"fedavg", "fedprox", "scaffold", "apfl", "flprotector"}
        if normalised not in allowed:
            raise ValueError(
                "algorithm.name must be one of: " + ", ".join(sorted(allowed))
            )
        return normalised

    @field_validator("ablation", mode="before")
    def _normalise_ablation(cls, value: str) -> str:
        normalised = str(value).lower()
        allowed = {"baseline", "no_autoencoder", "no_personalization", "no_lbfgs"}
        if normalised not in allowed:
            raise ValueError(
                "algorithm.ablation must be one of: " + ", ".join(sorted(allowed))
            )
        return normalised

    @model_validator(mode="after")
    def _validate_dependencies(cls, model: "AlgorithmConfig") -> "AlgorithmConfig":
        name = model.name
        mu = model.mu
        if name == "fedprox":
            if mu is None:
                raise ValueError("algorithm.mu must be provided for FedProx experiments")
            if mu < 0:
                raise ValueError("algorithm.mu must be non-negative")
        elif mu is not None:
            raise ValueError("algorithm.mu is only valid for FedProx")

        if name == "apfl" and model.k_active is None:
            return model.model_copy(update={"k_active": model.num_clients})
        return model


class RuntimeSettings(BaseModel):
    gpu: int = Field(default=0)
    use_cuda: Optional[bool] = Field(default=None)
    output_root: str = Field(default="PROBAS")
    run_name: Optional[str] = Field(default=None)
    timestamp: bool = Field(default=True)
    home_path: str = Field(default="")
    seed: Optional[int] = Field(default=None)


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    algorithm: AlgorithmConfig
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @classmethod
    def from_file(
        cls,
        path: Path,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        overrides = overrides or {}
        _apply_overrides(payload, overrides)
        try:
            return cls.model_validate(payload)
        except Exception as exc:  # pragma: no cover - surfacing validation errors
            raise ValueError(f"Invalid configuration in {path}: {exc}") from exc

    def to_runtime_config(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        source_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> "RuntimeConfig":
        return RuntimeConfig(
            experiment=self,
            overrides=overrides or {},
            source_path=source_path,
            output_path=output_path,
        )


@dataclass
class RuntimeConfig:
    experiment: ExperimentConfig
    overrides: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    dataset: DatasetConfig = field(init=False)
    optimizer: OptimizerConfig = field(init=False)
    algorithm: AlgorithmConfig = field(init=False)
    runtime: RuntimeSettings = field(init=False)

    COL: str = field(init=False)
    SIZE: int = field(init=False)
    NBYZ: int = field(init=False)
    RANK: int = field(init=False, default=0)
    GPU: int = field(init=False)
    HOME_PATH: str = field(init=False)
    DEVICE: torch.device = field(init=False)
    TIPO_EXEC: str = field(init=False)
    TIMESTAMP: bool = field(init=False)
    DATA_TYPE: str = field(init=False)
    EXTRA_DATA_TYPE: str = field(init=False)
    PATH: str = field(init=False)
    EPOCHS: int = field(init=False)
    LR: float = field(init=False)
    CHECK_PREC: int = field(init=False)
    FL_FREQ: int = field(init=False)
    DELTA_PERS: bool = field(init=False)
    FLDET_START: int = field(init=False, default=2)
    MU: float = field(init=False, default=0.0)
    OPTIMIZER: str = field(init=False)
    K: int = field(init=False)
    EPOCH_ENCODER: int = field(init=False)
    PATH_TOMODELS_ZERO: str = field(init=False)
    DATA_UBI: str = field(init=False)
    BATCH_TEST_SIZE: int = field(init=False)
    BATCH_SIZE: int = field(init=False)
    DIC: Dict[str, Any] = field(init=False)
    metadata_path: Optional[Path] = field(init=False, default=None)
    ABLATION_MODE: Optional[str] = field(init=False, default=None)
    BYZANTINE_ATTACK : str = field(init=False)
    BYZANTINE_SCALE : float = field(init=False, default=1.0)
    BYZANTINE_TARGET : int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.dataset = self.experiment.dataset
        self.optimizer = self.experiment.optimizer
        self.algorithm = self.experiment.algorithm
        self.runtime = self.experiment.runtime

        self.COL = self.dataset.collection
        self.SIZE = self.algorithm.num_clients
        self.NBYZ = self.dataset.nbyz
        self.GPU = self.runtime.gpu
        self.HOME_PATH = self.runtime.home_path
        name_map = {
            "fedavg": "FedAvg",
            "fedprox": "FedProx",
            "scaffold": "Scaffold",
            "apfl": "APFL",
            "flprotector": "FLProtector",
        }
        self.TIPO_EXEC = name_map[self.algorithm.name]
        self.TIMESTAMP = bool(self.runtime.timestamp)
        self.DATA_TYPE = self.dataset.data_type
        self.EXTRA_DATA_TYPE = self.dataset.extra_data_type
        self.EPOCHS = self.algorithm.num_epochs
        self.LR = self.optimizer.lr
        self.CHECK_PREC = self.algorithm.eval_interval
        self.FL_FREQ = self.algorithm.fl_freq
        self.DELTA_PERS = self.algorithm.delta_personalization
        self.MU = float(self.algorithm.mu or 0.0)
        self.OPTIMIZER = "SGD" if self.optimizer.name == "sgd" else "Adam"
        self.K = self.algorithm.k_active or self.algorithm.num_clients
        self.EPOCH_ENCODER = self.algorithm.epoch_encoder
        self.PATH_TOMODELS_ZERO = self.dataset.path_to_models_zero
        self.DATA_UBI = self.dataset.root
        self.BATCH_TEST_SIZE = self.dataset.test_batch_size
        self.BATCH_SIZE = self.dataset.batch_size
        self.DIC = dict_act_red
        self.BYZANTINE_ATTACK = self.dataset.attack_type
        self.BYZANTINE_SCALE = (self.SIZE - self.NBYZ) / self.NBYZ
        self.BYZANTINE_TARGET = 0

        self.ABLATION_MODE = None
        if self.TIPO_EXEC == "FLProtector":
            self.ABLATION_MODE = self.algorithm.ablation

        if self.output_path is not None:
            output_dir = Path(self.output_path)
        else:
            output_dir = Path(self.runtime.output_root) / self.COL / self.TIPO_EXEC
            if self.HOME_PATH:
                output_dir = output_dir / self.HOME_PATH
            if self.runtime.run_name:
                output_dir = output_dir / self.runtime.run_name
            if self.ABLATION_MODE:
                output_dir = output_dir / self.ABLATION_MODE
            if self.DATA_TYPE == self.EXTRA_DATA_TYPE:
                output_dir = output_dir / self.DATA_TYPE
            else:
                output_dir = output_dir / f"{self.DATA_TYPE}_{self.EXTRA_DATA_TYPE}"
            if self.BYZANTINE_ATTACK:
                output_dir = output_dir / f"byz_{self.BYZANTINE_ATTACK}"
            if self.TIMESTAMP:
                output_dir = output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.PATH = str(output_dir)

        use_cuda = self.runtime.use_cuda
        if use_cuda is None:
            use_cuda = torch.cuda.is_available() and self.GPU >= 0
        self.DEVICE = torch.device("cuda", self.GPU) if use_cuda and torch.cuda.is_available() else torch.device("cpu")

    def to_serializable_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "COL": self.COL,
            "SIZE": self.SIZE,
            "NBYZ": self.NBYZ,
            "GPU": self.GPU,
            "HOME_PATH": self.HOME_PATH,
            "TIPO_EXEC": self.TIPO_EXEC,
            "TIMESTAMP": self.TIMESTAMP,
            "DATA_TYPE": self.DATA_TYPE,
            "EXTRA_DATA_TYPE": self.EXTRA_DATA_TYPE,
            "PATH": self.PATH,
            "EPOCHS": self.EPOCHS,
            "LR": self.LR,
            "CHECK_PREC": self.CHECK_PREC,
            "FL_FREQ": self.FL_FREQ,
            "DELTA_PERS": self.DELTA_PERS,
            "FLDET_START": self.FLDET_START,
            "MU": self.MU,
            "OPTIMIZER": self.OPTIMIZER,
            "K": self.K,
            "EPOCH_ENCODER": self.EPOCH_ENCODER,
            "PATH_TOMODELS_ZERO": self.PATH_TOMODELS_ZERO,
            "DATA_UBI": self.DATA_UBI,
            "BATCH_TEST_SIZE": self.BATCH_TEST_SIZE,
            "BATCH_SIZE": self.BATCH_SIZE,
            "DEVICE": str(self.DEVICE),
            "BYZANTINE_ATTACK": self.BYZANTINE_ATTACK,
            "BYZANTINE_SCALE": self.BYZANTINE_SCALE,
            "BYZANTINE_TARGET": self.BYZANTINE_TARGET,
        }
        if self.ABLATION_MODE is not None:
            data["ABLATION_MODE"] = self.ABLATION_MODE
        return data

    def _git_metadata(self) -> Dict[str, Any]:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            commit = None
        try:
            status = subprocess.check_output(["git", "status", "--short"], text=True)
            dirty = bool(status.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            dirty = None
        return {"commit": commit, "dirty": dirty}

    def _environment_metadata(self) -> Dict[str, Any]:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.DEVICE),
        }

    def serialize_metadata(self) -> Path:
        output_dir = Path(self.PATH)
        metadata = {
            "config": self.experiment.model_dump(mode="python"),
            "overrides": self.overrides,
            "resolved": self.to_serializable_dict(),
            "git": self._git_metadata(),
            "environment": self._environment_metadata(),
        }
        yaml_path = output_dir / "metadata.yaml"
        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(metadata, fh, sort_keys=False)
        self.metadata_path = yaml_path
        self.metadata = metadata
        return yaml_path

    def leer_indices_dict(self):
        """Return the train/test index splits for the current client."""

        return self._leer_indices_5b() if self.COL == "5b" else self._leer_indices_d5()

    def _leer_indices_d5(self):
        data_type = self.EXTRA_DATA_TYPE if self.RANK < self.NBYZ else self.DATA_TYPE
        test_key = "test_fltrust" if (self.TIPO_EXEC == "FLTrust" and data_type == "usps") else "test"
        try:
            train_data = self.DIC[data_type][self.RANK]["train"]
            test_data = self.DIC[data_type][self.RANK][test_key]
        except (KeyError, TypeError):
            train_data = self.DIC[data_type][self.RANK]
            test_data = self.DIC[data_type]["test"]
        return train_data, test_data, data_type

    def _leer_indices_5b(self):
        mini_dict_5b = {0: "mnist", 1: "mnistm", 2: "svhn", 3: "syn", 4: "usps"}
        data_type = mini_dict_5b.get(self.RANK, "mnist")
        try:
            train_data = self.DIC[data_type][self.RANK]["train"]
            test_data = self.DIC[data_type][self.RANK]["test"]
        except (KeyError, TypeError):
            train_data = self.DIC[data_type][self.RANK]
            test_data = self.DIC[data_type]["test"]
        return train_data, test_data, data_type

    @classmethod
    def from_metadata(cls, directory: Path) -> "RuntimeConfig":
        yaml_path = directory / "metadata.yaml"
        data: Optional[Dict[str, Any]] = None
        metadata_path: Optional[Path] = None
        if yaml_path.exists():
            metadata_path = yaml_path
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        if data is None:
            raise FileNotFoundError(
                f"No metadata.yaml found in {directory}."
            )
        config_payload = data.get("config")
        if config_payload is None:
            raise ValueError(f"metadata file in {directory} is missing the 'config' section")
        overrides = data.get("overrides", {})
        experiment = ExperimentConfig.model_validate(config_payload)
        resolved = data.get("resolved", {})
        output_path = Path(resolved["PATH"]) if "PATH" in resolved else directory
        runtime_config = cls(
            experiment=experiment,
            overrides=overrides,
            source_path=metadata_path,
            output_path=output_path,
        )
        runtime_config.metadata = data
        runtime_config.metadata_path = metadata_path
        return runtime_config


def parse_override_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected format key=value")
        key, raw_value = pair.split("=", 1)
        if not key:
            raise ValueError("Override keys cannot be empty")
        overrides[key] = _coerce_override_value(raw_value)
    return overrides


__all__ = [
    "AlgorithmConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "OptimizerConfig",
    "RuntimeConfig",
    "RuntimeSettings",
    "parse_override_pairs",
]
