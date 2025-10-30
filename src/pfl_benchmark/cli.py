# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Optional

import torch
import typer

from utils.file_utils import print_config
from utils.init_utils import (
    init_d5_apfl,
    init_d5_fedavg,
    init_d5_fedprox,
    init_d5_flprotector,
    init_d5_scaffold,
)
from .algorithms import (
    APFLAlgorithm,
    FLProtectorAlgorithm,
    FedAvgAlgorithm,
    FedProxAlgorithm,
    ScaffoldAlgorithm,
)
from .config import ExperimentConfig, RuntimeConfig, parse_override_pairs
from .core.aggregators import WeightedAveragingAggregator
from .core.runner import Client, FederatedRunner
from .evaluation import (
    load_flprotector_artifacts,
    load_all_algorithm_artifacts,
    run_backdoor_attack_evaluation,
    run_flprotector_final_evaluation,
)

app = typer.Typer(help="Command line interface for running PFL benchmark experiments.")

ALGORITHM_REGISTRY = {
    "FedAvg": {"initializer": init_d5_fedavg, "algorithm": FedAvgAlgorithm},
    "FedProx": {"initializer": init_d5_fedprox, "algorithm": FedProxAlgorithm},
    "Scaffold": {"initializer": init_d5_scaffold, "algorithm": ScaffoldAlgorithm},
    "APFL": {"initializer": init_d5_apfl, "algorithm": APFLAlgorithm},
    "FLProtector": {"initializer": init_d5_flprotector, "algorithm": FLProtectorAlgorithm},
}


def _build_clients(trainings: Sequence[object]) -> List[Client]:
    """Create :class:`Client` wrappers for each training pipeline.

    The wrapped ``training`` objects usually expose a ``trainer`` attribute that
    implements ``model`` management as well as ``adestrar``/``testear`` methods.
    The wrapper keeps that behaviour encapsulated while exposing additional
    metadata (``client_id`` and ``data_type``) required by the runner.
    """

    clients: List[Client] = []
    for training in trainings:
        clients.append(
            Client(
                client_id=getattr(training, "client_id", None),
                training=training,
                data_type=getattr(training, "data_type", None)
            )
        )
    return clients


def _run_experiment(config: RuntimeConfig) -> None:
    """Instantiate the federated runner and execute a full training session.

        Responsibilities include resolving the requested algorithm, calling the
        initialization utilities that construct the global model and the list of
        client training pipelines, and wiring them into :class:`FederatedRunner`.
        The runner then takes care of alternating the aggregation rounds, running
        per-client training loops, and executing evaluation hooks.
        """
    entry = ALGORITHM_REGISTRY.get(config.TIPO_EXEC)
    if entry is None:
        raise typer.BadParameter(f"Unsupported algorithm '{config.TIPO_EXEC}'.")

    device = config.DEVICE
    initializer = entry["initializer"]
    init_result = initializer(config, device)
    if not isinstance(init_result, (list, tuple)) or len(init_result) < 2:
        raise RuntimeError(
            "Initialiser must return a sequence containing at least (global_model, clients)."
        )
    global_model, trainings = init_result[0], init_result[1]
    clients = _build_clients(trainings)
    algorithm = entry["algorithm"](config)
    runner = FederatedRunner(
        global_model=global_model,
        clients=clients,
        algorithm=algorithm,
        aggregator=WeightedAveragingAggregator(),
        epochs=config.EPOCHS,
        evaluation_interval=config.CHECK_PREC,
    )
    runner.run()


@app.command()
def train(
        config: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to experiment YAML"),
        override: List[str] = typer.Option(
            None,
            "--override",
            help="Override configuration values, e.g. --override algorithm.num_epochs=5",
        )
) -> None:
    """Run a federated experiment using the provided configuration file.

    ``config`` points to the YAML file describing the experiment (dataset,
    model, algorithm and runtime options). Optional overrides allow quick
    tweaking of nested values without modifying the file. The resulting
    configuration is validated, the random seed is applied when requested, and
    :func:`_run_experiment` is invoked to manage the full training lifecycle.
    """
    override_pairs = parse_override_pairs(override or []) if override else {}
    experiment = ExperimentConfig.from_file(config, overrides=override_pairs)
    runtime = experiment.to_runtime_config(overrides=override_pairs, source_path=config)

    ablation_mode = getattr(runtime.algorithm, "ablation_mode", "baseline")
    # Validate ablation_mode: permit only the three modes; handle 'no_lbfgs' with a clear message.
    if ablation_mode not in ("baseline", "no_lbfgs"):
        if ablation_mode in ("no_autoencoder", "no_personalization"):
            typer.secho(
                f"Option {ablation_mode} can be directly evaluated using final_test command with"
                f"a previous baseline training run.",
                fg=typer.colors.YELLOW,
            )
        raise typer.BadParameter(
            "Invalid ablation_mode. Allowed values: baseline, no_lbfgs."
        )

    if runtime.runtime.seed is not None:
        torch.manual_seed(runtime.runtime.seed)

    typer.secho(f"Output directory: {runtime.PATH}", fg=typer.colors.GREEN)
    metadata_path = runtime.serialize_metadata()
    print_config(runtime)
    typer.echo(f"Metadata saved to {metadata_path}")
    typer.echo(f"Using device: {runtime.DEVICE}")

    _run_experiment(runtime)


@app.command()
def final_test(
        run_dir: Path = typer.Option(
            ..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True,
            help="Path to a completed federated run directory"
        ),
        ablation_mode: str = typer.Option(
            "baseline",
            "--ablation-mode",
            help=(
                    "Name of the FLProtector ablation mode to evaluate. "
                    "Allowed: baseline, no_autoencoder, no_personalization. "
                    "If 'no_lbfgs' is provided a separate training/test execution is required."
            ),
        ),
        attack: Optional[str] = typer.Option(
            None,
            "--attack",
            help="Optional attack evaluation to run (supported: backdoor).",
        ),
        target_label: int = typer.Option(
            0, "--target-label", min=0, max=9,
            help="Label that the backdoor trigger should induce when --attack backdoor is used.",
        ),
) -> None:
    """Run stored-model evaluations, including optional attack assessments."""

    runtime = RuntimeConfig.from_metadata(Path(run_dir))
    runtime.PATH = str(run_dir)

    attack_normalized = attack.lower() if attack else None
    if attack_normalized:
        if attack_normalized != "backdoor":
            raise typer.BadParameter(
                "Unsupported attack option. Currently only 'backdoor' is available."
            )

        entry = ALGORITHM_REGISTRY.get(runtime.TIPO_EXEC)
        if entry is None:
            raise typer.BadParameter(
                f"Unsupported algorithm '{runtime.TIPO_EXEC}' for attack evaluation."
            )

        initializer = entry["initializer"]
        init_result = initializer(runtime, runtime.DEVICE)
        if not isinstance(init_result, (list, tuple)) or len(init_result) < 2:
            raise RuntimeError(
                "Initialiser must return a sequence containing at least (global_model, clients)."
            )

        global_model, trainings = init_result[0], init_result[1]
        clients = _build_clients(trainings)

        artifacts = load_all_algorithm_artifacts(
            Path(run_dir),
            clients,
            global_model,
            runtime.DEVICE,
        )

        metrics = run_backdoor_attack_evaluation(
            clients=clients,
            target_label=target_label,
            device=runtime.DEVICE,
            global_model=global_model,
            ablation_mode=runtime.ABLATION_MODE,
        )

        metrics.update(
            {
                "algorithm": runtime.TIPO_EXEC,
                "run_directory": str(Path(run_dir).resolve()),
                "attack": attack_normalized,
                "artifacts": {
                    "clients_dir": str(artifacts.clients_dir),
                    "global_model_path": (
                        str(artifacts.global_model_path)
                        if artifacts.global_model_path
                        else None
                    ),
                    "clients": {
                        key: {
                            "client_id": record.client_id,
                            "data_type": record.data_type,
                            "model_path": str(record.model_path)
                            if record.model_path
                            else None,
                            "metadata_path": str(record.metadata_path)
                            if record.metadata_path
                            else None,
                        }
                        for key, record in artifacts.clients.items()
                    },
                },
            }
        )

        output_path = Path(run_dir) / "final_test_backdoor.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        overall = metrics.get("overall", {})
        typer.secho("Backdoor evaluation summary", fg=typer.colors.GREEN)
        typer.echo(
            f"Normal accuracy: {overall.get('normal_accuracy', 0.0):.4f} | "
            f"Backdoor success: {overall.get('backdoor_success_rate', 0.0):.4f}"
        )
        typer.echo(f"Detailed metrics written to {output_path}")
        return

    if runtime.TIPO_EXEC != "FLProtector":
        raise typer.BadParameter(
            "final_test without --attack is only available for FLProtector runs."
        )

    # Validate ablation_mode: permit only the three modes; handle 'no_lbfgs' with a clear message.
    if ablation_mode not in ("baseline", "no_autoencoder", "no_personalization"):
        if ablation_mode == "no_lbfgs":
            typer.secho(
                "Option `no_lbfgs` needs an independent training/test execution. "
                "Equal ponderation on training aggregation is needed.",
                fg=typer.colors.YELLOW,
            )
        raise typer.BadParameter(
            "Invalid ablation_mode. Allowed values: baseline, no_autoencoder, no_personalization."
        )

    init_result = init_d5_flprotector(runtime, runtime.DEVICE)
    if not isinstance(init_result, (list, tuple)) or len(init_result) < 2:
        raise RuntimeError("FLProtector initialiser must return (global_model, clients).")

    global_model, trainings = init_result[0], init_result[1]
    clients = _build_clients(trainings)
    algorithm = FLProtectorAlgorithm(runtime, ablation_mode=ablation_mode)

    load_info = load_flprotector_artifacts(Path(run_dir), clients, global_model, runtime.DEVICE)
    metrics = run_flprotector_final_evaluation(
        algorithm=algorithm,
        clients=clients,
        global_model=global_model,
        ablation_mode=ablation_mode,
    )

    typer.echo(json.dumps(metrics, indent=2))
    folder = "ablations_test/"
    suffix = metrics.get("ablation_mode_slug")
    filename = "final_test.json" if not suffix or suffix == "full" else f"final_test_{suffix}.json"
    manifest_path = Path(run_dir) / folder / filename
    manifest = {
        "run_dir": str(run_dir),
        "metadata_path": str(runtime.metadata_path) if runtime.metadata_path else None,
        "ablation_mode": ablation_mode,
        "metrics": metrics,
        "artifacts": load_info,
    }
    # create folder if it does not exist
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    typer.echo(f"Final evaluation manifest written to {manifest_path}")


__all__ = ["app", "train", "final_test"]
