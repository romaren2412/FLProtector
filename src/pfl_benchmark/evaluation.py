# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""Post-training evaluation utilities for stored experiment runs."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn, optim

from models.redes import Autoencoder

from .core.aggregators import clone_state_dict
from .core.runner import Client


@dataclass
class ClientArtifactRecord:
    """Metadata describing a stored client artefact."""

    client_id: int
    data_type: Optional[str]
    storage_key: str
    model_path: Optional[Path]
    metadata_path: Optional[Path]
    metadata: Optional[Dict[str, object]] = None


@dataclass
class AllAlgorithmArtifacts:
    """Container holding artefacts recovered for a stored federated run."""

    clients_dir: Path
    global_model_path: Optional[Path]
    clients: Dict[str, ClientArtifactRecord]


def _client_storage_key(index: int, data_type: str | None) -> str:
    """Reproduce the identifier scheme used when persisting client artefacts."""

    if data_type is None:
        return str(index)
    return f"{index}_{data_type}"


def _normalise_filename_fragment(value: str) -> str:
    """Convert arbitrary strings into safe filename fragments."""

    slug = value.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9_-]", "-", slug)


def load_all_algorithm_artifacts(
        run_dir: Path,
        clients: Sequence[Client],
        global_model: torch.nn.Module,
        device: torch.device,
) -> AllAlgorithmArtifacts:
    """Populate ``clients`` and ``global_model`` with stored artefacts."""

    clients_dir = run_dir / "clients"
    if not clients_dir.exists():
        raise FileNotFoundError(f"No clients/ directory found in {run_dir}")

    global_model_path = clients_dir / "global_model.pt"
    global_state = None
    if global_model_path.exists():
        global_state = torch.load(global_model_path, map_location=device)
        global_model.load_state_dict(global_state)

    records: Dict[str, ClientArtifactRecord] = {}
    for client in clients:
        training = client.training
        data_type = getattr(training, "data_type", None)
        storage_key = _client_storage_key(client.client_id, data_type)
        model_path = clients_dir / f"client_{storage_key}.pt"
        metadata_path = clients_dir / f"client_{storage_key}_meta.json"

        metadata: Optional[Dict[str, object]] = None
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)
            training.trainer.model.load_state_dict(state_dict)
            if global_state is None:
                global_state = clone_state_dict(training.trainer.model.state_dict())
        elif global_state is not None:
            training.trainer.model.load_state_dict(global_state)

        records[storage_key] = ClientArtifactRecord(
            client_id=client.client_id,
            data_type=data_type,
            storage_key=storage_key,
            model_path=model_path if model_path.exists() else None,
            metadata_path=metadata_path if metadata_path.exists() else None,
            metadata=metadata,
        )

    if global_state is not None and not global_model_path.exists():
        # Persist the global model state in case it was reconstructed from client data.
        global_model.load_state_dict(global_state)

    return AllAlgorithmArtifacts(
        clients_dir=clients_dir,
        global_model_path=global_model_path if global_model_path.exists() else None,
        clients=records,
    )


def load_flprotector_artifacts(
        run_dir: Path,
        clients: Sequence[Client],
        global_model: torch.nn.Module,
        device: torch.device,
) -> Dict[str, object]:
    """Populate client and global models from a stored FLProtector run."""

    artifacts = load_all_algorithm_artifacts(run_dir, clients, global_model, device)
    clients_dir = artifacts.clients_dir

    client_records: List[Dict[str, object]] = []

    for client in clients:
        training = client.training
        data_type = getattr(training, "data_type", None)
        storage_key = _client_storage_key(client.client_id, data_type)
        record = artifacts.clients.get(storage_key)

        encoder_path = clients_dir / f"client_{storage_key}_encoder.pt"
        if encoder_path.exists():
            trainer = training.trainer
            trainer.encoder = Autoencoder().to(device)
            trainer.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            trainer.criterion_encoder = nn.MSELoss(reduction="none")
            trainer.optimizer_encoder = optim.Adam(trainer.encoder.parameters(), lr=0.001)

        metadata = record.metadata if record else None
        if metadata:
            mean_value = metadata.get("mean_encoder_train")
            std_value = metadata.get("std_encoder_train")
            if mean_value is not None:
                training.trainer.mean_encoder_train = mean_value
            if std_value is not None:
                training.trainer.std_encoder_train = std_value

        client_records.append(
            {
                "client_id": client.client_id,
                "data_type": data_type,
                "storage_key": storage_key,
                "model_path": str(record.model_path) if record and record.model_path else None,
                "encoder_path": str(encoder_path) if encoder_path.exists() else None,
                "metadata_path": str(record.metadata_path) if record and record.metadata_path else None,
            }
        )

    return {
        "clients_dir": str(clients_dir),
        "global_model_path": str(artifacts.global_model_path) if artifacts.global_model_path else None,
        "clients": client_records,
    }


def run_flprotector_final_evaluation(
        algorithm,
        clients: Sequence[Client],
        global_model: torch.nn.Module,
        ablation_mode: str,
) -> Dict[str, object]:
    """Replicate the algorithm-side FLProtector sample-wise final evaluation."""

    mode = (getattr(algorithm, "ablation_mode", None) or ablation_mode or "baseline").lower()
    global_state = clone_state_dict(global_model.state_dict())

    try:
        device = next(global_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    domain_map = {0: "mnist", 1: "mnistm", 2: "svhn", 3: "syn", 4: "usps"}

    totals = {
        "total_samples": 0,
        "testing_correct": 0,
        "real_correct": 0,
        "best_correct": 0,
        "selection_correct": 0,
        "selection_total": 0,
    }

    metrics_clients: Dict[str, Dict[str, float]] = {}

    for client in clients:
        training = client.training
        trainer = training.trainer
        testloader = getattr(training, "testloader", None)
        if testloader is None:
            continue

        personal_state = clone_state_dict(trainer.model.state_dict())

        base_model = copy.deepcopy(trainer.model).to(device)
        base_model.load_state_dict(global_state)
        base_model.eval()

        personal_model = None
        if personal_state is not None:
            personal_model = copy.deepcopy(trainer.model).to(device)
            personal_model.load_state_dict(personal_state)
            personal_model.eval()

        encoder = getattr(trainer, "encoder", None)
        criterion = getattr(trainer, "criterion_encoder", None)
        encoder_active = (
            mode in {"baseline", "no_lbfgs"}
            and encoder is not None
            and criterion is not None
        )

        if encoder_active:
            encoder = encoder.to(device)
            encoder_mode = encoder.training
            encoder.eval()
            threshold = float(
                getattr(trainer, "mean_encoder_train", 0.0)
                + 3 * getattr(trainer, "std_encoder_train", 0.0)
            )
        else:
            encoder_mode = None
            threshold = None

        total_samples = 0
        testing_correct = 0
        real_correct = 0
        best_correct = 0
        selection_correct = 0
        selection_total = 0

        with torch.no_grad():
            for batch in testloader:
                if len(batch) < 3:
                    inputs, labels = batch[0], batch[1]
                    domains = torch.full_like(labels, -1)
                else:
                    inputs, labels, domains = batch

                inputs = inputs.to(device).float()
                labels = labels.to(device)
                domains = domains.to(device)

                if encoder_active:
                    resized = F.interpolate(
                        inputs,
                        size=(32, 32),
                        mode="bilinear",
                        align_corners=False,
                    )
                    recon = encoder(resized)
                    loss = criterion(recon, resized)
                    error = loss.view(loss.size(0), -1).mean(dim=1)
                    use_personal_auto = error <= threshold
                else:
                    use_personal_auto = torch.zeros(
                        labels.size(0),
                        device=labels.device,
                        dtype=torch.bool,
                    )

                global_logits = base_model(inputs)
                _, global_pred = torch.max(global_logits, 1)

                if personal_model is not None:
                    personal_logits = personal_model(inputs)
                    _, personal_pred = torch.max(personal_logits, 1)
                else:
                    personal_pred = global_pred

                batch_size = labels.size(0)

                if personal_model is None:
                    real_use_personal_tensor = torch.zeros(
                        batch_size, device=labels.device, dtype=torch.bool
                    )
                elif mode in {"baseline", "no_lbfgs"}:
                    domain_names = [
                        domain_map.get(int(idx), None)
                        for idx in domains.detach().cpu().tolist()
                    ]
                    real_use_personal = [
                        domain_name == client.data_type for domain_name in domain_names
                    ]
                    real_use_personal_tensor = torch.tensor(
                        real_use_personal,
                        device=labels.device,
                        dtype=torch.bool,
                    )
                elif mode == "no_autoencoder":
                    real_use_personal_tensor = torch.ones(
                        batch_size, device=labels.device, dtype=torch.bool
                    )
                else:
                    real_use_personal_tensor = torch.zeros(
                        batch_size, device=labels.device, dtype=torch.bool
                    )

                if personal_model is None:
                    final_pred = global_pred
                elif encoder_active:
                    final_pred = torch.where(use_personal_auto, personal_pred, global_pred)
                    selection_correct += (use_personal_auto == real_use_personal_tensor).sum().item()
                    selection_total += labels.size(0)
                elif mode == "no_autoencoder":
                    final_pred = personal_pred
                else:
                    final_pred = global_pred

                testing_correct += (final_pred == labels).sum().item()

                real_pred = torch.where(real_use_personal_tensor, personal_pred, global_pred)
                real_correct += (real_pred == labels).sum().item()

                if personal_model is not None:
                    best_correct += ((global_pred == labels) | (personal_pred == labels)).sum().item()
                else:
                    best_correct += (global_pred == labels).sum().item()

                total_samples += labels.size(0)

        if encoder_active and encoder_mode:
            encoder.train()

        trainer.model.load_state_dict(global_state)

        if total_samples == 0:
            continue

        client_entry: Dict[str, float] = {
            "client_id": client.client_id,
            "data_type": client.data_type,
            "total_samples": total_samples,
            "testing_accuracy": testing_correct / total_samples,
            "real_accuracy": real_correct / total_samples,
            "possible_best_accuracy": best_correct / total_samples,
        }

        if threshold is not None:
            client_entry["encoder_threshold"] = threshold
        if selection_total > 0:
            client_entry["selection_accuracy"] = selection_correct / selection_total

        metrics_clients[str(client.client_id)] = client_entry

        totals["total_samples"] += total_samples
        totals["testing_correct"] += testing_correct
        totals["real_correct"] += real_correct
        totals["best_correct"] += best_correct
        totals["selection_correct"] += selection_correct
        totals["selection_total"] += selection_total

    overall: Dict[str, float] = {
        "total_samples": totals["total_samples"],
        "testing_accuracy": (
            totals["testing_correct"] / totals["total_samples"]
            if totals["total_samples"]
            else 0.0
        ),
        "real_accuracy": (
            totals["real_correct"] / totals["total_samples"]
            if totals["total_samples"]
            else 0.0
        ),
        "possible_best_accuracy": (
            totals["best_correct"] / totals["total_samples"]
            if totals["total_samples"]
            else 0.0
        ),
    }

    if totals["selection_total"] > 0:
        overall["selection_accuracy"] = (
            totals["selection_correct"] / totals["selection_total"]
        )

    return {
        "mode": mode,
        "clients": metrics_clients,
        "overall": overall,
        "ablation_mode_slug": _normalise_filename_fragment(mode),
        "num_clients": len(metrics_clients),
    }


def _apply_backdoor_pattern(batch: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``batch`` with a 2x2 trigger stamped on the bottom-right."""

    patched = batch.clone()
    if patched.dim() < 4:
        return patched
    _, _, height, width = patched.shape
    r1 = max(height - 2, 0)
    c1 = max(width - 2, 0)
    patched[..., r1:height, c1:width] = 1.0
    return patched


def run_backdoor_attack_evaluation(
        clients: Sequence[Client],
        target_label: int,
        device: Optional[torch.device] = None,
        *,
        global_model: Optional[torch.nn.Module] = None,
        ablation_mode: Optional[str] = None,
) -> Dict[str, object]:
    """Measure standard and backdoor accuracies using FLProtector inference rules."""

    device = device or torch.device("cpu")
    mode = (ablation_mode or "baseline").lower()
    global_state = (
        clone_state_dict(global_model.state_dict())
        if global_model is not None
        else None
    )

    def _autoencoder_mask(
            encoder_module: nn.Module,
            criterion_module: nn.Module,
            threshold_value: float,
            batch_tensor: torch.Tensor,
    ) -> torch.Tensor:
        resized = F.interpolate(
            batch_tensor,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        recon = encoder_module(resized)
        loss = criterion_module(recon, resized)
        error = loss.view(loss.size(0), -1).mean(dim=1)
        return error <= threshold_value

    metrics_clients: Dict[str, Dict[str, float]] = {}

    totals = {
        "normal_correct": 0,
        "normal_total": 0,
        "backdoor_hits": 0,
        "backdoor_total": 0,
    }

    for client in clients:
        training = client.training
        testloader = getattr(training, "testloader", None)
        if testloader is None:
            continue

        trainer = training.trainer
        personal_state = clone_state_dict(trainer.model.state_dict())

        base_model = copy.deepcopy(trainer.model).to(device)
        if global_state is not None:
            base_model.load_state_dict(global_state)
        elif personal_state is not None:
            base_model.load_state_dict(personal_state)
        base_model.eval()

        personal_model: Optional[nn.Module] = None
        if personal_state is not None and mode != "no_personalization":
            personal_model = copy.deepcopy(trainer.model).to(device)
            personal_model.load_state_dict(personal_state)
            personal_model.eval()

        encoder = getattr(trainer, "encoder", None)
        criterion = getattr(trainer, "criterion_encoder", None)
        encoder_active = (
            personal_model is not None
            and mode in {"baseline", "no_lbfgs"}
            and encoder is not None
            and criterion is not None
        )

        if encoder_active:
            encoder = encoder.to(device)
            encoder_mode = encoder.training
            encoder.eval()
            threshold = float(
                getattr(trainer, "mean_encoder_train", 0.0)
                + 3 * getattr(trainer, "std_encoder_train", 0.0)
            )
        else:
            encoder_mode = None
            threshold = None

        normal_correct = 0
        normal_total = 0
        backdoor_hits = 0
        backdoor_total = 0

        target_tensor: Optional[torch.Tensor] = None

        with torch.no_grad():
            for batch in testloader:
                inputs, labels = batch[0], batch[1]
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                batch_size = labels.size(0)

                global_logits = base_model(inputs)
                global_pred = torch.argmax(global_logits, dim=1)

                if personal_model is not None:
                    personal_logits = personal_model(inputs)
                    personal_pred = torch.argmax(personal_logits, dim=1)
                else:
                    personal_pred = global_pred

                if personal_model is None:
                    final_pred_clean = global_pred
                elif encoder_active:
                    use_personal_clean = _autoencoder_mask(encoder, criterion, threshold, inputs)
                    final_pred_clean = torch.where(use_personal_clean, personal_pred, global_pred)
                elif mode == "no_autoencoder":
                    final_pred_clean = personal_pred
                else:
                    final_pred_clean = global_pred

                normal_correct += (final_pred_clean == labels).sum().item()
                normal_total += batch_size

                poisoned_inputs = _apply_backdoor_pattern(inputs)

                poisoned_global_logits = base_model(poisoned_inputs)
                poisoned_global_pred = torch.argmax(poisoned_global_logits, dim=1)

                if personal_model is not None:
                    poisoned_personal_logits = personal_model(poisoned_inputs)
                    poisoned_personal_pred = torch.argmax(poisoned_personal_logits, dim=1)
                else:
                    poisoned_personal_pred = poisoned_global_pred

                if target_tensor is None or target_tensor.size(0) != batch_size:
                    target_tensor = labels.new_full((batch_size,), int(target_label))

                if personal_model is None:
                    final_pred_poisoned = poisoned_global_pred
                elif encoder_active:
                    use_personal_poisoned = _autoencoder_mask(
                        encoder,
                        criterion,
                        threshold,
                        poisoned_inputs,
                    )
                    final_pred_poisoned = torch.where(
                        use_personal_poisoned,
                        poisoned_personal_pred,
                        poisoned_global_pred,
                    )
                elif mode == "no_autoencoder":
                    final_pred_poisoned = poisoned_personal_pred
                else:
                    final_pred_poisoned = poisoned_global_pred

                backdoor_hits += (final_pred_poisoned == target_tensor).sum().item()
                backdoor_total += batch_size

        if encoder_active and encoder_mode:
            encoder.train()

        if normal_total == 0:
            continue

        metrics_clients[str(client.client_id)] = {
            "client_id": client.client_id,
            "data_type": getattr(training, "data_type", None),
            "normal_accuracy": normal_correct / normal_total,
            "backdoor_success_rate": (backdoor_hits / backdoor_total) if backdoor_total else 0.0,
            "normal_total": normal_total,
            "backdoor_total": backdoor_total,
            "backdoor_hits": backdoor_hits,
        }

        totals["normal_correct"] += normal_correct
        totals["normal_total"] += normal_total
        totals["backdoor_hits"] += backdoor_hits
        totals["backdoor_total"] += backdoor_total

    overall = {
        "normal_accuracy": (
            totals["normal_correct"] / totals["normal_total"]
            if totals["normal_total"]
            else 0.0
        ),
        "backdoor_success_rate": (
            totals["backdoor_hits"] / totals["backdoor_total"]
            if totals["backdoor_total"]
            else 0.0
        ),
        "normal_total": totals["normal_total"],
        "backdoor_total": totals["backdoor_total"],
        "backdoor_hits": totals["backdoor_hits"],
    }

    return {
        "target_label": int(target_label),
        "clients": metrics_clients,
        "overall": overall,
        "num_clients": len(metrics_clients),
        "mode": mode,
    }


__all__ = [
    "AllAlgorithmArtifacts",
    "ClientArtifactRecord",
    "load_all_algorithm_artifacts",
    "load_flprotector_artifacts",
    "run_backdoor_attack_evaluation",
    "run_flprotector_final_evaluation",
]