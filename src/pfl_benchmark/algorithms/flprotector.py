# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""FLProtector algorithm that reweights clients using trust scores."""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.redes import Autoencoder
from utils.file_utils import (
    save_clients_flprotector,
    save_flprotector_metrics,
    save_weight_scores,
    save_scores,
)
from utils.lbfgs_utils import calculo_FLDet
from utils.evaluate_utils import seleccion_representante, evaluar_local_models
from utils.seed_utils import set_seed
from utils.training_utils import adestramento_delta
from ..core.aggregators import AggregationResult, Aggregator, clone_state_dict
from ..core.runner import Client, FederatedAlgorithm

logger = logging.getLogger(__name__)


class FLProtectorAlgorithm(FederatedAlgorithm):
    """Federated defence using anomaly detection and adaptive weighting.

        Parameters
        ----------
        config:
            Needs ``CHECK_PREC`` for personalization checks, ``PATH`` for artifact
            storage, ``EPOCH_ENCODER`` for the final autoencoder training stage and
            ``DEVICE`` for model placement. The constructor seeds the RNG with
            ``set_seed(42)`` because the detection pipeline relies on consistent
            client ordering and sampling.
        """

    def __init__(self, config, ablation_mode: Optional[str] = None) -> None:
        super().__init__()
        set_seed(42)
        self.config = config
        self.ablation_mode = ablation_mode or self._resolve_ablation_mode(config)
        self.grad_list: List[List[torch.Tensor]] = []
        self.old_grad_list: List[List[torch.Tensor]] = []
        self.weight_record: List[torch.Tensor] = []
        self.grad_record: List[torch.Tensor] = []
        self.mal_score: List[float] = []
        self.pond_scores_array: List[List[float]] = []
        self.local_precisions = []
        self.delta_precisions = []
        self.last_weight = None
        self.last_grad = None

    def before_round(self, epoch: int, clients: Sequence[Client]) -> None:
        """Reset round-specific gradient buffers before collecting updates."""
        self.grad_list = []
        self.current_epoch = epoch

    def client_step(
            self,
            global_model: torch.nn.Module,
            client: Client,
            epoch: int,
    ) -> Dict[str, OrderedDict[str, torch.Tensor]]:
        """Collect client gradients to evaluate trustworthiness."""
        grad_update = client.training.trainer.adestrar()
        gradient_vector = [grad_update[name].clone().detach() for name in grad_update]
        self.grad_list.append(gradient_vector)
        return {"state_dict": client.model_state()}

    def aggregate(
            self,
            global_model: torch.nn.Module,
            clients: Sequence[Client],
            payloads: Sequence[Dict[str, OrderedDict[str, torch.Tensor]]],
            aggregator: Aggregator,
    ) -> AggregationResult:
        """Aggregate client updates using L-BFGS adapted-derived trust scores."""
        ablation_mode = getattr(self.config, "ABLATION_MODE", None)
        if ablation_mode == "no_lbfgs":
            aggregation = aggregator.aggregate(
                [payload["state_dict"] for payload in payloads]
            )
            self.last_weight = None
            self.last_grad = None
            self.weight_record.clear()
            self.grad_record.clear()
            self.old_grad_list = []
            return aggregation

        # L-BFGS-based trust score calculation
        distance, trust_scores, param_list, weight, grad = calculo_FLDet(
            getattr(self, "current_epoch", 0),
            global_model,
            self.grad_list,
            self.last_weight,
            self.old_grad_list,
            self.weight_record,
            self.grad_record,
            self.config,
        )
        if distance is not None:
            self.mal_score.append(distance)
            save_scores(self.mal_score, self.config.PATH, [c.training for c in clients])
        aggregation = aggregator.aggregate(
            [payload["state_dict"] for payload in payloads],
            trust_scores,
        )
        if aggregation.weights is not None:
            self.pond_scores_array.append([float(w) for w in aggregation.weights])
            save_weight_scores(
                self.pond_scores_array,
                self.config.PATH,
                [c.training for c in clients],
            )
        if self.last_weight is not None:
            self.weight_record.append(weight - self.last_weight)
            self.grad_record.append(grad - self.last_grad)
            if len(self.weight_record) > 10:
                self.weight_record.pop(0)
                self.grad_record.pop(0)
        self.last_weight = weight
        self.last_grad = grad
        self.old_grad_list = param_list
        return aggregation

    def after_aggregation(self, epoch: int, clients: Sequence[Client], aggregation) -> None:
        """Run personalization delta training using the robust global model."""
        self.current_epoch = epoch
        trainings = [client.training for client in clients]

        if self.ablation_mode == "no_personalization":
            if (epoch + 1) % self.config.CHECK_PREC == 0:
                representatives = seleccion_representante(trainings)
                evaluar_local_models(epoch, representatives, self.config.PATH, self.local_precisions)
            return

        adestramento_delta(
            trainings,
            epoch,
            self.local_precisions,
            self.delta_precisions,
            self.config.CHECK_PREC,
            self.config.PATH,
        )

    def on_complete(self) -> None:
        """Fine-tune detector autoencoders and serialize client metadata."""
        for client in self.runner.clients:
            trainer = client.training.trainer
            trainer.encoder = Autoencoder().to(self.config.DEVICE)
            trainer.criterion_encoder = nn.MSELoss(reduction='none')
            trainer.optimizer_encoder = optim.Adam(trainer.encoder.parameters(), lr=0.001)
            trainer.adestrar_encoder(epochs=self.config.EPOCH_ENCODER)
        evaluation = self._final_evaluation()
        save_flprotector_metrics(self.config.PATH, evaluation)
        save_clients_flprotector(
            self.config.PATH,
            [c.training for c in self.runner.clients],
            metrics=evaluation,
        )

    def _resolve_ablation_mode(self, config) -> str:
        """Extract the requested ablation mode from the runtime configuration."""
        algorithm_cfg = getattr(config, "algorithm", None)
        overrides = getattr(config, "overrides", {}) or {}
        metadata = getattr(config, "metadata", {}) or {}

        candidates = [
            getattr(config, "ablation_mode", None),
            getattr(algorithm_cfg, "ablation_mode", None),
            overrides.get("algorithm.ablation_mode"),
            metadata.get("config", {}).get("algorithm", {}).get("ablation_mode"),
        ]

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return "baseline"

    def _run_personalization_stage(self, trainings: Sequence, epoch: int) -> None:
        """Invoke the delta-personalization routine respecting the ablation mode."""
        adestramento_delta(
            trainings,
            epoch,
            self.local_precisions,
            self.delta_precisions,
            self.config.CHECK_PREC,
            self.config.PATH,
            mode=self.ablation_mode,
        )

    def _evaluate_state(
            self,
            client: Client,
            state_dict: OrderedDict[str, torch.Tensor],
    ) -> float:
        """Load ``state_dict`` into ``client`` and return its accuracy."""
        client.load_model_state(state_dict)
        accuracy = client.test()
        return float(accuracy) if accuracy is not None else 0.0

    def _autoencoder_decision(
            self,
            client: Client,
    ) -> Tuple[str, Optional[float], Optional[float]]:
        """Use the trained autoencoder to decide between personalised/global models."""
        trainer = client.training.trainer
        encoder = getattr(trainer, "encoder", None)
        criterion = getattr(trainer, "criterion_encoder", None)
        testloader = getattr(client.training, "testloader", None)
        if encoder is None or criterion is None or testloader is None:
            return "global", None, None

        threshold = float(trainer.mean_encoder_train + 3 * trainer.std_encoder_train)
        was_training = encoder.training
        encoder.eval()
        losses: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in testloader:
                inputs = batch[0].to(self.config.DEVICE)
                inputs = transforms.Resize((32, 32), antialias=True)(inputs)
                outputs = encoder(inputs.float())
                loss = criterion(outputs, inputs.float())
                dims = tuple(range(1, loss.dim()))
                loss = loss.mean(dim=dims)
                losses.append(loss.detach().cpu())
        if was_training:
            encoder.train()

        if not losses:
            return "global", None, threshold

        recon_error = torch.cat([loss.view(-1) for loss in losses]).mean().item()
        decision = "personalized" if recon_error <= threshold else "global"
        logger.info(
            "Autoencoder decision for client %s (%s): %s [error=%.6f threshold=%.6f]",
            client.client_id,
            getattr(client, "data_type", "unknown"),
            decision,
            recon_error,
            threshold,
        )
        return decision, recon_error, threshold

    def _final_evaluation(self) -> Dict[str, object]:
        """Evaluate personalised vs global models using the trained encoder."""
        mode = (self.ablation_mode or "baseline").lower()
        global_state = clone_state_dict(self.runner.global_model.state_dict())

        return self._final_evaluation_samplewise(mode, global_state)

    def _final_evaluation_samplewise(
            self,
            mode: str,
            global_state: OrderedDict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """Compute sample-level metrics for the different ablation modes."""

        domain_map = {0: "mnist", 1: "mnistm", 2: "svhn", 3: "syn", 4: "usps"}
        device = self.config.DEVICE
        autoencoder_enabled = mode in {"baseline", "no_lbfgs"}
        auto_selection_mode = "encoder" if autoencoder_enabled else ("personal" if mode == "no_autoencoder" else "global")

        metrics_clients: Dict[str, Dict[str, float]] = {}
        totals = {
            "total_samples": 0,
            "testing_correct": 0,
            "real_correct": 0,
            "best_correct": 0,
            "selection_correct": 0,
            "selection_total": 0,
        }

        for client in self.runner.clients:
            training = client.training
            trainer = training.trainer
            testloader = getattr(training, "testloader", None)
            if testloader is None:
                continue

            encoder = getattr(trainer, "encoder", None)
            criterion = getattr(trainer, "criterion_encoder", None)
            evaluate_with_encoder = autoencoder_enabled and encoder is not None and criterion is not None

            threshold: Optional[float] = None
            if evaluate_with_encoder:
                threshold = float(trainer.mean_encoder_train + 3 * trainer.std_encoder_train)

            global_model = copy.deepcopy(trainer.model).to(device)
            global_model.load_state_dict(clone_state_dict(global_state))
            global_model.eval()

            personalization_state = client.personalization_state
            personalized_model = None
            if personalization_state is not None:
                personal_state = OrderedDict(
                    (name, global_state[name] + personalization_state[name]) for name in global_state
                )
                personalized_model = copy.deepcopy(trainer.model).to(device)
                personalized_model.load_state_dict(clone_state_dict(personal_state))
                personalized_model.eval()

            total_samples = 0
            testing_correct = 0
            real_correct = 0
            best_correct = 0
            selection_correct = 0
            selection_total = 0

            encoder_was_training = getattr(encoder, "training", False)
            if evaluate_with_encoder:
                encoder = encoder.to(device)
                encoder.eval()

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

                    resized = F.interpolate(inputs, size=(32, 32), mode="bilinear", align_corners=False)

                    if personalized_model is None:
                        use_personal_auto = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
                    elif auto_selection_mode == "personal":
                        use_personal_auto = torch.ones(labels.size(0), dtype=torch.bool, device=labels.device)
                    elif auto_selection_mode == "global":
                        use_personal_auto = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
                    elif evaluate_with_encoder and threshold is not None:
                        recon = encoder(resized.float())
                        loss = criterion(recon, resized.float())
                        error = loss.view(loss.size(0), -1).mean(dim=1)
                        use_personal_auto = error <= threshold
                    else:
                        use_personal_auto = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)

                    global_logits = global_model(inputs.float())
                    global_pred = torch.argmax(global_logits, dim=1)

                    if personalized_model is not None:
                        personal_logits = personalized_model(inputs.float())
                        personal_pred = torch.argmax(personal_logits, dim=1)
                    else:
                        personal_pred = global_pred

                    auto_pred = torch.where(use_personal_auto, personal_pred, global_pred)
                    testing_correct += (auto_pred == labels).sum().item()

                    domain_matches = torch.zeros_like(labels, dtype=torch.bool)
                    if personalized_model is not None and client.data_type is not None:
                        mapped = [domain_map.get(int(idx), None) if int(idx) in domain_map else None for idx in domains.cpu().tolist()]
                        domain_matches = torch.tensor(
                            [name == client.data_type for name in mapped],
                            device=labels.device,
                            dtype=torch.bool,
                        )

                    real_pred = torch.where(domain_matches, personal_pred, global_pred)
                    real_correct += (real_pred == labels).sum().item()

                    if personalized_model is not None:
                        best_correct += ((global_pred == labels) | (personal_pred == labels)).sum().item()
                        if auto_selection_mode == "encoder" and evaluate_with_encoder:
                            selection_correct += (use_personal_auto == domain_matches).sum().item()
                            selection_total += labels.size(0)
                    else:
                        best_correct += (global_pred == labels).sum().item()

                    total_samples += labels.size(0)

            if evaluate_with_encoder and encoder is not None and encoder_was_training:
                encoder.train()

            if total_samples == 0:
                client.load_model_state(clone_state_dict(global_state))
                continue

            client_entry: Dict[str, object] = {
                "client_id": client.client_id,
                "data_type": client.data_type,
                "total_samples": total_samples,
                "testing_accuracy": testing_correct / total_samples,
                "testing_correct": testing_correct,
                "real_accuracy": real_correct / total_samples,
                "real_correct": real_correct,
                "possible_best_accuracy": best_correct / total_samples,
                "possible_best_correct": best_correct,
            }
            if threshold is not None:
                client_entry["encoder_threshold"] = threshold
            if selection_total > 0:
                client_entry["selection_accuracy"] = selection_correct / selection_total
                client_entry["selection_correct"] = selection_correct
                client_entry["selection_total"] = selection_total

            metrics_clients[str(client.client_id)] = client_entry

            totals["total_samples"] += total_samples
            totals["testing_correct"] += testing_correct
            totals["real_correct"] += real_correct
            totals["best_correct"] += best_correct
            totals["selection_correct"] += selection_correct
            totals["selection_total"] += selection_total

            client.load_model_state(clone_state_dict(global_state))

        overall: Dict[str, float | int] = {
            "total_samples": totals["total_samples"],
            "testing_accuracy": (
                totals["testing_correct"] / totals["total_samples"] if totals["total_samples"] else 0.0
            ),
            "testing_correct": totals["testing_correct"],
            "real_accuracy": (
                totals["real_correct"] / totals["total_samples"] if totals["total_samples"] else 0.0
            ),
            "real_correct": totals["real_correct"],
            "possible_best_accuracy": (
                totals["best_correct"] / totals["total_samples"] if totals["total_samples"] else 0.0
            ),
            "possible_best_correct": totals["best_correct"],
        }
        if totals["selection_total"] > 0:
            overall["selection_accuracy"] = totals["selection_correct"] / totals["selection_total"]
            overall["selection_correct"] = totals["selection_correct"]
            overall["selection_total"] = totals["selection_total"]

        return {"mode": mode, "clients": metrics_clients, "overall": overall}