# SPDX-FileCopyrightText: 2021, ki-ljl
# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: MIT
# SPDX-Comment: Derived from https://github.com/ki-ljl/Scaffold-Federated-Learning (commit 6114b73).


"""SCAFFOLD algorithm with control variate tracking for personalization."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from utils.file_utils import save_clients_global
from utils.evaluate_utils import evaluar_local_models, seleccion_representante
from utils.seed_utils import set_seed
from ..core.aggregators import AggregationResult, Aggregator, clone_state_dict
from ..core.runner import Client, FederatedAlgorithm


def _c_client_norm(c_client: Dict[str, torch.Tensor]) -> float:
    """Return the L2 norm of the client control variate."""
    norm = 0.0
    for tensor in c_client.values():
        norm += (tensor ** 2).sum().item()
    return float(torch.sqrt(torch.tensor(norm)))


def _c_client_drift(prev: Dict[str, torch.Tensor], current: Dict[str, torch.Tensor]) -> float:
    """Measure the difference between consecutive control variates."""
    drift = 0.0
    for name in prev.keys():
        diff = current[name] - prev[name]
        drift += (diff ** 2).sum().item()
    return float(torch.sqrt(torch.tensor(drift)))


def _copy_c_client(c_client: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep copy a control variate without tracking gradients."""
    return {name: tensor.clone().detach() for name, tensor in c_client.items()}


class ScaffoldAlgorithm(FederatedAlgorithm):
    """SCAFFOLD

        Parameters
        ----------
        config:
            Relies on ``CHECK_PREC`` to trigger evaluations, ``PATH`` for history
            serialization and ``DEVICE`` for model management. ``set_seed(42)`` is
            invoked to stabilise client selection and stochastic trainers.
        """

    def __init__(self, config) -> None:
        super().__init__()
        set_seed(42)
        self.config = config
        self.local_precisions = []
        self.representatives = None
        self.c_server: Dict[str, torch.Tensor] = {}
        self.prev_c_clients: Dict[int, Dict[str, torch.Tensor]] = {}

    def attach(self, runner) -> None:
        """Initialise the global control variate and evaluation helpers."""
        super().attach(runner)
        self.c_server = {
            name: torch.zeros_like(param)
            for name, param in runner.global_model.named_parameters()
        }
        trainings = [client.training for client in runner.clients]
        self.representatives = seleccion_representante(trainings)

    def before_round(self, epoch: int, clients: Sequence[Client]) -> None:
        """Capture previous control variates before client updates occur."""
        self.prev_c_clients = {client.client_id: _copy_c_client(client.training.trainer.c_client) for client in clients}

    def client_step(self, global_model: torch.nn.Module, client: Client, epoch: int):
        """Run SCAFFOLD's variance-reduced update on the local client."""
        delta_y, delta_c = client.training.trainer.adestrar(c_server=self.c_server)
        return {
            "state_dict": client.model_state(),
            "delta_y": delta_y,
            "delta_c": delta_c,
        }

    def aggregate(
            self,
            global_model: torch.nn.Module,
            clients: Sequence[Client],
            payloads: Sequence[Dict[str, Dict[str, torch.Tensor]]],
            aggregator: Aggregator,
    ) -> AggregationResult:
        """Update both model parameters and the server control variate."""
        with torch.no_grad():
            # Training params only
            param_names = [name for name, _ in global_model.named_parameters()]
            state = global_model.state_dict()
            for name in param_names:
                avg_delta = torch.mean(torch.stack([payload["delta_y"][name] for payload in payloads]), dim=0)
                state[name].add_(avg_delta)
            for name in self.c_server.keys():
                avg_delta_c = torch.mean(torch.stack([payload["delta_c"][name] for payload in payloads]), dim=0)
                self.c_server[name].add_(avg_delta_c)
        return AggregationResult(state_dict=clone_state_dict(global_model.state_dict()))

    def after_aggregation(self, epoch: int, clients: Sequence[Client], aggregation) -> None:
        """Log control variate statistics and schedule evaluations."""
        if (epoch + 1) % self.config.CHECK_PREC == 0:
            evaluar_local_models(epoch, self.representatives, self.config.PATH, self.local_precisions)

    def on_complete(self) -> None:
        """Persist the representative models gathered during training."""
        save_clients_global(self.config.PATH, self.representatives)
