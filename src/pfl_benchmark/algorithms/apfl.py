# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0
"""
Adaptive Personalized Federated Learning (APFL)
------------------------------------------------
Re-implemented from the paper:

M. Deng, K. Kamani, and M. Mahdavi,
"Adaptive Personalized Federated Learning,"
arXiv:2003.13461, 2020.
URL: https://arxiv.org/abs/2003.13461

This implementation follows the algorithmic description in the paper,
with modifications for compatibility with FLProtector.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import OrderedDict
from typing import Dict, MutableMapping, Sequence

import numpy as np
import torch
import torch.optim as optim

from utils.file_utils import save_alpha, save_clients_apfl, save_accuracies
from utils.seed_utils import set_seed
from ..core.runner import Client, FederatedAlgorithm

logger = logging.getLogger(__name__)


def _alpha_update(
        v_state: MutableMapping[str, torch.Tensor],
        w_state: MutableMapping[str, torch.Tensor],
        grad_v_tilde: MutableMapping[str, torch.Tensor],
        alpha: float,
        lr: float,
        lr_alpha: float = 1e-5,
) -> float:
    """Perform the personalization weight update used by APFL."""
    grad_alpha = 0.0
    for name in v_state.keys():
        dif = v_state[name] - w_state[name]
        grad_alpha += torch.dot(dif.view(-1), (grad_v_tilde[name] / lr).view(-1))
    alpha = alpha - lr_alpha * grad_alpha
    alpha = float(np.clip(alpha.item() if isinstance(alpha, torch.Tensor) else alpha, 0.0, 1.0))
    return alpha


def _v_update(
        v_state: MutableMapping[str, torch.Tensor],
        grad_v_tilde: MutableMapping[str, torch.Tensor],
        alpha: float,
) -> Dict[str, torch.Tensor]:
    """Gradient step that updates the personalized model copy ``v``."""
    updated = {}
    for name in v_state.keys():
        updated[name] = v_state[name] + alpha * grad_v_tilde[name]
    return updated


def _v_tilde(
        v_state: MutableMapping[str, torch.Tensor],
        w_state: MutableMapping[str, torch.Tensor],
        alpha: float,
) -> Dict[str, torch.Tensor]:
    """Blend global and personalized parameters using the current ``alpha``."""
    updated = {}
    for name in v_state.keys():
        updated[name] = alpha * v_state[name] + (1 - alpha) * w_state[name]
    return updated


class APFLAlgorithm(FederatedAlgorithm):
    """Adaptive Personalized Federated Learning implementation.

        Parameters
        ----------
        config:
            Must expose ``PATH`` for logging artifacts, ``K`` for client sampling,
            ``FL_FREQ`` for the number of local updates per round and ``CHECK_PREC``
            for evaluation cadence. The constructor intentionally seeds the global
            RNG state via :func:`set_seed` to guarantee deterministic behaviour
            across runs.
        """

    def __init__(self, config) -> None:
        super().__init__()
        set_seed(42)
        self.config = config
        self.precisions = []
        self.alpha_history = []
        self.active_clients: Sequence[Client] = []
        self.last_gradients: Dict[int, Dict[str, torch.Tensor]] = {}

    def attach(self, runner) -> None:
        """Initialise client-side buffers for APFL's personalized copies."""
        super().attach(runner)
        for client in runner.clients:
            training = client.training
            training.delta.load_state_dict(training.trainer.model.state_dict())
            training.delta_tilde = copy.deepcopy(training.delta)
            training.delta_tilde.load_state_dict(training.trainer.model.state_dict())
            training.trainer.delta_optimizer = optim.Adam(training.delta_tilde.parameters(), lr=training.c.LR)
            training.alpha = 0.5
        self.active_clients = random.sample(list(runner.clients), k=min(self.config.K, len(runner.clients)))

    def select_clients(self, clients: Sequence[Client], epoch: int) -> Sequence[Client]:
        self.active_clients = random.sample(list(clients), k=min(self.config.K, len(clients)))
        selected_ids = [c.client_id for c in self.active_clients]
        logger.info("Selected clients for epoch %s: %s", epoch, selected_ids)
        return self.active_clients

    def client_step(
            self,
            global_model: torch.nn.Module,
            client: Client,
            epoch: int,
    ) -> Dict[str, OrderedDict[str, torch.Tensor]]:
        """Alternate local updates between the main and personalized models."""
        training = client.training
        for _ in range(self.config.FL_FREQ):
            training.trainer.adestrar(epochs=1)
            grad_v_tilde = training.trainer.adestrar(epochs=1, rede=training.delta_tilde, delta=True)
            v_state = training.delta.state_dict()
            updated_v = _v_update(v_state, grad_v_tilde, training.alpha)
            training.delta.load_state_dict(updated_v)
        self.last_gradients[client.client_id] = grad_v_tilde

        return {
            "state_dict": client.model_state(),
        }

    def after_aggregation(self, epoch: int, clients: Sequence[Client], aggregation) -> None:
        """Update personalized buffers and record metrics after global sync."""
        global_state = aggregation.state_dict
        alphas = []
        accuracies = []
        for client in self.runner.clients:
            training = client.training
            grad_dict = self.last_gradients.get(client.client_id)
            if grad_dict is not None:
                v_state = training.delta.state_dict()
                alpha = _alpha_update(v_state, global_state, grad_dict, training.alpha, training.c.LR)
                training.alpha = alpha
            v_tilde_state = _v_tilde(training.delta.state_dict(), global_state, training.alpha)
            training.delta_tilde.load_state_dict(v_tilde_state)
            training.trainer.model.load_state_dict(training.delta_tilde.state_dict())
            if ((epoch + 1) % self.config.CHECK_PREC) == 0:
                acc = training.testear()
                logger.info("[Client %s - %s] Local accuracy: %.4f", client.client_id, client.data_type, acc)
                accuracies.append(acc)
            alphas.append(training.alpha)
            training.trainer.model.load_state_dict(global_state)
        if accuracies:
            self.precisions.append([epoch] + accuracies)
            self.alpha_history.append([epoch] + alphas)
        self.active_clients = random.sample(list(self.runner.clients), k=min(self.config.K, len(self.runner.clients)))
        self.last_gradients = {}

    def on_complete(self) -> None:
        """Persist accumulated metrics once training finishes."""
        save_accuracies(self.config.PATH, self.precisions, [c.training for c in self.runner.clients])
        save_alpha(self.config.PATH, self.alpha_history)
        save_clients_apfl(self.config.PATH, [c.training for c in self.runner.clients])
