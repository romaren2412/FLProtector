# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0
"""
Federated Averaging (FedAvg)
-----------------------------
Re-implemented from the paper:

H. Brendan McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Arcas,
"Communication-Efficient Learning of Deep Networks from Decentralized Data,"
Proceedings of AISTATS 2017, pp. 1273–1282.
URL: https://arxiv.org/abs/1602.05629

This implementation follows the algorithmic description in the paper,
with adaptations for compatibility with FLProtector.
"""


from __future__ import annotations

"""Federated Averaging baseline with optional delta personalization."""

from collections import OrderedDict
from typing import Dict, Sequence

import torch

from utils.evaluate_utils import evaluar_local_models, seleccion_representante
from utils.file_utils import save_clients, save_clients_global
from utils.seed_utils import set_seed
from utils.training_utils import adestramento_delta
from ..core.runner import Client, FederatedAlgorithm


class FedAvgAlgorithm(FederatedAlgorithm):
    """Plain Federated Averaging with optional delta personalization rounds.

        Parameters
        ----------
        config:
            Exposes ``DELTA_PERS`` to toggle the delta fine-tuning flow,
            ``CHECK_PREC`` to schedule evaluations, ``PATH`` for persistence and
            ``TIPO_EXEC`` for CLI discovery. ``set_seed(42)`` is invoked so the
            client sampling and training pipelines are reproducible across runs.
    """

    def __init__(self, config) -> None:
        super().__init__()
        set_seed(42)
        self.config = config
        self.local_precisions = []
        self.delta_precisions = []
        self.representatives = None

    def attach(self, runner) -> None:
        """Select representative clients used during evaluation."""
        super().attach(runner)
        trainings = [client.training for client in runner.clients]
        self.representatives = seleccion_representante(trainings)

    def client_step(
            self,
            global_model,
            client: Client,
            epoch: int,
    ) -> Dict[str, OrderedDict[str, torch.Tensor]]:
        client.training.trainer.adestrar()
        return {"state_dict": client.model_state()}

    def after_aggregation(
            self,
            epoch: int,
            clients: Sequence[Client],
            aggregation,
    ) -> None:
        """Trigger personalization or evaluation depending on configuration."""
        trainings = [client.training for client in clients]
        if self.config.DELTA_PERS:
            adestramento_delta(
                trainings,
                epoch,
                self.local_precisions,
                self.delta_precisions,
                self.config.CHECK_PREC,
                self.config.PATH,
            )
        elif (epoch + 1) % self.config.CHECK_PREC == 0:
            evaluar_local_models(epoch, self.representatives, self.config.PATH, self.local_precisions)

    def on_complete(self) -> None:
        """Persist client artifacts (delta or global models) at the end."""
        trainings = [client.training for client in self.runner.clients]
        if self.config.DELTA_PERS:
            save_clients(self.config.PATH, trainings)
        else:
            save_clients_global(self.config.PATH, self.representatives)
