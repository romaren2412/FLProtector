# SPDX-FileCopyrightText: 2020, Tian Li et al.
# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: MIT
# SPDX-Comment: Derived from https://github.com/litian96/FedProx (commit d2a4501).


"""FedProx implementation tracking proximal terms and client drift."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence

import torch

from utils.file_utils import save_clients_global
from utils.evaluate_utils import evaluar_local_models, seleccion_representante
from utils.seed_utils import set_seed
from ..core.runner import Client, FederatedAlgorithm


class FedProxAlgorithm(FederatedAlgorithm):
    """FedProx Algorithm

        Parameters
        ----------
        config:
            Requires ``MU`` for the proximal penalty, ``CHECK_PREC`` for evaluation
            cadence, ``PATH`` for persistence and ``DEVICE`` when storing artifacts.
            ``set_seed(42)`` is applied to keep the local solver deterministic.
        """

    def __init__(self, config) -> None:
        super().__init__()
        set_seed(42)
        self.config = config
        self.local_precisions = []
        self.representatives = None

    def attach(self, runner) -> None:
        """Choose evaluation representatives used throughout training."""
        super().attach(runner)
        trainings = [client.training for client in runner.clients]
        self.representatives = seleccion_representante(trainings)

    def client_step(
            self,
            global_model: torch.nn.Module,
            client: Client,
            epoch: int,
    ) -> Dict[str, OrderedDict[str, torch.Tensor]]:
        global_params = {key: value.clone().detach() for key, value in global_model.state_dict().items()}
        client.training.trainer.adestrar(global_params=global_params, mu=self.config.MU)
        return {"state_dict": client.model_state()}

    def after_aggregation(self, epoch: int, clients: Sequence[Client], aggregation) -> None:
        """Persist history logs and run periodic evaluation."""
        if (epoch + 1) % self.config.CHECK_PREC == 0:
            evaluar_local_models(epoch, self.representatives, self.config.PATH, self.local_precisions)

    def on_complete(self) -> None:
        """Store the final set of representative client models."""
        save_clients_global(self.config.PATH, self.representatives)
