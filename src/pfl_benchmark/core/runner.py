# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from .aggregators import (
    AggregationResult,
    Aggregator,
    WeightedAveragingAggregator,
    clone_state_dict,
)

logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

EvaluationHook = Callable[["FederatedRunner", int], None]


@dataclass
class Client:
    """Wrapper for client specific state and behaviour."""

    client_id: int
    training: object
    data_type: Optional[str] = None

    def model_state(self) -> OrderedDict[str, torch.Tensor]:
        """Return a detached copy of the underlying trainer model state."""
        return clone_state_dict(self.training.trainer.model.state_dict())

    def load_model_state(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        """Overwrite the trainer model with the supplied parameters."""
        self.training.trainer.model.load_state_dict(state_dict)

    def train(self, **kwargs):
        """Proxy ``adestrar`` allowing algorithms to pass custom kwargs."""
        return self.training.trainer.adestrar(**kwargs)

    def test(self):
        """Run the client's evaluation routine if available."""
        if hasattr(self.training, "testear"):
            return self.training.testear()
        raise AttributeError("Client training object does not implement testear()")

    @property
    def personalization_state(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        """Return personalization parameters maintained by the training object."""
        return getattr(self.training, "personalization_buffer", None)

    @personalization_state.setter
    def personalization_state(self, state: OrderedDict[str, torch.Tensor]) -> None:
        """Persist personalization parameters when the training object supports it."""
        if hasattr(self.training, "personalization_buffer"):
            self.training.personalization_buffer = state
        else:
            raise AttributeError(
                "Client training object does not expose personalization_buffer"
            )

    @property
    def delta_state(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        """Return cached client delta updates when available."""
        return getattr(self.training, "delta_buffer", None)

    @delta_state.setter
    def delta_state(self, state: OrderedDict[str, torch.Tensor]) -> None:
        """Update the cached client delta when supported by the training object."""
        if hasattr(self.training, "delta_buffer"):
            self.training.delta_buffer = state
        else:
            raise AttributeError("Client training object does not expose delta_buffer")


@dataclass
class FederatedRunner:
    """Coordinates client training, aggregation and evaluation."""

    global_model: torch.nn.Module
    clients: Sequence[Client]
    algorithm: "FederatedAlgorithm"
    aggregator: Aggregator = field(default_factory=WeightedAveragingAggregator)
    epochs: int = 1
    evaluation_interval: int = 1
    evaluation_hooks: List[EvaluationHook] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Attach the algorithm and prepare client-specific state."""
        self.algorithm.attach(self)
        for client in self.clients:
            self.algorithm.initialize_client(client)

    def add_evaluation_hook(self, hook: EvaluationHook) -> None:
        self.evaluation_hooks.append(hook)

    def run(self) -> None:
        """Execute the federated optimization loop.

                The loop coordinates the following phases for each round:

                1. Client selection based on the algorithm strategy.
                2. Local training via :meth:`FederatedAlgorithm.client_step`.
                3. Aggregation of client payloads into the global model state.
                4. Model broadcast and optional evaluation hooks.
                """
        for epoch in range(self.epochs):
            selected_clients = self.algorithm.select_clients(self.clients, epoch)
            logger.info(
                "[Round %d/%d] Starting with %d clients",
                epoch + 1,
                self.epochs,
                len(selected_clients),
            )
            self.algorithm.before_round(epoch, selected_clients)

            client_payloads = []
            for client in selected_clients:
                payload = self.algorithm.client_step(self.global_model, client, epoch)
                client_payloads.append(payload)

            aggregation_result = self.algorithm.aggregate(
                self.global_model,
                selected_clients,
                client_payloads,
                self.aggregator,
            )

            self.global_model.load_state_dict(aggregation_result.state_dict)
            for client in selected_clients:
                client.load_model_state(aggregation_result.state_dict)
                self.algorithm.after_client_sync(client)
            logger.info(
                "[Round %d/%d] Aggregation complete",
                epoch + 1,
                self.epochs,
            )

            self.algorithm.after_aggregation(
                epoch,
                selected_clients,
                aggregation_result,
            )

        self.algorithm.on_complete()


class FederatedAlgorithm:
    """Base class for algorithms run by :class:`FederatedRunner`."""

    def __init__(self) -> None:
        self.runner: Optional[FederatedRunner] = None

    def attach(self, runner: FederatedRunner) -> None:
        self.runner = runner

    def initialize_client(self, client: Client) -> None:
        """Called once per client when the runner is created."""

    def select_clients(self, clients: Sequence[Client], epoch: int) -> Sequence[Client]:
        return clients

    def before_round(self, epoch: int, clients: Sequence[Client]) -> None:
        """Called before any client has been trained in a round."""

    def client_step(
            self,
            global_model: torch.nn.Module,
            client: Client,
            epoch: int,
    ):
        client.train()
        return {"state_dict": client.model_state()}

    def aggregate(
            self,
            global_model: torch.nn.Module,
            clients: Sequence[Client],
            payloads,
            aggregator: Aggregator,
    ) -> AggregationResult:
        states = [payload["state_dict"] for payload in payloads]
        return aggregator.aggregate(states)

    def after_aggregation(
            self,
            epoch: int,
            clients: Sequence[Client],
            aggregation: AggregationResult,
    ) -> None:
        """Called after the global model has been updated."""

    def after_client_sync(self, client: Client) -> None:
        """Called after each client receives the aggregated model."""

    def on_complete(self) -> None:
        """Called once after the final epoch."""


__all__ = ["Client", "FederatedRunner", "FederatedAlgorithm"]
