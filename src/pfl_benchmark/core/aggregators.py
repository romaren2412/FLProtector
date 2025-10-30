# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Mapping, MutableMapping, Optional, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]
MutableStateDict = MutableMapping[str, torch.Tensor]


@dataclass
class AggregationResult:
    """Container returned by aggregators."""

    state_dict: OrderedDict
    weights: Optional[List[float]] = None


class Aggregator:
    """Base class for aggregation strategies."""

    def aggregate(
        self,
        client_states: Sequence[StateDict],
        weights: Optional[Sequence[float]] = None,
    ) -> AggregationResult:
        raise NotImplementedError


class WeightedAveragingAggregator(Aggregator):
    """Classic FedAvg-style aggregation with optional trust weights."""

    def aggregate(
        self,
        client_states: Sequence[StateDict],
        weights: Optional[Sequence[float]] = None,
    ) -> AggregationResult:
        if not client_states:
            raise ValueError("client_states must contain at least one model state")

        num_clients = len(client_states)
        if weights is None:
            weights = [1.0 / num_clients] * num_clients
        else:
            if len(weights) != num_clients:
                raise ValueError("weights length must match number of client states")
            weight_sum = float(sum(weights))
            if weight_sum == 0:
                raise ValueError("weights must not sum to zero")
            weights = [float(w) / weight_sum for w in weights]

        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        first_state = client_states[0]
        for key in first_state.keys():
            aggregated[key] = torch.zeros_like(first_state[key])

        for weight, client_state in zip(weights, client_states):
            for key, value in client_state.items():
                aggregated[key] += value * weight

        for key, tensor in aggregated.items():
            aggregated[key] = tensor.clone().detach()

        return AggregationResult(state_dict=aggregated, weights=list(weights))


def clone_state_dict(state_dict: StateDict) -> OrderedDict:
    """Create a detached copy of a state dict."""

    cloned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        cloned[key] = value.clone().detach()
    return cloned
