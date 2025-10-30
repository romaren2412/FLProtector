# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""Helpers that bootstrap Digit-Five clients and shared global models."""

import logging
from typing import List, Tuple

import torch

from models.redes import DigitFiveNet
from src.pfl_benchmark.data.digitfive.clients import DigitFiveTraining

logger = logging.getLogger(__name__)

def _load_clients(c, global_net=None):
    """Instantiate Digit-Five clients and optionally load a shared model state."""
    clients = []
    for i in range(c.SIZE):
        logger.info("Loading data for client %s", i)
        c.RANK = i
        ind_train, ind_test, data_type = c.leer_indices_dict()
        client = DigitFiveTraining(c, ind_train, ind_test)
        client.data_type = data_type
        if global_net is not None:
            client.net.load_state_dict(global_net.state_dict())
        clients.append(client)
    return clients


def init_d5_with_global_net(c, device) -> Tuple[DigitFiveNet, List[DigitFiveTraining]]:
    """Create a shared global network and clone it into each client."""
    global_net = DigitFiveNet().to(device)
    clients = _load_clients(c, global_net)
    return global_net, clients


# Reuse the same function object for initializers that require a global net
init_d5_apfl = init_d5_with_global_net
init_d5_flprotector = init_d5_with_global_net
init_d5_fedavg = init_d5_with_global_net
init_d5_fedprox = init_d5_with_global_net


def init_d5_without_global_net(c) -> List[DigitFiveTraining]:
    """Initialise Digit-Five clients without synchronising a global model."""
    return _load_clients(c)


# Alias for local init
init_d5_local = init_d5_without_global_net


def init_d5_scaffold(c, device):
    """Initialiser for SCAFFOLD which returns the control variates alongside clients."""
    global_net, clients = init_d5_with_global_net(c, device)
    c_server = {name: torch.zeros_like(param) for name, param in global_net.state_dict().items()}
    return global_net, clients, c_server


def init_d5_central(c):
    """Aggregate all client indices into a single central training object."""
    logger.info("Loading centralised Digit-Five dataset")
    ind_train, ind_test = [], []
    for rank in range(c.SIZE):
        c.RANK = rank
        train_i, test_i, _ = c.leer_indices_dict()
        ind_train.extend(train_i)
        ind_test.extend(test_i)

    ind_train, ind_test = set(ind_train), set(ind_test)
    server_model = DigitFiveTraining(c, ind_train, ind_test)
    server_model.data_type = c.DATA_TYPE if c.DATA_TYPE == c.EXTRA_DATA_TYPE else 'Mix'
    return server_model
