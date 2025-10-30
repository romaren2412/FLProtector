# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""Collection of federated learning algorithms supported by the benchmark."""

from .apfl import APFLAlgorithm
from .fedavg import FedAvgAlgorithm
from .fedprox import FedProxAlgorithm
from .flprotector import FLProtectorAlgorithm
from .scaffold import ScaffoldAlgorithm

__all__ = [
    "APFLAlgorithm",
    "FedAvgAlgorithm",
    "FedProxAlgorithm",
    "FLProtectorAlgorithm",
    "ScaffoldAlgorithm",
]
