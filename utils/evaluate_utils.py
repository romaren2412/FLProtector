# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""Evaluation helpers shared across training scripts."""

import logging
from typing import Dict, List
from utils.file_utils import save_accuracies


logger = logging.getLogger(__name__)


def evaluar_local_models(e, aprendedores, path, local_precisions) -> Dict[str, float]:
    """Evaluate each client's personalised model on its local test set."""

    dic: Dict[str, float] = {}
    lep = [e]
    for ap in aprendedores:
        prec = ap.testear()
        lep.append(prec)
        logger.info("[Client %s - %s] Local accuracy: %.4f", ap.client_id, ap.data_type, prec)
        dic[ap.data_type] = prec
    local_precisions.append(lep)
    save_accuracies(path, local_precisions, aprendedores)
    return dic


def seleccion_representante(aprendedores):
    """Select a single representative per domain for datasets with duplicated tests."""
    seleccion = []
    for i, ap in enumerate(aprendedores):
        if ap.data_type not in [a.data_type for a in seleccion] and not ap.is_byz:
            seleccion.append(ap)
            logger.info("Selected representative client #%s for domain %s", ap.client_id, ap.data_type)
    return seleccion


def testear_precision_server(server_model, e, precision_array, path) -> None:
    """Evaluate the global (federated) model on the aggregated test set."""
    acc = server_model.testear()
    logger.info("Global model accuracy: %.4f", acc)
    precision_array_ep = [e, acc]
    precision_array.append(precision_array_ep)
    save_accuracies(path, precision_array)


def testear_precisions_locais(aprendedores, e, precision_array, path) -> None:
    """Evaluate each client's model on its local test set and persist the metrics."""
    precision_array_ep = [e]

    for ap in aprendedores:
        acc = ap.testear()
        logger.info("[Client %s] Local accuracy: %.4f", ap.data_type, acc)
        precision_array_ep.append(acc)

    precision_array.append(precision_array_ep)
    save_accuracies(path, precision_array, aprendedores)
