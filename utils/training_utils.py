# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""Utility classes and helpers that orchestrate client-side training.
This module centralises the reusable training logic that is shared across
personalised and standard federated learning algorithms.  Trainers encapsulate
the optimisation loop for a single client and expose a consistent interface so
that higher level orchestration code can interact with them without worrying
about the underlying algorithmic details.
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from utils.file_utils import save_accuracies_delta

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base class that defines the interface for client-side trainers.
       Parameters
       ----------
       c:
           Experiment configuration object containing optimisation hyperparameters
           (such as ``FL_FREQ``) and metadata about the current client.  The exact
           type depends on the calle,  but it must expose the attributes used by the
           trainer.
       model:
           Neural network assigned to the client.  The trainer updates its weights
           in-place during :meth:`adestrar`.
       trainloader:
           Iterable that yields batches of training samples for the client.  The
           expected batch structure depends on the dataset (``inputs, labels`` for
           standard training and ``inputs, labels, metadata`` for FLProtector).
       device:
           PyTorch device where the computation should take place (e.g. ``"cuda"``
           or ``"cpu"``).
       optimizer:
           Optimiser used for the primary training loop.
       criterion:
           Loss function to optimise.
       delta_optimizer:
           Optional optimizer for the *delta* personalization stage used by
           algorithms such as APFL.
       is_byz: bool = False
            Flag indicating whether the client is Byzantine (malicious) or not.
       """

    def __init__(self, c, model, trainloader, device, optimizer, criterion, delta_optimizer=None, is_byz=False) -> None:
        self.c = c
        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.delta_optimizer = delta_optimizer
        self.is_byz = is_byz

    def adestrar(self):
        """Run the local training loop.
        Subclasses must implement the specific optimisation routine.
        """
        raise NotImplementedError("Método adestrar() debe ser implementado por cada algoritmo específico.")

    def calcular_diferencias(self, pre_params: Dict[str, torch.Tensor], rede: Optional[nn.Module] = None) -> Dict[
        str, torch.Tensor]:
        """Compute the parameter delta between the pre-training and post-training weights.
        Parameters
        ----------
        pre_params:
            Snapshot of the parameters **before** calling :meth:`adestrar`.
        rede:
            Optional model whose state should be compared against ``pre_params``.
            When ``None`` the trainer's ``model`` attribute is used.
        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping of parameter names to the difference ``current - pre_params``.
        """
        local_params = self.model.state_dict() if rede is None else rede.state_dict()
        local_update = {key: local_params[key] - pre_params[key] for key in local_params}
        return local_update


class APFLTrainer(BaseTrainer):
    """Trainer for the Adaptive Personalized Federated Learning (APFL) algorithm."""

    def adestrar(self, epochs: Optional[int] = None, rede: Optional[nn.Module] = None, delta: bool = False):
        """Run the APFL local optimization loop.
        Parameters
        ----------
        epochs:
            Number of local epochs to execute.  Defaults to ``c.FL_FREQ`` when
            ``None``.
        rede:
            Personalised copy of the model (``aprendedores`` manage this when
            delta personalisation is active).  If omitted the trainer uses its
            internal ``model`` reference.
        delta:
            When ``True`` the method leverages ``delta_optimizer`` and enables
            the personalised branch of the algorithm.
        Returns
        -------
        Dict[str, torch.Tensor]
            Parameter updates computed as ``post_params - pre_params``. (if not delta)
        """
        epochs = self.c.FL_FREQ if epochs is None else epochs
        optimizer = self.optimizer if not delta else self.delta_optimizer
        rede = self.model.to(self.device) if rede is None else rede

        global_snapshot = None
        if delta:
            global_snapshot = {name: param.detach().clone() for name, param in self.model.state_dict().items()}

        # --- Solo parámetros entrenables (sin buffers) ---
        pre_params = {k: v.detach().clone() for k, v in rede.state_dict().items()}

        rede.train()
        for _ in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        local_update = self.calcular_diferencias(pre_params, rede)
        inf_update = apply_update_attack(local_update, self.c.BYZANTINE_ATTACK, self.is_byz, self.c.BYZANTINE_SCALE)

        if delta:
            # Update only the personalised network while keeping the global model intact
            personalised_state = rede.state_dict()
            for name in pre_params:
                personalised_state[name] = pre_params[name] + inf_update[name]
            rede.load_state_dict(personalised_state)
            if global_snapshot is not None:
                self.model.load_state_dict(global_snapshot)
        else:
            full_state = self.model.state_dict()
            for name, _ in self.model.named_parameters():
                full_state[name] = pre_params[name] + inf_update[name]
            self.model.load_state_dict(full_state)
        return inf_update


class FLProtectorTrainer(BaseTrainer):
    """Trainer used by the FLProtector variant which also handles an auto-encoder."""

    def __init__(self, c, model, trainloader, device, optimizer, criterion, delta_optimizer=None, is_byz=False):
        super().__init__(c, model, trainloader, device, optimizer, criterion, delta_optimizer, is_byz)
        self.encoder = None
        self.optimizer_encoder = None
        self.criterion_encoder = None
        self.delta_optimizer = delta_optimizer
        self.mean_encoder_train = 0.0
        self.std_encoder_train = 0.0

    def adestrar(self, epochs: Optional[int] = None, delta: bool = False) -> Dict[str, torch.Tensor]:
        epochs = self.c.FL_FREQ if epochs is None else epochs
        optimizer = self.optimizer if not delta else self.delta_optimizer
        rede = self.model.to(self.device)

        pre_params = {key: value.clone() for key, value in self.model.state_dict().items()}

        rede.train()
        for ep in range(epochs):
            for inputs, labels, _ in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        local_update = self.calcular_diferencias(pre_params)
        inf_update = apply_update_attack(local_update, self.c.BYZANTINE_ATTACK, self.is_byz, self.c.BYZANTINE_SCALE)
        self.model.load_state_dict({k: pre_params[k] + inf_update[k] for k in pre_params})
        return inf_update

    def adestrar_encoder(self, epochs: Optional[int] = None) -> None:
        """Train the auxiliary encoder network owned by FLProtector clients."""
        epochs = self.c.FL_FREQ if epochs is None else epochs
        optimizer = self.optimizer_encoder
        rede = self.encoder.to(self.device)

        logger.info("Training FLProtector encoder for %s epochs", epochs)
        for _ in range(epochs):
            for inputs, _, _ in self.trainloader:
                inputs = transforms.Resize((32, 32), antialias=True)(inputs.to(self.device))
                optimizer.zero_grad()
                outputs = rede(inputs.float())

                loss = self.criterion_encoder(outputs, inputs.float())
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

        loss_list = []
        with torch.no_grad():
            for inputs, _, _ in self.trainloader:
                inputs = transforms.Resize((32, 32), antialias=True)(inputs.to(self.device))
                output = rede(inputs.float())
                loss = self.criterion_encoder(output, inputs.float())
                loss = torch.mean(loss, dim=tuple(range(1, len(loss.shape))))
                if len(loss_list) == 0:
                    loss_list = loss.cpu().numpy()
                else:
                    loss_list = np.concatenate((loss_list, loss.cpu().numpy()))

        self.mean_encoder_train = np.mean(loss_list)
        self.std_encoder_train = np.std(loss_list)


class FedAvgTrainer(BaseTrainer):
    """Classic FedAvg trainer that performs standard local SGD."""

    def adestrar(self, epochs: Optional[int] = None, delta: bool = False):
        epochs = self.c.FL_FREQ if epochs is None else epochs
        optimizer = self.optimizer if not delta else self.delta_optimizer
        rede = self.model.to(self.device)

        pre_params = {key: value.clone() for key, value in self.model.state_dict().items()}

        rede.train()
        for ep in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        local_update = self.calcular_diferencias(pre_params)
        inf_update = apply_update_attack(local_update, self.c.BYZANTINE_ATTACK, self.is_byz, self.c.BYZANTINE_SCALE)
        self.model.load_state_dict({k: pre_params[k] + inf_update[k] for k in pre_params})
        return inf_update


class FedProxTrainer(BaseTrainer):
    """Implementation of the FedProx local objective."""

    def adestrar(self, global_params: Dict[str, torch.Tensor], mu: float, epochs: Optional[int] = None, **kwargs):
        epochs = self.c.FL_FREQ if epochs is None else epochs
        rede = self.model.to(self.device)

        rede.train()
        for _ in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = self.criterion(outputs, labels)

                proximal_term = 0.0
                for w_local, (name, w_global) in zip(self.model.parameters(), global_params.items()):
                    proximal_term += ((w_local - w_global) ** 2).sum()
                loss += (mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()

        # Cálculo final de métricas
        local_update = self.calcular_diferencias(global_params)
        inf_update = apply_update_attack(local_update, self.c.BYZANTINE_ATTACK, self.is_byz, self.c.BYZANTINE_SCALE)

        total_diff = sum((v ** 2).sum().item() for v in inf_update.values())
        weight_difference = torch.sqrt(torch.tensor(total_diff)).item()
        final_prox_term = (mu / 2) * total_diff

        self.model.load_state_dict({k: global_params[k] + inf_update[k] for k in global_params})
        return final_prox_term, weight_difference


class ScaffoldTrainer(BaseTrainer):
    """SCAFFOLD trainer that tracks control variates for variance reduction."""

    def __init__(self, c, model, trainloader, device, optimizer, criterion, is_byz=False):
        super().__init__(c, model, trainloader, device, optimizer, criterion, is_byz)
        self.c_client = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def adestrar(self, c_server: Dict[str, torch.Tensor] = None, epochs: Optional[int] = None, **kwargs):
        epochs = self.c.FL_FREQ if epochs is None else epochs
        rede = self.model.to(self.device)

        # Copiar pesos antes de entrenar
        pre_params = {name: p.detach().clone() for name, p in rede.named_parameters()}
        old_c_client = {k: v.detach().clone() for k, v in self.c_client.items()}

        rede.train()
        total_steps = 0
        for _ in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()

                # grad ← grad + c_server - c_client
                for (name, param) in rede.named_parameters():
                    if param.grad is not None:
                        param.grad.add_(c_server[name] - self.c_client[name])

                self.optimizer.step()
                total_steps += 1

        # Calcular delta_y
        post_params = {name: p.detach().clone() for name, p in rede.named_parameters()}
        delta_y = {k: post_params[k] - pre_params[k] for k in pre_params}

        inf_update = apply_update_attack(delta_y, self.c.BYZANTINE_ATTACK, self.is_byz, self.c.BYZANTINE_SCALE)

        # reconstruir el modelo local como pre + update (teniendo en cuenta el posible ataque)
        new_state = rede.state_dict()
        for name, _ in rede.named_parameters():
            new_state[name] = pre_params[name] + inf_update[name]
        rede.load_state_dict(new_state)

        # Actualizar c_client siguiendo opción II del paper
        eta = self.c.LR
        scale = 1.0 / (eta * float(total_steps))
        for name in self.c_client:
            self.c_client[name].add_((pre_params[name] - post_params[name]) * scale)
            self.c_client[name].sub_(c_server[name])

        # Calcular delta_c
        delta_c = {k: self.c_client[k] - old_c_client[k] for k in self.c_client}

        return inf_update, delta_c


## -- PERSONALIZED NET TRAINING -- ##
def adestramento_delta(
        aprendedores,
        e: int,
        local_precisions,
        delta_precisions,
        check_freq: int,
        path: str,
        mode: str = "baseline",
) -> Optional[List[float]]:
    """Run the periodic personalisation stage for algorithms that support it.
    Parameters
    ----------
    aprendedores:
        Iterable of client wrappers.  Each object must expose ``trainer``,
        ``beta_pond`` and helper methods like ``add_personalization``.
    e:
        Zero-indexed global round.
    local_precisions:
        Mutable list that accumulates the evaluation results before
        personalisation.
    delta_precisions:
        Mutable list that accumulates the evaluation results after the delta
        personalisation step.
    check_freq:
        Frequency (in communication rounds) for running the personalised
        evaluation.
    path:
        Directory where the precision arrays should be persisted.
    mode:
        Ablation selector that controls which parts of the delta routine are
        executed. ``"baseline"`` and ``"no_lbfgs"`` run both global and
        personalised evaluations, ``"no_autoencoder"`` only executes the
        personalised branch, and ``"no_personalization"`` skips the
        personalisation updates altogether.

    Returns
    -------
    Optional[List[float]]
        The global accuracy list when ``mode`` is ``"no_personalization"``.
    """
    mode = (mode or "baseline").lower()
    evaluation_round = (e + 1) % check_freq == 0
    record_global = mode != "no_autoencoder"
    record_personal = mode != "no_personalization"

    if not evaluation_round:
        if record_personal:
            for ap in aprendedores:
                fed_weights = copy.deepcopy(ap.net.state_dict())
                ap.add_personalization()
                ap.trainer.adestrar(delta=True)
                ap.remove_personalization(fed_weights)
        return None

    lep1 = [e] if record_global else []
    lep2 = [e] if record_personal else []

    for i, ap in enumerate(aprendedores):
        fed_weights = copy.deepcopy(ap.trainer.model.state_dict())
        acc1 = None
        if record_global:
            acc1 = ap.testear()
            lep1.append(acc1)

        if record_personal:
            ap.add_personalization()
            ap.trainer.adestrar(delta=True)
            personalized_weights = copy.deepcopy(ap.trainer.model.state_dict())

            pond_weights = ponderar_beta(fed_weights, personalized_weights, ap.beta_pond)
            ap.trainer.model.load_state_dict(pond_weights)
            acc2 = ap.testear()
            lep2.append(acc2)

            ap.trainer.model.load_state_dict(personalized_weights)
            ap.remove_personalization(fed_weights)

            if mode in ["baseline", "no_lbfgs"] and acc1 is not None:
                logger.info(
                    "[Round %s] Client #%s (%s) accuracy before/after personalization: %.4f -> %.4f",
                    e, i, ap.data_type, acc1, acc2,
                )
            elif mode == "no_autoencoder":
                logger.info(
                    "[Round %s] Client #%s (%s) accuracy after personalization: %.4f",
                    e, i, ap.data_type, acc2,
                )
            elif mode == "no_personalization" and acc1 is not None:
                logger.info(
                    "[Round %s] Client #%s (%s) accuracy with global model: %.4f",
                    e, i, ap.data_type, acc1,
                )
    if lep1 is not None and len(lep1) > 1:
        local_precisions.append(lep1)
        save_accuracies_delta(path, local_precisions, aprendedores, name='acc_local')
    if lep2 is not None and len(lep2) > 1:
        delta_precisions.append(lep2)
        save_accuracies_delta(path, delta_precisions, aprendedores, name='acc_delta')

    if mode == "no_personalization" and lep1 is not None and len(lep1) > 1:
        return lep1
    return None


def ponderar_beta(fed_dict: Dict[str, torch.Tensor], personalized_dict: Dict[str, torch.Tensor], beta: float) -> Dict[
    str, torch.Tensor]:
    """Interpolate between the federated and personalised models."""
    return {key: fed_dict[key] * (1 - beta) + personalized_dict[key] * beta for key in fed_dict}


# ATTACKS
def apply_update_attack(local_update, attack_type='none', byz=False, scale=1):
    """
    Aplica un ataque al gradiente local (update) si el cliente es byzantino.

    Args:
        local_update (dict): Diccionario de tensores (update del cliente).
        attack_type (str): Tipo de ataque ('mean', 'backdoor', 'none').
        byz (bool): Si el cliente es malicioso.
        scale (float): Factor de escalado para el ataque (positivo).

    Returns:
        dict: Update posiblemente modificado.
    """
    if not byz or attack_type in ['none', 'label_flip']:
        return local_update
    elif attack_type == 'mean':
        updated = {k: -v.clone() * scale for k, v in local_update.items()}
    elif attack_type == 'backdoor':
        updated = {k: v.clone() * scale for k, v in local_update.items()}
    else:
        raise ValueError(f"Unsupported attack: {attack_type}")

    return updated
