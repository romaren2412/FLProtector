# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Mapping, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.d5_dataset import DigitFiveDatasheet
from models.redes import DigitFiveNet
from utils.training_utils import APFLTrainer, FLProtectorTrainer, FedAvgTrainer, FedProxTrainer, ScaffoldTrainer


def _build_trainer(config, training):
    if config.TIPO_EXEC == 'FedProx':
        return FedProxTrainer(config, training.net, training.trainloader, config.DEVICE, training.optimizer, training.criterion, training.is_byz)
    if config.TIPO_EXEC == 'Scaffold':
        return ScaffoldTrainer(config, training.net, training.trainloader, config.DEVICE, training.optimizer, training.criterion, training.is_byz)
    if config.TIPO_EXEC == 'APFL':
        return APFLTrainer(config, training.net, training.trainloader, config.DEVICE, training.optimizer, training.criterion, training.delta_optimizer, training.is_byz)
    if config.TIPO_EXEC == 'FLProtector':
        return FLProtectorTrainer(config, training.net, training.trainloader, config.DEVICE, training.optimizer, training.criterion, training.delta_optimizer, training.is_byz)
    return FedAvgTrainer(config, training.net, training.trainloader, config.DEVICE, training.optimizer, training.criterion, training.delta_optimizer, training.is_byz)


class DigitFiveTraining:
    """Container for DigitFive model, data loaders and optimization state."""

    def __init__(self, config, train_indices, test_indices=None) -> None:
        self.c = config
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.client_id = config.RANK

        self.test_indices = test_indices
        self.train_indices = train_indices

        self.attack_type = getattr(config, "BYZANTINE_ATTACK", None)
        self.is_byz = self.client_id < getattr(config, "NBYZ", 0)

        self.net = DigitFiveNet().to(self.c.DEVICE)

        if self.c.OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.c.LR)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.c.LR)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.create_train()
        if test_indices is not None:
            self.create_test()

        self.delta = copy.deepcopy(self.net)
        self.delta.load_state_dict(torch.load(config.PATH_TOMODELS_ZERO))
        self.delta_optimizer = optim.Adam(self.net.parameters(), lr=self.c.LR)
        self.beta_pond = 1

        self.trainer = _build_trainer(config, self)

    # Data ---------------------------------------------------------------------
    def create_train(self) -> None:
        self.df_train = DigitFiveDatasheet(self.train_indices, self.c, self.transform, train=True)
        self.trainloader = DataLoader(
            self.df_train,
            batch_size=self.c.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

    def create_test(self) -> None:
        self.df_test = DigitFiveDatasheet(self.test_indices, self.c, self.transform, train=False)
        self.testloader = DataLoader(
            self.df_test,
            batch_size=self.c.BATCH_TEST_SIZE,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
        )

    # Evaluation ---------------------------------------------------------------
    def testear(self) -> float:
        rede = self.trainer.model
        rede.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.testloader:
                images, labels = batch[0].to(self.c.DEVICE), batch[1].to(self.c.DEVICE)
                outputs = rede(images.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    # Personalisation utilities -----------------------------------------------
    @property
    def personalization_buffer(self) -> OrderedDict:
        return OrderedDict(
            (name, tensor.clone().detach()) for name, tensor in self.delta.state_dict().items()
        )

    @personalization_buffer.setter
    def personalization_buffer(self, state: Mapping[str, torch.Tensor]) -> None:
        self.delta.load_state_dict(state)

    @property
    def delta_buffer(self) -> Optional[OrderedDict]:
        delta_tilde = getattr(self, 'delta_tilde', None)
        if delta_tilde is None:
            return None
        return OrderedDict(
            (name, tensor.clone().detach()) for name, tensor in delta_tilde.state_dict().items()
        )

    @delta_buffer.setter
    def delta_buffer(self, state: Mapping[str, torch.Tensor]) -> None:
        delta_tilde = getattr(self, 'delta_tilde', None)
        if delta_tilde is None:
            raise AttributeError('delta_buffer is not available for this training instance')
        delta_tilde.load_state_dict(state)

    def add_personalization(self) -> None:
        for name in self.net.state_dict().keys():
            self.net.state_dict()[name] += self.delta.state_dict()[name]

    def remove_personalization(self, fed_state: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in self.delta.state_dict().items():
            tensor.zero_()
            tensor += self.net.state_dict()[name] - fed_state[name]
        self.net.load_state_dict(fed_state)


__all__ = ['DigitFiveTraining']
