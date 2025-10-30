# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

"""Dataset helpers for the Digit-Five benchmark."""

import logging
from typing import List, Sequence

import numpy as np
import scipy.io
import torch

logger = logging.getLogger(__name__)

# -- DIGIT FIVE -- #
main_data_folder_d5 = "data/d5/"
dic_d5 = {
    'mnist': np.arange(0, 60).tolist(),
    'mnistm': np.arange(60, 119).tolist(),
    'svhn': np.arange(119, 192).tolist(),
    'syn': np.arange(192, 204).tolist(),
    'usps': np.arange(204, 213).tolist()
}

dict_act_red = {
    'mnist': {
        0: [5, 7, 12, 29, 36, 47, 51],
        1: [2, 14, 19, 22, 32, 41, 55],
        2: [1, 8, 17, 27, 40, 48, 50],
        3: [3, 10, 24, 28, 35, 43, 46],
        4: [6, 9, 21, 26, 37, 44, 58],
        'test': [13, 45]
    },
    'mnistm': {
        0: [61, 65, 66, 70, 79, 80, 90],
        1: [62, 68, 71, 76, 81, 84, 97],
        2: [67, 73, 77, 82, 85, 91, 100],
        3: [63, 64, 74, 75, 83, 87, 95],
        4: [69, 72, 78, 86, 89, 92, 102],
        'test': [88, 93]

    },
    'svhn': {
        0: [119, 123, 124, 129, 139, 141, 152],
        1: [120, 127, 131, 135, 143, 146, 153],
        2: [121, 125, 130, 136, 148, 150, 158],
        3: [122, 126, 134, 137, 140, 151, 157],
        4: [128, 132, 133, 138, 142, 149, 159],
        'test': [144, 147]
    },
    'syn': {
        # Important: DIGIT-Five only defines a single synthetic (SYN) client per split.
        # Allocating more than one client from this pool will duplicate examples
        # and break the benchmark protocol.
        0: [193, 195, 196, 198, 199, 201, 202],
        1: [193, 195, 196, 198, 199, 201, 202],
        2: [193, 195, 196, 198, 199, 201, 202],
        3: [193, 195, 196, 198, 199, 201, 202],
        4: [193, 195, 196, 198, 199, 201, 202],
        'test': [192, 194]
    },
    'usps': {
        # Important: the USPS domain mirrors the SYN restriction; create at most
        # one client per split to avoid sampling overlapping data points.
        0: [205, 207, 208, 210, 211, 209, 212],
        1: [205, 207, 208, 210, 211, 209, 212],
        2: [205, 207, 208, 210, 211, 209, 212],
        3: [205, 207, 208, 210, 211, 209, 212],
        4: [205, 207, 208, 210, 211, 209, 212],
        'test': [204, 206]
    }
}


def _load_mat_file(path: str):
    """Load a MATLAB archive and return its label and image arrays."""

    dict_mat = scipy.io.loadmat(path)
    labels = dict_mat["labels"][0].astype(np.int64)
    images = dict_mat["images"]
    return labels, images


class DigitFiveDatasheet:
    """Lazy loader around the Digit-Five MATLAB archives.
        The dataset spans five digit recognition domains: MNIST, MNIST-M, SVHN, SYN,
        and USPS.  Individual ``data_*.mat`` files include the RGB images as well as
        integer labels.  ``aprendedores`` pass an instance of this class to PyTorch
        ``DataLoader`` objects, which is why the ``__getitem__`` method returns
        tuples.
        Parameters
        ----------
        num:
            Sequence with the indices of the ``data_*.mat`` files that should be
            loaded.
        c:
            Configuration object.  Only the ``RANK`` and ``TIPO_EXEC`` attributes are
            accessed here.
        transform:
            Optional callable applied to each raw image.
        train:
            Flag indicating whether the dataset will serve a training or evaluation
            split.  Used purely for logging purposes.
    """

    def __init__(self, num: Sequence[int], c, transform=None, train: bool = True) -> None:
        self.c = c
        self.id = {"mt": 0, "mm": 0, "sv": 0, "syn": 0, "us": 0}
        self.id_list = ["mt", "mm", "sv", "syn", "us"]
        self.last_data = np.array([59, 118, 191, 203, 213])
        self.images_root = List[np.ndarray]
        self.labels_root = List[np.ndarray]
        path = main_data_folder_d5
        for idx, i in enumerate(num):
            mat_path = f"{path}data_{i}.mat"
            try:
                labels, images = _load_mat_file(mat_path)
            except (FileNotFoundError, OSError):
                labels, images = _load_mat_file(f"../{mat_path}")
            data_base = np.argwhere(i <= self.last_data).flatten()[0]
            if idx == 0:
                self.images_root = images
                self.labels_root = labels
                self.data_base = [data_base] * len(self.labels_root)
            else:
                self.images_root = np.concatenate((self.images_root, images))
                self.labels_root = np.concatenate((self.labels_root, labels))
                self.data_base = np.concatenate((self.data_base, [data_base] * len(labels)))

            self.id[self.id_list[data_base]] += len(labels)
        self.transform = transform
        if train:
            logger.info("[Client %s - train] indices=%s, domain_counts=%s", c.RANK, num, self.id)
        else:
            logger.info("[Client %s - eval] indices=%s, domain_counts=%s", c.RANK, num, self.id)

        self._rng = self._build_rng()
        if train and self._is_byzantine_client():
            self._apply_byzantine_attack()

    def __len__(self) -> int:
        return len(self.labels_root)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.images_root[idx]
        y = self.labels_root[idx]
        db = self.data_base[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)

        img = img.float()

        sample = (img, y, db) if getattr(self.c, "TIPO_EXEC", None) == 'FLProtector' else (img, y)

        return sample

# -- Byzantine helpers -------------------------------------------------
    def _build_rng(self) -> np.random.Generator:
        seed = getattr(getattr(self.c, "runtime", None), "seed", None)
        rank = getattr(self.c, "RANK", 0)
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(seed + rank)

    def _is_byzantine_client(self) -> bool:
        return getattr(self.c, "RANK", 0) < getattr(self.c, "NBYZ", 0)

    def _apply_byzantine_attack(self) -> None:
        attack = getattr(self.c, "BYZANTINE_ATTACK", None)
        if attack is None or attack == 'none':
            return

        attack = attack.lower()
        logger.info(
            "[Client %s - train] applying byzantine attack '%s'", self.c.RANK, attack
        )
        if attack == "backdoor":
            self._inject_backdoor_trigger()
        elif attack == "label_flip":
            self._flip_labels()

    def _inject_backdoor_trigger(self, n_channels=1) -> None:
        """
        Apply the 2x2 pattern in the bottom-right corner to the samples i = 0,2,4,... of the batch
        and force their labels to `BYZANTINE_TARGET_LABEL`.
        """
        imgs = self.images_root
        lbls = self.labels_root

        if imgs is None or lbls is None:
            return

        imgs = torch.from_numpy(np.asarray(imgs)).float()
        lbls = torch.from_numpy(np.asarray(lbls)).long()

        B, H, W, C = imgs.shape

        n_ch = min(int(n_channels), C)

        idxs = torch.arange(0, B, 2, dtype=torch.long)
        if idxs.numel() == 0:
            return

        r1, r2 = max(H - 2, 0), H
        c1, c2 = max(W - 2, 0), W

        imgs[idxs, r1:r2, c1:c2, :n_ch] = 1.0

        target_label = int(getattr(self.c, "BYZANTINE_TARGET_LABEL", 0) or 0)
        lbls[idxs] = target_label

        self.images_root = imgs.numpy()
        self.labels_root = lbls.numpy()

    def _flip_labels(self) -> None:
        labels = self.labels_root.copy()
        if labels.size == 0:
            return

        flipped = labels.copy()
        for i, y in enumerate(labels):
            new_label = self._rng.integers(0, 10)
            while new_label == y:
                new_label = self._rng.integers(0, 10)
            flipped[i] = new_label

        self.labels_root = flipped
