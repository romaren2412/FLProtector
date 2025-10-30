# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import torch

from utils.file_utils import save_clients
from utils.evaluate_utils import testear_precisions_locais
from utils.init_utils import init_d5_local
from utils.seed_utils import set_seed


def localAlgorithm(c):
    set_seed(42)
    # Decide el dispositivo de ejecución
    device = torch.device('cuda', c.GPU) if c.GPU != -1 else torch.device('cpu')

    # EJECUCIÓN
    with device:
        aprendedores = init_d5_local(c)
        precision_array = []

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCHS):
            print(f"[INFO] Epoca {e}")

            # CADA CLIENTE
            for i, ap in enumerate(aprendedores):
                ap.trainer.adestrar()

            if (e + 1) % c.CHECK_PREC == 0 or e == c.EPOCHS - 1:
                testear_precisions_locais(aprendedores, e, precision_array, c.PATH)

        save_clients(c.PATH, aprendedores)
    print("FIN DO ADESTRAMENTO")
    return True
