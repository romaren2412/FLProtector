# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import torch

from utils.file_utils import save_clients
from utils.evaluate_utils import testear_precision_server
from utils.init_utils import init_d5_central
from utils.seed_utils import set_seed


def centralAlgorithm(c):
    """
    :param c: obxecto de configuración
    """
    set_seed(42)
    # Decide el dispositivo de ejecución
    device = torch.device('cuda', c.GPU) if c.GPU != -1 else torch.device('cpu')

    # EJECUCIÓN
    with device:
        # INICIALIZACIÓN DAS REDES E DATASETS
        server_model = init_d5_central(c)
        precision_array = []

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCHS):
            print(f"[INFO] Epoca {e}")
            server_model.trainer.adestrar()

            #############################################################################
            # PRECISIÓNS
            # CALCULAR A PRECISIÓN DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % c.CHECK_PREC == 0 or e == c.EPOCHS - 1:
                testear_precision_server(server_model, e, precision_array, c.PATH)

        save_clients(c.PATH, aprendedores=[server_model])
        print("FIN DO ADESTRAMENTO")
    return True
