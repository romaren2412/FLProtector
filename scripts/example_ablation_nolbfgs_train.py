# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import subprocess

for dt in ['mnist', 'svhn']:
    for attack in ['none', 'backdoor', 'mean', 'label_flip']:
        config = f"configs/d5/flprotector.yaml"
        print(f"ALGORITHM=flprotector",
              f"ATTACK_TYPE={attack}",
              f"DATA_TYPE={dt}",
              f"EXTRA_DATA_TYPE={dt}")
        subprocess.run(
            ["python",
             "-m", "pfl_benchmark", "train",
             "--config", config,
             "--override", f"runtime.run_name=ABLATION_NOLBFGS",
             "--override", f"dataset.attack_type={attack}",
             "--override", f"dataset.data_type={dt}",
             "--override", f"dataset.extra_data_type={dt}",
             "--override", "algorithm.ablation=no_lbfgs"])
