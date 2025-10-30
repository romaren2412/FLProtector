# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import subprocess

for dt in ['mnist']:
    for algorithm in ['apfl', 'fedavg', 'scaffold', 'flprotector', 'fedprox']:
        for attack in ['mean', 'backdoor']:
            config = f"configs/d5/{algorithm}.yaml"
            print(f"ALGORITHM={algorithm}",
                  f"ATTACK_TYPE={attack}",
                  f"DATA_TYPE={dt}")
            subprocess.run(
                ["python",
                 "-m", "pfl_benchmark", "train",
                 "--config", config,
                 "--override", f"runtime.run_name=ATTACKS",
                 "--override", f"dataset.attack_type={attack}",
                 "--override", f"dataset.data_type={dt}",
                 "--override", f"dataset.extra_data_type={dt}"])
