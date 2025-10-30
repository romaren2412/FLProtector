# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import subprocess

for algorithm in ['apfl', 'fedavg', 'scaffold', 'flprotector', 'fedprox']:
    for dt in ['mnist', 'svhn']:
        for edt in ['mnist', 'mnistm', 'usps', 'svhn', 'syn']:
            if dt == edt:
                continue
            config = f"configs/d5/{algorithm}.yaml"
            print(f"ALGORITHM={algorithm}",
                  f"DATA_TYPE={dt}",
                  f"EXTRA_DATA_TYPE={edt}")
            subprocess.run(
                ["python",
                 "-m", "pfl_benchmark", "train",
                 "--config", config,
                 "--override", f"runtime.run_name=FULL",
                 "--override", f"dataset.data_type={dt}",
                 "--override", f"dataset.extra_data_type={edt}"])