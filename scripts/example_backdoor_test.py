# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import subprocess

for algorithm in ['apfl', 'fedavg', 'scaffold', 'fedprox', 'flprotector']:
    config = f"configs/d5/{algorithm}.yaml"
    print(f"ALGORITHM={algorithm}",
          f"ATTACK_TEST=backdoor")
    subprocess.run(
        ["python",
         "-m", "pfl_benchmark", "final-test",
         "--run-dir", f"outputs/d5/{algorithm}/ATTACKS/byz_backdoor",
         "--attack", "backdoor"])
