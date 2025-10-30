# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

import subprocess

for col in ['d5', '5b']:
    for ablation in ['baseline']:
        config = f"configs/{col}/flprotector.yaml"
        print(f"COLLECTION={col}",
              f"ABALATION_MODE={ablation}")
        subprocess.run(
            ["python",
             "-m", "pfl_benchmark", "final-test",
             "--run-dir", "outputs/d5/FLProtector/baseline",
             "--ablation-mode", ablation])
