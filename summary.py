import csv
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

# root = "output/original/TanksAndTemple"
# root = "output/ablation/TanksAndTemple_residual"
# root = "output/ablation/GlossySynthetic_residual"
# root = "output/original/GlossySynthetic"
# root = "output/rotreal"
root = sys.argv[1]

DEFAULT_MESH_DICT = {
    "completeness": -1,
    "accuracy": -1,
    "chamfer-L1": -1,
    "f-score": -1,
    "f-score-15": -1,
    "f-score-20": -1,
}

case_names = sorted(glob(os.path.join(root, "*")))
case_names = [c for c in case_names if os.path.isdir(c)]
summarys = {}
summary = pd.DataFrame(columns=["Case", "Method", "PSNR", "SSIM", "LPIPS"] + list(DEFAULT_MESH_DICT.keys()))
for case in case_names:
    print("Processing", case)
    case_name = os.path.basename(case)
    method_names = sorted([m for m in os.listdir(case) if os.path.isdir(os.path.join(case, m))])

    for method in method_names:
        result_dict = {"Case": case_name, "Method": method}
        results_path = os.path.join(case, method, "results.json")
        best = None
        if os.path.exists(results_path):
            with open(results_path) as json_file:
                contents = json.load(json_file)
                # method = sorted(contents.keys(), key=lambda method : int(method.split('_')[-1]))[-1]
                best = sorted(contents.keys(), key=lambda method: float(contents[method]["PSNR"]))[-1]
                result_dict["PSNR"] = contents[best]["PSNR"]
                result_dict["SSIM"] = contents[best]["SSIM"]
                result_dict["LPIPS"] = contents[best]["LPIPS"]
        mesh_results_path = os.path.join(case, method, "results_mesh.json")
        if os.path.exists(mesh_results_path):
            with open(mesh_results_path) as json_file:
                best_metric_dict = json.load(json_file)[best]
                for key, value in DEFAULT_MESH_DICT.items():
                    result_dict[key] = best_metric_dict.get(key)
        summary = pd.concat([summary, pd.DataFrame([result_dict])], ignore_index=True)

    # summarys[case] = summary
summary.to_csv(os.path.join(root, "summary.csv"), index=False, float_format="%f")

# for case, summary in summarys.items():
#     print("Case:", case)
#     print(summary)
#     summary.to_csv(os.path.join(case, "summary.csv"), index=False)
#     print("Summary saved to", os.path.join(case, "summary.csv"))
#     print()
