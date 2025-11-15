# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import csv
import os

results_folder_path = "results"  # Default location for local user results

folders = [f for f in os.listdir(results_folder_path) if os.path.isdir(os.path.join(results_folder_path, f))]

results_list = [
    [
        "dataset",
        "extractor",
        "num_kps",
        "refinement_model",
        "auc5",
        "auc10",
        "auc20",
        "inlier_ratio",
        "mean",
        "median",
        "loss",
    ]
]
for folder in folders:
    folder_split = folder.split("_")
    if len(folder_split) > 4:
        raise ValueError(
            "Folder name should be in the format: [dataset]_[extractor]_[num_kps]_[refinement_model], but got: "
            f"{folder_split}"
        )
    dataset, extractor, num_kps, refinement_model = folder_split

    # Check if the folder contains a "averaged_results.txt" file
    averaged_results_path = os.path.join(results_folder_path, folder, "averaged_results.txt")
    if os.path.exists(averaged_results_path):
        with open(averaged_results_path, "r") as f:
            lines = f.readlines()

        # Extract the last line of the file
        last_line = lines[-1].strip()
        last_line_split = last_line.split("\t")

        # Do not put the RefinementTime result in the summary
        first_line = lines[0].strip()
        first_line_split = first_line.split("\t")
        for index in range(len(first_line_split)):
            if first_line_split[index] == "RefinementTime":
                del last_line_split[index]

        results_list.append([dataset, extractor, num_kps, refinement_model] + last_line_split)
    else:
        print("Results not found for folder: ", folder)

output_csv_path = os.path.join(results_folder_path, "summary.csv")
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(results_list)

print(f"Results saved to {output_csv_path}")
