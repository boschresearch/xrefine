# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Union


class SimpleLogger:
    def __init__(self, file_path: Union[str, Path], decimals: int = 6):
        self.file = None
        self.metrics = None
        self.decimals = decimals
        if not file_path:
            raise ValueError("Please provide a valid file path!")
        if Path(file_path).exists():
            print(f"Warning: Log file '{file_path}' already exists, results will be appended.")
            self.file = open(file_path, "a")
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    self.metrics = first_line.split("\t")
        else:
            self.file = open(file_path, "w")
            print("Created logger with file path: ", file_path)

    def set_metrics(self, metrics: list[str]):
        if self.metrics is None:
            self.metrics = metrics
            for metric in self.metrics:
                self.file.write(str(metric) + "\t")
            self.file.write("\n")
            self.file.flush()
        else:
            if len(metrics) != len(self.metrics):
                raise ValueError("Number of metrics does not match the existing metrics in the log file!")

    def add_results(self, values: list[float]):
        if len(values) != len(self.metrics):
            raise ValueError("Please provide one value per metric!")
        else:
            for value in values:
                self.file.write("{:.{}f}".format(float(value), self.decimals))
                self.file.write("\t")
            self.file.write("\n")
            self.file.flush()
