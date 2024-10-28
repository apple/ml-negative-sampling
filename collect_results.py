"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
import mord as m
import seaborn as sns
import json
import numpy as np
import pandas as pd
from make_scatter_plots import get_balance


if __name__ == "__main__":

    results_path = "./all_res/"
    methods = [
        "uniform",
        "adaptive",
        "adaptive-only",
        "mixed",
        "popularity",
        "in-batch",
        "inverse",
        "balanced",
    ]
    methods_name_map = {
        "uniform": "RNS",
        "adaptive": "AMNS",
        "adaptive-only": "ANS",
        "mixed": "MNS",
        "popularity": "PNS",
        "in-batch": "BNS",
        "inverse": "IPNS",
        "balanced": "PCNS",
    }
    cohorts = ["head", "mid", "tail"]
    datasets = ["ml10m", "Retail", "Beauty"]
    dataset_name_map = {
        "ml10m": "Movielens",
        "Beauty": "Beauty",
        "Retail": "RetailRocket",
    }
    num_exp = 20

    
    all_res = []
    for dataset in datasets:
        for method in methods:
            for exp in range(num_exp):
                exp_path = (
                    results_path
                    + "method_"
                    + method
                    + "_dataset_"
                    + dataset
                    + "_exp_"
                    + str(exp)
                    + ".json"
                )
                try:
                    with open(exp_path, "r") as f:
                        res = json.load(f)
                        res_ser = [
                            dataset_name_map[dataset],
                            methods_name_map[method],
                            exp,
                        ]
                        res_ser += [
                            res["total_hr"],
                            res["total_ndcg"],
                            res["best_epoch"],
                        ]
                        for cohort in cohorts:
                            res_ser.append(res["cohort_metrics"][cohort]["HT"])
                            res_ser.append(res["cohort_metrics"][cohort]["NDCG"])
                        all_res.append(res_ser)
                except Exception as e:
                    print(e)

    columns = ["dataset", "method", "exp", "total_hr", "total_ndcg", "best_epoch"]
    for cohort in cohorts:
        columns.append(cohort + "_hr")
        columns.append(cohort + "_ndcg")

    df = pd.DataFrame(all_res, columns=columns)
    df["balance"] = df.apply(get_balance, axis=1)
    df.to_csv("data/all_results.csv")