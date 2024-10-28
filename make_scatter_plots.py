"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.25)
line_colour = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "SF Pro Text"

fig_spec = {
    "dpi": 300,
    "bbox_inches": "tight",
}

def is_pareto_efficient(costs):
    """
    Find the Pareto efficient points from an array of costs.
    Parameters:
    - costs: numpy array where each row represents the cost vector of an object.
    Returns:
    - A boolean mask indicating whether each point is Pareto efficient or not.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def get_balance(row):
    return 1.0 - gini(np.array([row["head_ndcg"], row["mid_ndcg"], row["tail_ndcg"]]))



res_df = pd.read_csv("data/all_results.csv")
res_df['dataset'] = res_df['dataset'].apply(lambda x: 'ML10M' if x == 'Movielens' else x)
res_df["balance"] = res_df.apply(get_balance, axis=1)
grouped = res_df.groupby(["dataset", "method"]).agg(
    {"total_ndcg": ["mean", "std"], "balance": ["mean", "std"]}
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
index = 0
for dataset, group1 in grouped.groupby(level=0):
    methods = []
    acc = []
    bal = []
    acc_std = []
    bal_std = []
    # plt.figure()
    for method, group2 in group1.groupby(level=1):
        methods.append(method)
        acc.append(group2.loc[(dataset, method), ("total_ndcg", "mean")])
        bal.append(group2.loc[(dataset, method), ("balance", "mean")])
        acc_std.append(group2.loc[(dataset, method), ("total_ndcg", "std")])
        bal_std.append(group2.loc[(dataset, method), ("balance", "std")])

    pareto = is_pareto_efficient(-np.column_stack((np.array(acc), np.array(bal))))

    for i, label in enumerate(methods):
        axes[index].errorbar(
            bal[i],
            acc[i],
            xerr=bal_std[i],
            yerr=acc_std[i],
            marker="^" if pareto[i] else "o",
            color=line_colour[0] if pareto[i] else line_colour[1],
        )
        axes[index].text(bal[i], acc[i], label, fontsize=11, ha="right", va="bottom")

    if dataset == "Beauty":
        # Adding labels and title
        axes[index].set_xlabel("Balance")
        axes[index].set_ylabel("NDCG@10")
    else:
        axes[index].set_xlabel(" ")
        axes[index].set_ylabel(" ")

    axes[index].set_title(dataset)
    index+=1 
    # plt.xlim(0.0,1.0)
    # plt.ylim(0.0,0.17)
    # plt.grid(True)

    plt.savefig("data/figure.png", format="png", **fig_spec)