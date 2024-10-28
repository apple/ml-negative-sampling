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
    return 1.0 - gini(np.array([row["head_hr"], row["mid_hr"], row["tail_hr"]]))


if __name__ == "__main__":

    res_df = pd.read_csv("data/all_results.csv")
    res_df["balance"] = res_df.apply(lambda x: get_balance(x), axis=1)
    grouped = res_df.groupby(["dataset", "method"]).agg(
        {
            "head_hr": ["mean"],
            "mid_hr": ["mean"],
            "tail_hr": ["mean"],
            "total_hr": ["mean"],
            "balance": ["mean"],
        }
    )

    pd.set_option("display.precision", 3)

    print(100 * grouped)