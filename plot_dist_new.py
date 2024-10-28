"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

from cProfile import label
from util import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.25)
line_colour = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "SF Pro Text"

fig_spec = {
    "dpi": 300,
    "bbox_inches": "tight",
}

if __name__ == "__main__":

    plt.figure()
    for i, dataset_name in enumerate(["Retail", "Beauty", "ml10m"]):

        # Plot
        dataset = data_partition(dataset_name)
        cohorts, freq_array = obtain_cohorts(dataset)
        plt.semilogy(
            [x / len(freq_array) for x in range(1, len(freq_array) + 1)],
            -np.sort(-freq_array) / np.sum(freq_array),
            linewidth=3,
            label=dataset_name
            # color=line_colour[2*i],
        )
        # plt.axvline(
        #     x=len(cohorts["head"]), color=line_colour[2*i + 1], linestyle="--", linewidth=2
        # )
        # plt.axvline(
        #     x=len(cohorts["head"]) + len(cohorts["mid"]),
        #     color=line_colour[2*i + 1],
        #     linestyle="--",
        #     linewidth=2,
        # )
  
    # plt.grid()
    plt.ylabel("log(Normalized Item Frequency)")
    plt.xlabel("Item Index / Num Items")
    plt.legend()
    plt.savefig("data/all_cohorts" + ".png", format="png", **fig_spec)
    plt.show()