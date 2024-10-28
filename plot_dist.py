"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

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

    # Plot
    dataset = data_partition("Retail")
    cohorts, freq_array = obtain_cohorts(dataset)
    #plt.figure(figsize=(100, 100), dpi=100)
    plt.figure()
    plt.semilogy(
        [x / len(freq_array) for x in range(1, len(freq_array) + 1)],
        -np.sort(-freq_array) / np.sum(freq_array),
        linewidth=3,
        color=line_colour[0],
    )
    plt.axvline(
        x=len(cohorts["head"]) / len(freq_array), color=line_colour[1], linestyle="--", linewidth=2
    )
    plt.axvline(
        x=len(cohorts["head"]) / len(freq_array) + len(cohorts["mid"]) / len(freq_array),
        color=line_colour[1],
        linestyle="--",
        linewidth=2,
    )
    # plt.text(len(cohorts["head"]) / 2, 1000, 'Head', fontsize=12, color='blue')
    # plt.text(len(cohorts["head"]) + len(cohorts["body"]) / 2, 1000, 'Body', fontsize=12, color='blue')
    # plt.text(len(cohorts["head"]) + len(cohorts["body"]) + len(cohorts["tail"]) / 2, 1000, 'Tail', fontsize=12, color='blue')
    plt.text(
        -0.05,
        0.000002,
        str(len(cohorts["head"])) + " \n head \n items",
        fontsize=12,
        color="black",
    )
    plt.text(
        0.15,
        0.000002,
        str(len(cohorts["mid"])) + " \n mid \n items",
        fontsize=12,
        color="black",
    )
    plt.text(
        0.5,
        0.000002,
        str(len(cohorts["tail"])) + " \n tail \n items",
        fontsize=12,
        color="black",
    )
    # plt.grid()
    plt.ylabel("log(item frequency)")
    plt.xlabel("log(number of items)")
    plt.savefig("data/cohorts.png", format="png", **fig_spec)
    plt.show()

    plt.figure()
    plt.title("Popularity distributions (normalized)")
    for dataset_idx, dataset_name in enumerate(
        ["ml10m", "Beauty", "Retail"]
    ):
        dataset = data_partition(dataset_name)
        cohorts, freq_array = obtain_cohorts(dataset)
        # plt.loglog(-np.sort(-freq_array) / np.sum(freq_array), label=dataset_name)
        # sns.kdeplot(freq_array, color='blue', fill=False, label=dataset_name)
        # plt.hist(np.log(freq_array/ np.sum(freq_array)), bins=60, histtype='step', linewidth=3, label=dataset_name)

        counts, bin_edges = np.histogram(
            np.log10(freq_array / np.sum(freq_array) + 0.000001), bins=15, density=False
        )
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.plot(
            bin_centers,
            counts,
            linestyle="-",
            linewidth=3,
            marker="",
            label=dataset_name,
            color=line_colour[dataset_idx],
        )
    plt.yscale("log")
    # plt.grid()
    plt.legend()
    plt.ylabel("log(Normalized Item Frequency)")
    plt.xlabel("Item Index / Num Items")
    plt.savefig("data/all_dist.png", format="png", **fig_spec)
    plt.show()

    # Create table with median user coverage per user/cohort
    from statistics import median

    res = {}
    for dataset_name in ["Retail", "ml10m", "Beauty"]:
        dataset = data_partition(dataset_name)
        cohorts, freq_array = obtain_cohorts(dataset)

        total_downloads = np.sum(freq_array)
        num_items = len(freq_array)
        num_users = dataset[3]

        data_res = {}
        for c in ["head", "mid", "tail"]:
            data_res[c] = {
                "Cohort Size": len(cohorts[c]),
                "Median User Coverage": 100
                * median([freq_array[x - 1] for x in cohorts[c]])
                / num_users,
            }

        res[dataset_name] = data_res

    print(res)