"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import os
import time
import argparse
import tensorflow as tf
from tqdm import tqdm
from util import *
import numpy as np

tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
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


if __name__ == "__main__":

    osf = [1, 3, 10, 20, 30, 50]
    ada = [
        [2412, 9.6],
        [2442, 12.7],
        [2695, 13.8],
        [2753, 14.1],
        [3000, 14.5],
        [3300, 15],
    ]
    uni = [
        [2341, 11.8],
        [2365, 13.1],
        [2372, 13.55],
        [2375, 13.6],
        [2395, 13.8],
        [2420, 13.9],
    ]

    plt.figure()
    plt.scatter(uni[0][0], uni[0][1], color=line_colour[0], label="RNS")
    plt.scatter(ada[0][0], ada[0][1], color=line_colour[1], label="AMNS", marker="s")
    plt.text(
        uni[0][0],
        uni[0][1],
        osf[0],
        fontsize=9,
        ha="right",
        va="bottom"
    )
    plt.text(
        ada[0][0],
        ada[0][1],
        osf[0],
        fontsize=9,
        ha="right",
        va="bottom"
    )
    for i in range(1, len(osf)):
        plt.scatter(uni[i][0], uni[i][1], color=line_colour[0])
        plt.scatter(ada[i][0], ada[i][1], color=line_colour[1], marker="s")
        plt.text(uni[i][0], uni[i][1], osf[i], fontsize=9, ha="right", va="bottom")
        plt.text(ada[i][0], ada[i][1], osf[i], fontsize=9, ha="right", va="bottom")
    plt.legend()
    plt.xlabel("Total Runtime (s)")
    plt.ylabel("Accuracy")
    plt.savefig("osf.png", **fig_spec)
    plt.show
