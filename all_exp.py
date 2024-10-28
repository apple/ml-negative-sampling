"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import os
import argparse
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import subprocess
import json


def str2bool(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"
    
   # python main.py --dataset=ml10m --train_dir=default --maxlen=200 --dropout_rate=0.2 --num_neg_samples 50 --batch_oversampling_factor 30 --neg_sampler_type uniform;

def parsed_args(parser):

    # Experiments
    parser.add_argument("--num_exp", default=10, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parsed_args(parser)

    if not os.path.exists("all_res"):
        os.mkdir("all_res")

    for method in [
        "uniform",
        "mixed",
        "adaptive",
        "adaptive-only",
        "in-batch",
        "popularity",
    ]:
        for dataset in [
            "ml10m",
            "Beauty",
            "Retail",
        ]:
            for experiment in range(args.num_exp):
                print("Method: ", method,", Dataset: ", dataset, ", Experiment: ", experiment)

                arg_list = [
                    "python",
                    "main.py",
                    "--run_from_config",
                    "--dataset=" + dataset,
                    "--neg_sampler_type=" + method, 
                    "--exp_id=" + str(experiment),
                ]

                subprocess.run(arg_list)
