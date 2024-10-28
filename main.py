"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
from util_eval import *

tf.compat.v1.disable_eager_execution()
import random


def str2bool(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parsed_args(parser):
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_epochs", default=201, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--loss_type", default="sigmoid", type=str)
    parser.add_argument("--batch_oversampling_factor", default=1.5, type=float)
    parser.add_argument('--run_from_config', action='store_true', help="Use hyperparameters from config")

    parser.add_argument("--exp_id", default=0, type=int)

    # Negative sampling parameters
    parser.add_argument(
        "--neg_sampler_type",
        default="uniform",
        const="all",
        nargs="?",
        choices=[
            "uniform",
            "in-batch",
            "mixed",
            "adaptive",
            "balanced",
            "popularity",
            "inverse",
            "adaptive-only",
        ],
        help="Select type of negative sampling",
    )
    parser.add_argument(
        "--num_neg_samples", default=200, type=int, help="Number of negative samples"
    )
    parser.add_argument(
        "--num_mixed_samples", default=50, type=int, help="Number of mixed negative samples"
    )
    parser.add_argument(
        "--num_adaptive_samples",
        default=50,
        type=int,
        help="Number of adaptive negative samples",
    )

    # Validation / testing parameters
    parser.add_argument(
        "--num_eval_neg",
        default=100,
        type=int,
        help="Number of items in the eval set for each user",
    )
    parser.add_argument(
        "--top_k",
        default=10,
        type=int,
        help="Top elements to rate for list-wise metrics",
    )

    args = parser.parse_args()

    if args.neg_sampler_type == "adaptive-only":
        args.neg_sampler_type = "adaptive"
        args.num_mixed_samples = 0

    if args.num_neg_samples is None:
        args.num_neg_samples = args.maxlen

    if args.num_eval_neg < args.top_k:
        raise ValueError("`num_eval_neg` should be greater than `top_k`")
    if args.num_neg_samples + args.num_mixed_samples < args.num_adaptive_samples:
        raise ValueError(
            "`num_adaptive_samples` should be lesser than than `num_neg_samples` + `num_mixed_samples`"
        )

    if not args.run_from_config:

        # Read hyperparameters from file containing HP tuning results
        import json
        with open('./all_exp_hp.json', 'r') as hp_file:
            hp_dict = json.load(hp_file)
        hyperparameters = hp_dict[args.neg_sampler_type][args.dataset]

        print('Loaded hyperparameters', hyperparameters)

        args.maxlen = int(hyperparameters["maxlen"])
        args.num_neg_samples = int(hyperparameters["num_neg_samples"])
        args.num_mixed_samples = int(hyperparameters["num_mixed_samples"])
        args.num_adaptive_samples = int(hyperparameters["num_adaptive_samples"])
        args.batch_size = int(hyperparameters["batch_size"])
        args.num_epochs = int(hyperparameters["num_epochs"])
        args.batch_oversampling_factor = int(hyperparameters["batch_oversampling_factor"])
        args.dropout_rate = float(hyperparameters["dropout_rate"])
        args.lr = float(hyperparameters["lr"])

        if args.neg_sampler_type == 'adaptive-only':
            args.neg_sampler_type = 'adaptive'
        
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parsed_args(parser)
    assert args.batch_oversampling_factor >= 1.0
    print(f"Training jobs with arguments: \n {args}")

    # Partition dataset into training, validation, and testing
    dataset = data_partition(args.dataset)
    cohorts, freq_array = obtain_cohorts(dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    print(f"Total number of users = {usernum}")
    print(f"Total number of items = {itemnum}")
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)

    sampler = WarpSampler(
        user_train, usernum, itemnum, freq_array, cohorts, args, n_workers=10
    )
    model = Model(usernum, itemnum, args)
    sess.run(tf.compat.v1.initialize_all_variables())

    T = 0.0
    t0 = time.time()

    batch_buffer = []
    max_num_batches = int(args.batch_oversampling_factor * num_batch)

    best_valid_epoch_test_stats = {}
    best_valid_epoch_accuracy = 0.0

    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        print(f"epoch: {epoch}")

        for step in tqdm(
                range(num_batch), total=num_batch, ncols=70, leave=False, unit="b"
        ):

            if len(batch_buffer) >= max_num_batches:
                u, seq, pos, neg = random.choice(batch_buffer)
            else:
                u, seq, pos, neg = sampler.next_batch()
                batch_buffer.append((u, seq, pos, neg))

            loss, _ = sess.run(
                [model.loss, model.train_op],
                {
                    model.u: u,
                    model.input_seq: seq,
                    model.pos: pos,
                    model.neg: neg,
                    model.is_training: True,
                },
            )

        if epoch % 25 == 0:
            t1 = time.time() - t0
            T += t1
            print("Evaluating"),
            t_test = evaluate_cohorts(model, dataset, cohorts, freq_array, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print("/********/")
            print(args.dataset)
            print(args.neg_sampler_type)
            print("/********/")
            print(
                "epoch:%d, time: %f(s), valid (NDCG@10: %.7f, HR@10: %.7f), test (NDCG@10: %.7f, HR@10: %.7f), test_pop (NDCG@10: %.7f, HR@10: %.7f)"
                % (
                    epoch,
                    T,
                    t_valid[0],
                    t_valid[1],
                    t_test[2],
                    t_test[3],
                    t_test[4],
                    t_test[5],
                )
            )
            print("Popularity Correlation: %.4f" % (t_test[6]))
            print("COHORT METRICS:")
            print("Model:")
            print(t_test[0])
            print("Popularity:")
            print(t_test[1])

            if t_valid[1] > best_valid_epoch_accuracy:
                best_valid_epoch_accuracy = t_valid[1]
                best_valid_epoch_test_stats = {
                    "best_epoch": epoch,
                    "total_ndcg": t_test[2],
                    "total_hr": t_test[3],
                    "cohort_metrics": t_test[0],
                }
            t0 = time.time()

    best_valid_epoch_test_stats["time"] = time.time() - start_time

    sampler.close()
    print("Done")

    if args.neg_sampler_type == "adaptive" and args.num_mixed_samples == 0:
        sampler_type = "adaptive-only"
    else:
        sampler_type = args.neg_sampler_type

    path = (
        "all_res/method_"
        + sampler_type
        + "_dataset_"
        + args.dataset
        + "_exp_"
        + str(args.exp_id)
        + ".json",
    )    
    import json

    with open(path, "w") as f:
        json.dump(best_valid_epoch_test_stats, f, indent=4)        