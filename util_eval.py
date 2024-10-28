"""
Overall and popularity-based evaluation methods

For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
import sys
import copy
import random
import numpy as np


def evaluate_cohorts(model, dataset, cohorts, freq_array, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    cohort_metrics = {"head":{"NDCG":0.0,"HT":0.0,"user_count": 0.0},
                      "mid":{"NDCG":0.0,"HT":0.0,"user_count": 0.0},
                      "tail":{"NDCG":0.0,"HT":0.0,"user_count": 0.0}}
    cohort_metrics_popularity_rank = {"head":{"NDCG":0.0,"HT":0.0,"user_count": 0.0},
                                      "mid":{"NDCG":0.0,"HT":0.0,"user_count": 0.0},
                                      "tail":{"NDCG":0.0,"HT":0.0,"user_count": 0.0}}
    cohorts_index = cohorts["index"]

    popularity_ranking = np.empty_like(freq_array)
    popularity_ranking[np.argsort(-freq_array)] = np.arange(1, len(freq_array) + 1)

    NDCG = 0.0
    HT = 0.0
    NDCG_POP = 0.0
    HT_POP = 0.0
    valid_user = 0.0
    pop_corr = 0.0

    # users = random.sample(range(1, usernum + 1), 10000)
    for u in range(1, usernum + 1):

        if (len(train[u]) < 1 and len(valid[u]) < 1) or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(valid[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx] = i
            idx -= 1

        rated = set(train[u])
        rated.add(0)
        for item in test[u]:
            item_idx = [item]
            for t in range(1, itemnum + 1):
                if t != item_idx[0]:
                    item_idx.append(t)

            # Test model
            predictions = -model.predict(sess, [u], [seq], item_idx)
            predictions = predictions[0]
            rank = predictions.argsort().argsort()[0]

            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

            # Get popularity benchmark
            rank_pop = popularity_ranking[item_idx[0] - 1]
            if rank_pop < 10:
                NDCG_POP += 1 / np.log2(rank_pop + 2)
                HT_POP += 1

            # Cohort-based metrics calculations
            cohort = cohorts_index[item_idx[0] - 1]
            cohort_metrics[cohort]['user_count'] += 1
            cohort_metrics_popularity_rank[cohort]['user_count'] += 1
            if rank < 10:
                cohort_metrics[cohort]['NDCG'] += 1 / np.log2(rank + 2)
                cohort_metrics[cohort]['HT'] += 1
            if rank_pop < 10:
                cohort_metrics_popularity_rank[cohort]['NDCG'] += 1 / np.log2(rank_pop + 2)
                cohort_metrics_popularity_rank[cohort]['HT'] += 1


            if valid_user % 100 == 0:
                sys.stdout.write ('.')
                sys.stdout.flush()

        # if valid_user > 9999:
        #     break


    for cohort in cohort_metrics.keys():
        cohort_metrics[cohort]['NDCG'] /= cohort_metrics[cohort]['user_count'] + 0.01
        cohort_metrics[cohort]['HT'] /= cohort_metrics[cohort]['user_count'] + 0.01
        cohort_metrics_popularity_rank[cohort]['NDCG'] /= cohort_metrics_popularity_rank[cohort]['user_count'] + 0.01
        cohort_metrics_popularity_rank[cohort]['HT'] /= cohort_metrics_popularity_rank[cohort]['user_count'] + 0.01

    return cohort_metrics, cohort_metrics_popularity_rank, NDCG / valid_user, HT / valid_user, \
                                                           NDCG_POP / valid_user, HT_POP / valid_user, 0.0 / valid_user

def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # Sample users, if greater than 10K
    # TODO: Calculate full metrics
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        # Exclude users that were not in the training set
        # Exclude users with no items
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # To evaluate the test set, we also need to include
        # the one item in the validation set
        seq[idx] = valid[u][0]
        idx -= 1
        # Add all items in the training to the input sequence
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        # Label to predict / score
        item_idx = [test[u][0]]
        # Sample other items to create a recall set of 101
        # 1 label + args.num_eval_neg sample w/ replacement negative items
        for _ in range(args.num_eval_neg):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1
        # TODO: Make this configurable
        if rank < args.top_k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            sys.stdout.write ('.')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    for u in valid:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        for valid_item in valid[u]:
            item_idx = [valid_item]
            for t in range(1, itemnum + 1):
                if t != item_idx[0]:
                    item_idx.append(t)

            predictions = -model.predict(sess, [u], [seq], item_idx)
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0]

            valid_user += 1

            if rank < args.top_k:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        if valid_user > 9999:
            break

    return NDCG / valid_user, HT / valid_user