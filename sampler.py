"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import numpy as np
from multiprocessing import Process, Queue
import sys
import random

from tensorflow.python.util import tf_stack


cohort_sampling_probabilities = {"head" :[0.8, 0.1, 0.1], "body": [0.1, 0.8, 0.1], "tail": [0.1, 0.1, 0.1]}

def popularity_sampler_array(item_pop_dist, num_samples, ts):
    p_s = item_pop_dist
    p_s[[i-1 for i in ts]] = 0.0
    p_s = p_s / p_s.sum()

    samples = np.random.choice(list(range(1,len(item_pop_dist) + 1)), size = num_samples, p= p_s)[0]
    return samples

def popularity_sampler(item_pop_dist, ts):

    item_index = np.random.choice(list(range(1,len(item_pop_dist) + 1)), size = 1, p= item_pop_dist)[0]
    # Ensure item was not in user's history
    while item_index in ts:
        item_index = np.random.choice(list(range(1,len(item_pop_dist) + 1)), size = 1, p= item_pop_dist)[0]
    return item_index


def balanced_sampler(positive_sample, cohorts, ts):

    # Cohort sampling distribution depends on positive items cohort
    cohort_high_level_pmf = cohort_sampling_probabilities[cohorts["index"][positive_sample - 1]]

    start = True
    negative_sample = -1
    # Ensure item was not in user's history
    while start or negative_sample in ts:
        start = False
        # Uniform sample to determine which cohort to sample from
        u = random.random()
        if u < cohort_high_level_pmf[0]:
            # Sample from head
            negative_sample = cohorts["head_list"][np.random.randint(0, len(cohorts["head"]))]
        elif u < cohort_high_level_pmf[0] + cohort_high_level_pmf[1]:
            # Sample from body
            negative_sample = cohorts["body_list"][np.random.randint(0, len(cohorts["body"]))]
        else:
            # Sample from tail
            negative_sample = cohorts["tail_list"][np.random.randint(0, len(cohorts["tail"]))]

    return negative_sample


def get_density_estimates(item_freq, h = 10):
    freq_sorted = np.sort(item_freq)
    item_index = np.argsort(item_freq) + 1


    n = len(freq_sorted)
    # Dict to store the result
    density_est_dict = {}

    # Sliding window approach
    left = 0  # Initialize the left pointer of the window
    right = 0
    for i in range(n):
        # Move the left pointer to satisfy the condition x[i] - h
        while left < i and freq_sorted[left] < freq_sorted[i] - h:
            left += 1

        # Move the right pointer to satisfy the condition x[i] + h
        while right < n - 1 and freq_sorted[right] < freq_sorted[i] + h:
            right += 1

        # Count points in the interval, excluding the point itself
        density_est_dict[item_index[i]] = right - left + 1

        # print(x[i], counts[i])

    density_est = [density_est_dict[i + 1] for i in range(n)]

    return density_est

def uniform_sampler_optimized(low, high, positive_set, num_samples):
    """ Sample integers with replacement within [high, low]

    :param low: Minimum item index
    :param high: Maximum item index
    :param positive_set: Set of positive indices
    :param num_samples: Number of samples to return
    :return: Int
    """
    # Select a random item, assuming index starting from 0
    item_indices = set(np.arange(low, high, 1, dtype=int)) - positive_set
    return list(np.random.choice(list(item_indices), num_samples))
def uniform_sampler(low, high, positive_set):
    """ Sample integers with replacement within [high, low]

    :param low: Minimum item index
    :param high: Maximum item index
    :param positive_set: Set of positive indices
    :return: Int
    """
    # Select a random item, assuming index starting from 0
    item_index = np.random.randint(low, high)
    # Ensure item was not in user's history
    while item_index in positive_set:
        item_index = np.random.randint(low, high)
    return item_index

def sample_function(user_train, usernum, itemnum, item_freq, cohorts, args, result_queue, SEED):

    item_freq_dist = item_freq / item_freq.sum()
    ones_array = np.ones_like(item_freq)
    inverse_item_freq_dist = np.divide(ones_array, item_freq, out=np.zeros_like(ones_array, dtype=float), where=item_freq!=0)
    inverse_item_freq_dist = inverse_item_freq_dist / inverse_item_freq_dist.sum()

    def sample():
        # Select a random user
        user = np.random.randint(1, usernum + 1)
        # If the user sequence is <1 item, then sample other users
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        # Collect the user sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        # Positive samples
        pos = np.zeros([args.maxlen], dtype=np.int32)
        # Negative samples
        neg = np.zeros([args.num_neg_samples], dtype=np.int32)
        # Label item is the last in sequence
        nxt = user_train[user][-1]
        # Position index for adding to the sequence
        idx = args.maxlen - 1

        # Create a set of all items in the user sequence
        positive_set = set(user_train[user])
        # Iterating from the second-last item in the user sequence
        for i in reversed(user_train[user][:-1]):
            # Item in each part of the sequence
            seq[idx] = i
            # Assign the next item as positive
            pos[idx] = nxt
            # Reduce all counters by 1
            nxt = i
            idx -= 1
            if idx == -1: break

        # Creating negative samples
        idx = args.num_neg_samples - 1
        item_freq_dist = item_freq / item_freq.sum()
        if args.neg_sampler_type != 'balanced':
            for i in range(args.num_neg_samples):
                if args.neg_sampler_type == 'uniform':
                    neg[idx -i] = uniform_sampler(1, itemnum + 1, positive_set)
                elif args.neg_sampler_type == 'popularity':
                    neg[idx - i] = popularity_sampler(item_freq_dist, positive_set)
                elif args.neg_sampler_type == 'inverse':
                    neg[idx -i] = popularity_sampler(inverse_item_freq_dist, positive_set)
                elif args.neg_sampler_type in [ "mixed", "in-batch", "adaptive"]:
                    # Negative samples will only arrive from batch
                    pass
                else:
                    raise Exception('Unknown sampler type: ' + args.neg_sampler_type)

        elif args.neg_sampler_type == 'balanced':
            neg = []
            for i in range(args.maxlen):
                anchor_item = pos[args.maxlen -1 -i]
                sub_neg = np.zeros([args.maxlen], dtype=np.int32)
                if anchor_item != 0:
                    for sub_i in range(args.num_neg_samples):
                        sub_neg[args.maxlen -1 -sub_i] = balanced_sampler(anchor_item, cohorts, positive_set)
                neg.append(sub_neg)
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    batch_counter = 0
    while True:
        one_batch = []
        batch_negatives = []
        for _ in range(args.batch_size):
            one_batch.append(sample())
            # Collecting all positives, so they can become in-batch negatives
            batch_negatives.extend(one_batch[-1][2])
        batch_negatives = [i for i in batch_negatives if i !=0 ]

        # Create in-batch negatives here, more effective due to code structure
        if args.neg_sampler_type in ["in-batch", "mixed", "adaptive"]:
            new_batch = []
            for i in range(args.batch_size):
                user, seq, pos, neg = one_batch[i]
                allowed_negatives = [i for i in batch_negatives if i not in pos]
                # Always get maxlen number of negatives
                neg = list(np.random.choice(allowed_negatives, args.num_neg_samples, replace=False))
                if args.neg_sampler_type in ["mixed", "adaptive"]:
                    # Add `n_random_negs` random uniform negatives
                    random_negs = [uniform_sampler(1, itemnum + 1, pos) for _ in range(args.num_mixed_samples)]
                    neg.extend(random_negs)
                new_batch.append((user, seq, pos, neg))
            one_batch = new_batch

        result_queue.put(zip(*one_batch))
        batch_counter += 1

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, item_frequencies, cohorts, args, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      item_frequencies,
                                                      cohorts,
                                                      args,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
