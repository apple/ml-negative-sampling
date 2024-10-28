"""
Functions to partition train and test data
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""
from collections import defaultdict
import sys
import zipfile

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = {}
    user_train = {}
    user_valid = {}
    user_test = {}

    with zipfile.ZipFile('data/' + fname + '/full_compressed.zip' , 'r') as zipf:
        with zipf.open('full.txt') as file:
            content = file.read().decode('utf-8')

    for line in content.split('\n'):
        u, i, split = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        split = int(split)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if u in User and split in User[u]:
            User[u][split].append(i)
        else:
            User[u] = defaultdict(list)
            User[u][0] = []
            User[u][1] = []
            User[u][2] = []
            User[u][split].append(i)
    for user in User:
        # Add training sequence
        if 0 in User[user]:
            user_train[user] = User[user][0]
        else:
            user_train[user] = []
        # Add validation sequence
        if 1 in User[user]:
            user_valid[user] = User[user][1]
        else:
            user_valid[user] = []
        # Add testing sequence
        if 2 in User[user]:
            user_test[user] = User[user][2]
        else:
            user_test[user] = []
    return [user_train, user_valid, user_test, usernum, itemnum]

def obtain_cohorts(dataset):
    import numpy as np
    itemnum = dataset[4]
    freq_array = np.zeros(itemnum)
    for (u_train, u_valid, u_test) in zip(dataset[0].items(), dataset[1].items(), dataset[2].items()):
        freq_array[[i - 1 for i in u_train[1]]] += 1
        # freq_array[[i - 1 for i in u_valid[1]]] += 1
        # freq_array[[i - 1 for i in u_test[1]]] += 1

    total_downloads = freq_array.sum()

    cohorts = {"head":set(), "mid":set(), "tail":set()}
    cohorts_index = itemnum * ['']
    cumul_freq = 0
    for item, freq in zip(list(np.argsort(-freq_array)), list(np.sort(-freq_array))):
        cumul_freq += -freq
        if cumul_freq < total_downloads / 3:
            cohorts["head"].add(item+1)
            cohorts_index[item] = "head"
        elif cumul_freq < 2 *total_downloads / 3:
            cohorts["mid"].add(item+1)
            cohorts_index[item] = "mid"
        else:
            cohorts["tail"].add(item+1)
            cohorts_index[item] = "tail"

    print('Cohort Sizes: head={}, mid={}, tail={}'.format(str(len(cohorts["head"])), str(len(cohorts["mid"])), str(len(cohorts["tail"]))))

    cohorts["index"] = cohorts_index

    cohorts["head_list"] = list(cohorts["head"])
    cohorts["mid_list"] = list(cohorts["mid"])
    cohorts["tail_list"] = list(cohorts["tail"])

    return cohorts, freq_array
