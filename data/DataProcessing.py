"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import gzip
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import zipfile

test_valid_split_ratio = 0.05


def process_dataset(dataset_prefix, output_data_dir):
    full_df = pd.read_csv('./raw/' + dataset_prefix + '_full.tsv', sep='\t', header=0)

    print(full_df.columns)

    df = full_df.sort_values(by='Time')
    total_rows = len(df)
    split_percent_count = int(np.ceil(total_rows * test_valid_split_ratio))
    df['split'] = [0] * total_rows
    df.iloc[-split_percent_count:, df.columns.get_loc('split')] = 2
    df.iloc[-2 * split_percent_count:-split_percent_count, df.columns.get_loc('split')] = 1
    df = df.sort_values(by=['SessionId', 'Time'], ascending=[True, True])

    user_map = {}
    item_map = {}

    string_seq = []
    for _, row in df.iterrows():
        if row['SessionId'] not in user_map:
            user_map[row['SessionId']] = {'id' : len(user_map) + 1, 'stats' : [0, 0, 0]}                
        if row['ItemId'] not in item_map:
            item_map[row['ItemId']] = len(item_map) + 1                
        user_map[row['SessionId']]['stats'][row['split']] += 1

        string_seq.append(str(user_map[row['SessionId']]['id']) + ' ' + str(item_map[row['ItemId']]) + ' ' + str(row['split']))
    with zipfile.ZipFile(output_data_dir  + '/full_compressed.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr('full.txt', '\n'.join(string_seq))       

    with open(output_data_dir  + '/stats.txt', 'w') as f:
        f.write('num_users: ' + str(len(user_map)) + '\n')
        num_users_fully_split = 0
        num_users_in_train = 0
        num_users_in_valid = 0
        num_users_in_test = 0
        for key, val in user_map.items():
            stats = val['stats']
            if stats[0] > 0:
                num_users_in_train += 1
            if stats[1] > 0:
                num_users_in_valid += 1  
            if stats[2] > 0:
                num_users_in_test += 1      
            if stats[0] > 0 and stats[1] > 0 and stats[2] > 0:
                num_users_fully_split += 1
        f.write('num_users_fully_split: ' + str(num_users_fully_split) + '\n')
        f.write('num_users_in_train: ' + str(num_users_in_train) + '\n')
        f.write('num_users_in_valid: ' + str(num_users_in_valid) + '\n')
        f.write('num_users_in_test: ' + str(num_users_in_test) + '\n')



datasets = [('beauty', 'Beauty'), ('ml10m', 'ml10m'), ('retailrocket', 'Retail')]

for dataset in datasets:
    process_dataset(dataset[0], dataset[1])