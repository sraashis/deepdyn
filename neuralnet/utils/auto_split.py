"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import json
import os
import sys

try:
    BASE_PROJECT_DIR = '/home/ak/PycharmProjects/ature'
    sys.path.append(BASE_PROJECT_DIR)
    os.chdir(BASE_PROJECT_DIR)
except:
    BASE_PROJECT_DIR = '/home/akhanal1/ature'
    sys.path.append(BASE_PROJECT_DIR)
    os.chdir(BASE_PROJECT_DIR)


def load_split_json(json_file):
    if os.path.isfile(json_file):
        with open(json_file) as f:
            print('####SPLIT FOUND####: ', json_file + ' Loaded')
            return json.load(f)


def create_splits(files, test_count, val_count, file_name):
    for i in range(0, len(files), test_count):
        test = files[i:i + test_count]
        print(len(test))
        t_val = [item for item in files if item not in test]
        validation = t_val[0:val_count]
        train = [item for item in t_val if item not in validation]
        configuration = {
            'train': train,
            'validation': validation,
            'test': test
        }
        f = open(str(i) + file_name, "w")
        f.write(json.dumps(configuration))
        f.close()
