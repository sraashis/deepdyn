"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import json
import os
import sys
import random

try:
    BASE_PROJECT_DIR = '/home/ak/PycharmProjects/ature'
    sys.path.append(BASE_PROJECT_DIR)
    os.chdir(BASE_PROJECT_DIR)
except:
    BASE_PROJECT_DIR = '/home/akhanal1/ature'
    sys.path.append(BASE_PROJECT_DIR)
    os.chdir(BASE_PROJECT_DIR)


def load_split_json(json_file):
    try:
        f = open(json_file)
        f = json.load(f)
        print('### SPLIT FOUND: ', json_file + ' Loaded')
        return f
    except:
        print(json_file + ' FILE NOT LOADED !!!')


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

# import neuralnet.mapnet.runs as r
# files = os.listdir((r.DRIVE['Dirs']['image']))
# random.shuffle(files)
# create_splits(files, 10, 10, 'THRNET-DRIVE.json')