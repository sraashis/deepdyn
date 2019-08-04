"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import json
import os
import random
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
    try:
        f = open(json_file)
        f = json.load(f)
        print('### SPLIT FOUND: ', json_file + ' Loaded')
        return f
    except:
        print(json_file + ' FILE NOT LOADED !!!')


def create_splits(files, sep1=('', 0), sep2=('', 0), json_file=None):
    """
    :param files: List of files to split into three disjoint sets of any size
    :param sep1: Key for the result of 1st set and number of items as (k, v)
    :param sep2: Key for the result of 2nd set and number of items as (k, v)
    :param json_file: name of a  file json to write the results to
    :return: Json files with keys 'train', sep1[0], sep2[0] and list of files
    """
    confs = []
    for i in range(0, len(files), sep1[1]):
        sep1s = files[i:i + sep1[1]]
        t_val = [item for item in files if item not in sep1s]
        sep2s = t_val[0:sep2[1]]
        train = [item for item in t_val if item not in sep2s]

        configuration = {}
        configuration['train'] = train
        if sep1[1] > 0:
            configuration[sep1[0]] = sep1s
        if sep2[1] > 0:
            configuration[sep2[0]] = sep2s
        confs.append(configuration)
        if json_file is not None:
            f = open(str(i) + json_file, "w")
            f.write(json.dumps(configuration))
            f.close()

    return confs if json_file is None else None


if __name__ == "__main__":

    files = [
    "27_training.tif",
    "33_training.tif",
    "22_training.tif",
    "24_training.tif",
    "25_training.tif",
    "32_training.tif",
    "26_training.tif",
    "36_training.tif",
    "23_training.tif",
    "29_training.tif",
    "30_training.tif",
    "34_training.tif",
    "28_training.tif",
    "31_training.tif",
    "40_training.tif",
    "21_training.tif",
    "39_training.tif",
    "37_training.tif",
    "35_training.tif",
    "38_training.tif"
  ]
    random.shuffle(files)
    create_splits(files, ('validation', 4), ('test', 0), json_file='UNET-DRIVE.json')
