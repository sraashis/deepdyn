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


def create_split_json(images_src_dir=None, ratio=[0.64, 0.18, 0.18], to_file='split.json', file_names=None):
    if os.path.isfile(to_file):
        with open(to_file) as f:
            print('####SPLIT FOUND####: ', to_file + ' Loaded')
            return json.load(f)

    image_files = os.listdir(images_src_dir) if images_src_dir is not None else file_names
    n = len(image_files)
    i = int(round(ratio[0] * n))
    j = i + int(round(ratio[1] * n))
    k = j + int(round(ratio[2] * n))
    configuration = {
        'train': image_files[0:i],
        'validation': image_files[i:j],
        'test': image_files[j:k]
    }

    f = open(to_file, "w")
    f.write(json.dumps(configuration))
    f.close()
    return configuration


# create_split_json(images_src_dir='data/AV-WIDE/images', ratio=[0.6, 0.20, 0.20], to_file='WIDE.json')
