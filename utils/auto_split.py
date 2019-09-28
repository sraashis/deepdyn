"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import json


def load_split_json(json_file):
    try:
        f = open(json_file)
        f = json.load(f)
        print('### SPLIT FOUND: ', json_file + ' Loaded')
        return f
    except:
        print(json_file + ' FILE NOT LOADED !!!')


def create_splits(files, k=0, json_file='SPLIT', shuffle_files=True):
    from random import shuffle
    from itertools import chain
    import numpy as np

    json_file = json_file.split('.')[0]
    if shuffle_files:
        shuffle(files)

    ix_splits = np.array_split(np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        print('Valid:', set(files) - set(list(chain(*splits.values()))) == set([]))

        f = open(json_file + '_' + str(i) + '.json', "w")
        f.write(json.dumps(splits))
        f.close()
