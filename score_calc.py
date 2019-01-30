import pandas as pd
import os, glob

path = 'data/DRIVE/UNET_LOGS'

files = [f for f in os.listdir(path) if '-TEST.csv' in f]
gl = glob.glob(os.path.join(path, "*-TEST.csv"))
for f in gl:
    print(f.split('\\')[-1])

df = pd.concat(map(pd.read_csv, gl))
images = set(df['ID'].values)
assert (len(df['ID'].values) == len(images))
print(df.mean())
