import os
import glob
import shutil
import numpy as np

data_path = './data/Training_Dataset_Cropped/'
dest = './data/Training_Dataset_Cropped_Split/'

filenames = sorted(glob.glob(data_path + '*.png'))
filenames = [os.path.basename(f) for f in filenames]
labels = [f[0] for f in filenames]
# create (filename, label) pairs
ds = list(zip(filenames, labels))

# shuffle the dataset using a reproducible seed
rng = np.random.default_rng(0)
idxs = rng.permutation(len(filenames))
ds = [ds[i] for i in idxs] # re-order the dataset
filenames, labels = zip(*ds) # unpack the dataset

# split the dataset
for label in set(labels):
    # get the files for the label
    files = [f for f, l in ds if l == label]
    train_size = int(0.8 * len(files))
    train_files = files[:train_size]
    test_files = files[train_size:]
    # create the directories
    os.makedirs(dest + 'train/' + label, exist_ok=True)
    os.makedirs(dest + 'val/' + label, exist_ok=True)
    # copy the files
    for f in train_files:
        shutil.copy(data_path + f, dest + 'train/' + label + '/' + f)
    for f in test_files:
        shutil.copy(data_path + f, dest + 'val/' + label + '/' + f)
    