# Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
import pandas as pd
import numpy as np
import pickle

data_path = 'data/cifar-10-batches-py'

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

batch_1 = load_cfar10_batch(data_path, '1')

print('First 10 labels:', batch_1[1][:10])