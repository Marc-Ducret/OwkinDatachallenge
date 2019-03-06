import numpy as np
import pandas as pd
import os
import tqdm
import parse
import keras
from constants import *


def time(f, *args, **kwargs):
    import time
    start = time.time()
    result = f(*args, **kwargs)
    end = time.time()
    print('{:.2f} ms'.format(1000 * (end - start)))
    return result


def explore_dataset(folder):
    ids = []
    reverse_ids = {}
    annotated = []
    for file in os.listdir('{}/resnet_features'.format(folder)):
        if file.endswith('.npy'):
            ids.append(int(file[3:6]))
            annotated.append(file.endswith('annotated.npy'))
            reverse_ids[ids[-1]] = len(ids)-1

    return np.array(ids), np.array(annotated), reverse_ids


def load_resnet_features(folder, ids, annotated):
    features = np.zeros(ids.shape + (n_tiles, n_resnet_features))
    mask = np.zeros(ids.shape + (n_tiles,), dtype=bool)

    for i in range(ids.shape[0]):
        file = '{}/resnet_features/ID_{:03d}{}.npy'.format(folder, ids[i], '_annotated' if annotated[i] else '')
        loaded = np.load(file)[:, -n_resnet_features:]
        features[i, :loaded.shape[0]] = loaded
        mask[i, :loaded.shape[0]] = 1

    return features, mask


def load_labels(folder, ids):
    labels = pd.read_csv('{}/labels.csv'.format(folder))
    assert np.all(labels['ID'] == ids)
    return np.array(labels['Target'])


def load_tile_annotations(folder, reverse_ids, annotated):
    tile_annotations = pd.read_csv('{}/tile_annotations.csv'.format(folder))
    annotations = - np.ones(annotated.shape + (n_tiles,))
    for _, annotation in tile_annotations.iterrows():
        image, label = annotation['Image'], annotation['Target']
        patient_id, tile_id, _ = parse.parse('ID_{:03d}_annotated_tile_{:d}_{}', image)

        assert patient_id in reverse_ids
        assert annotated[reverse_ids[patient_id]]

        annotations[reverse_ids[patient_id]][tile_id] = label

    return annotations


class DataLoader(keras.utils.Sequence):
    # TODO implement image loading
    def __init__(self, folder, ids, annotated, labels=None, annotations=None, batch_size=32,
                 preload=False, shuffle=False):
        self.folder = folder
        self.ids = ids
        self.annotated = annotated
        self.batch_size = batch_size
        self.labels = labels
        self.annotations = annotations
        self.preload = preload
        if self.preload:
            self.features, self.masks = load_resnet_features(folder, ids, annotated)
        self.shuffle = shuffle
        self.permutation = np.arange(len(self.ids))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.ids)+self.batch_size-1) // self.batch_size

    def __getitem__(self, item):
        select = self.permutation[item * self.batch_size:(item+1) * self.batch_size]

        if self.preload:
            feature, mask = self.features[select], self.masks[select]
        else:
            feature, mask = load_resnet_features(self.folder, self.ids[select], self.annotated[select])

        if self.labels is not None:
            return [feature, mask], [self.labels[select], self.annotations[select]]
        else:
            return [feature, mask]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.permutation)


def train_loader(folder, validation_ratio=0, validation_annotated_ratio=0, one_vs_all=None,
                 train_batch_size=32, validation_batch_size=32, shuffle_train=True):

    ids, annotated, reverse_ids = explore_dataset(folder)

    labels = load_labels(folder, ids)
    annotations = load_tile_annotations(folder, reverse_ids, annotated)

    train_select = []
    validation_select = []
    if one_vs_all is not None:
        for i in range(ids.size):
            if i == one_vs_all:
                validation_select.append(i)
            else:
                train_select.append(i)
    else:
        validation_count = int(validation_ratio * ids.size)
        validation_annotated_count = int(validation_annotated_ratio * ids.size)

        assert validation_annotated_count <= validation_count
        assert validation_count <= ids.size

        permutation = np.arange(ids.size)
        np.random.shuffle(permutation)


        validation_annotated_selected = 0

        for i in permutation:
            if len(validation_select) < validation_count and (
                    validation_annotated_selected < validation_annotated_count or not annotated[i]):
                validation_select.append(i)
                validation_annotated_selected += annotated[i]
            else:
                train_select.append(i)

    train_select = np.array(train_select)
    validation_select = np.array(validation_select)

    print('train size: {}, validation size: {}'.format(train_select.size, validation_select.size))

    return (
        DataLoader(folder, ids[train_select], annotated[train_select],
                   labels[train_select], annotations[train_select], batch_size=train_batch_size,
                   preload=False, shuffle=shuffle_train),
        DataLoader(folder, ids[validation_select], annotated[validation_select],
                   labels[validation_select], annotations[validation_select], batch_size=validation_batch_size,
                   preload=False),
    )





def test_loader(folder, batch_size):
    ids, annotated, reverse_ids = explore_dataset(folder)
    return DataLoader(folder, ids, annotated, batch_size=batch_size)
