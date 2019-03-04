import numpy as np
import pandas as pd
import os
import tqdm
import parse

n_tiles = 1000
n_resnet_features = 2048


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

    for i in tqdm.trange(ids.shape[0]):
        file = '{}/resnet_features/ID_{:03d}{}.npy'.format(folder, ids[i], '_annotated' if annotated[i] else '')
        loaded = np.load(file)[:, -n_resnet_features:]
        features[i, :loaded.shape[0]] = loaded
        mask[i, :loaded.shape[0]] = 1

    return features, mask


def load_labels(folder, ids):
    labels = pd.read_csv('{}/labels.csv'.format(folder))
    assert np.all(labels['ID'] == ids)
    return np.array(labels)


def load_tile_annotations(folder, reverse_ids, mask, annotated):
    tile_annotations = pd.read_csv('{}/tile_annotations.csv'.format(folder))
    annotations = {}
    for _, annotation in tile_annotations.iterrows():
        image, label = annotation['Image'], annotation['Target']
        patient_id, tile_id, _ = parse.parse('ID_{:03d}_annotated_tile_{:d}_{}', image)

        if patient_id not in annotations:
            assert patient_id in reverse_ids
            assert annotated[reverse_ids[patient_id]]
            annotations[patient_id] = -np.ones(n_tiles)

        annotations[patient_id][tile_id] = label

    for patient_id in reverse_ids:
        if annotated[reverse_ids[patient_id]]:
            assert patient_id in annotations
            assert (annotations[patient_id][mask[reverse_ids[patient_id]]] >= 0).all()

    return annotations


def generator(data_folder):
    # TODO images, validation, batching, testing

    train_folder = '{}/train'.format(data_folder)
    ids, annotated, reverse_ids = explore_dataset(train_folder)
    features, mask = load_resnet_features(train_folder, ids, annotated)

    labels = load_labels(train_folder, ids)
    annotations = load_tile_annotations(train_folder, reverse_ids, mask, annotated)


if __name__ == '__main__':
    generator('../data')
