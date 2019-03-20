from dataloading import *
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from constants import *


def load_from_folder(folder):
    ids, annotated, _ = explore_dataset(folder)
    features, masks = load_resnet_features(folder, ids, annotated)
    return features.reshape(-1, 2048)[masks.reshape(-1)]


def save_pca_features(folder, pca):
    ids, annotated, _ = explore_dataset(folder)
    features, masks = load_resnet_features(folder, ids, annotated)
    for i, id in enumerate(tqdm(ids)):
        np.save(
            '{}/pca/ID_{:03d}{}.npy'.format(folder, id, '_annotated' if annotated[i] else ''),
            pca.transform(features[i][masks[i]])
        )


def main():
    train_features = load_from_folder('../data/train')
    print('train features: {}'.format(train_features.shape))
    # test_features = load_from_folder('../data/test')
    # print('test features: {}'.format(test_features.shape))

    pca = PCA(n_components=n_pca, svd_solver='randomized')
    # pca.fit(np.concatenate((train_features, test_features)))
    pca.fit(train_features)

    save_pca_features('../data/train', pca)
    save_pca_features('../data/test', pca)


if __name__ == '__main__':
    main()
