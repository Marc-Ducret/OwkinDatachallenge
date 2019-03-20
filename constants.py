n_tiles = 1000
n_resnet_features = 2048
n_pca = 64
n_features = dict(resnet_features=n_resnet_features, pca=n_pca)
image_shape = (224, 224, 3)
seed = 113
hard_samples = [25,  99, 144, 204,  94, 100, 157, 115,  15,  28, 143, 230, 158,
                142,   6, 203, 175, 153,  53, 255, 130, 272,  14,  23, 273, 189,
                125, 167,   8, 196, 248]

image_loading_split = 10
image_loading_workers = 4
