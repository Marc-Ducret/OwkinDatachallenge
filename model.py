import keras
from keras.layers import *
import dataloading
from constants import *
import os


def Sum(axis, keepdims=False):
    return Lambda(lambda x: K.sum(x, axis=axis, keepdims=keepdims))


def Power(k):
    return Lambda(lambda x: K.pow(x, k))


def global_model(tile_shape, local_model):

    tiles = Input(shape=(n_tiles,) + tile_shape)
    masks = Input(shape=(n_tiles,))

    tile_predictions = Reshape((n_tiles,))(
        local_model(
            Reshape((n_tiles,) + tile_shape)(tiles)
        )
    )
    tile_predictions = Multiply(name='tile_predictions')([tile_predictions, masks])

    prediction = Dense(1, activation='sigmoid', name='prediction', kernel_initializer='ones')(
        Multiply()([
            Sum(axis=1, keepdims=True)(tile_predictions),
            Power(-1)(Sum(axis=1, keepdims=True)(masks))
        ])
    )

    return keras.Model(inputs=[tiles, masks], outputs=[prediction, tile_predictions])


def annotation_criterion(target, pred):
    annotated = K.cast(K.greater_equal(target, 0), float)
    return K.binary_crossentropy(target * annotated, pred * annotated)


def roc_auc(target, pred):
    mask = K.equal(target[:, None], 1) & K.equal(target[None, :], 0)
    mask = K.cast(mask, float)
    greater = K.cast(K.greater(pred[:, None], pred[None, :]), float)
    return K.sum(greater * mask) / K.sum(mask)


if __name__ == '__main__':
    model = global_model((n_resnet_features,), keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(keras.optimizers.Adam(lr=1e-3, decay=1e-3),
                  loss=dict(
                      prediction='binary_crossentropy',
                      tile_predictions=annotation_criterion
                  ),
                  loss_weights=dict(
                      prediction=1,
                      tile_predictions=10
                  ),
                  metrics=dict(
                      prediction=roc_auc,
                      tile_predictions=roc_auc
                  )
    )

    train, validation = dataloading.train_loader('../data/train', validation_ratio=.5,
                                                 train_batch_size=16, validation_batch_size=512)
    model.fit_generator(train, validation_data=validation,
                        callbacks=[keras.callbacks.TensorBoard(
                            log_dir='../tensorboard/{:02d}'.format(len(os.listdir('../tensorboard'))))],
                        epochs=100, verbose=2)
