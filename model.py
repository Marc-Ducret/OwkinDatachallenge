from constants import *
import numpy as np
np.random.seed(seed)
import tensorflow
tensorflow.set_random_seed(seed)
import keras
from keras.layers import *
import dataloading
import os
import pandas as pd
import sklearn


def Sum(axis, keepdims=False):
    return Lambda(lambda x: K.sum(x, axis=axis, keepdims=keepdims))


def Power(k):
    return Lambda(lambda x: K.pow(x, k))


def Exp():
    return Lambda(lambda x: K.exp(x))


def global_model(tile_shape, local_model):

    tiles = Input(shape=(n_tiles,) + tile_shape)
    masks = Input(shape=(n_tiles,))

    local_model_output = local_model(tiles)
    tile_predictions = Reshape((n_tiles,))(local_model_output)
    tile_predictions = Multiply(name='tile_predictions')([tile_predictions, masks])

    prediction = Dense(1, name='prediction', kernel_initializer='ones')(
        Multiply()([
            Sum(axis=1, keepdims=True)(
                Multiply()([
                    Reshape((n_tiles,))(Dense(1, activation='relu', kernel_initializer='ones')(local_model_output)),
                    masks
                ])
            ),
            Power(-1)(Sum(axis=1, keepdims=True)(masks))
        ])
    )

    return keras.Model(inputs=[tiles, masks], outputs=[prediction, tile_predictions])


def annotation_criterion(target, pred):
    annotated = K.cast(K.greater_equal(target, 0), float)
    return K.binary_crossentropy(target * annotated, pred * annotated - 1e10 * (1 - annotated), from_logits=True)


def roc_auc(target, pred):
    mask = K.equal(target[:, None], 1) & K.equal(target[None, :], 0)
    mask = K.cast(mask, float)
    greater = K.cast(K.greater(pred[:, None], pred[None, :]), float)
    return K.sum(greater * mask) / (K.sum(mask) + 1e-5)


def predict_test(model, name):
    test = dataloading.test_loader('../data/test', batch_size=512)
    pred, _ = model.predict_generator(test)
    pred = 1 / (1 + np.exp(-pred))
    df = pd.DataFrame(dict(
        ID=['{:03d}'.format(i) for i in test.ids],
        Target=pred.reshape(-1)
    ))
    df.set_index('ID', inplace=True)
    df.to_csv('../predictions/pred_{}.csv'.format(name))


def auc_callback(model, validation):
    def compute_auc(epoch, logs):
        if len(validation) > 0:
            pred, _ = model.predict_generator(validation)
            pred = 1 / (1 + np.exp(-pred))
            logs['auc'] = sklearn.metrics.roc_auc_score(validation.labels, pred)
        else:
            logs['auc'] = 0
        print('Epoch {:02d} | auc: {:.2f}'.format(epoch+1, logs['auc']))
    return compute_auc


def train_model():
    model = global_model((n_resnet_features,),
                         keras.Sequential((
                            Dropout(.5),
                            Dense(8),
                            LeakyReLU(),
                            Dropout(.5),
                            Dense(16),
                            LeakyReLU(),
                            Dropout(.5),
                            Dense(32),
                            LeakyReLU(),
                            Dropout(.5),
                            Dense(1),
                         ), name='local_model')
    )
    model.summary()

    model.compile(keras.optimizers.Adam(lr=2e-3, decay=1e-3),
                  loss=dict(
                      prediction=annotation_criterion,
                      tile_predictions=annotation_criterion
                  ),
                  loss_weights=dict(
                      prediction=1,
                      tile_predictions=1e1
                  )
    )

    train, validation = dataloading.train_loader('../data/train', validation_ratio=.2,
                                                 train_batch_size=8, validation_batch_size=512)

    model_name = 'model_{:02d}'.format(len(os.listdir('../tensorboard')))
    model.fit_generator(train,
                        callbacks=[
                            keras.callbacks.LambdaCallback(
                              on_epoch_end=auc_callback(model, validation)
                            ),
                            keras.callbacks.ReduceLROnPlateau(verbose=1, monitor='loss'),
                            keras.callbacks.TensorBoard(
                                log_dir='../tensorboard/{}'.format(model_name))],
                        epochs=20, verbose=0)
    model.save('../models/{}.h5'.format(model_name))
    return model, model_name


if __name__ == '__main__':
    predict_test(*train_model())