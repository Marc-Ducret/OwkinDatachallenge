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
from sklearn.metrics import roc_auc_score
import tqdm
import tensorflow as tf


def Sum(axis, keepdims=False):
    return Lambda(lambda x: K.sum(x, axis=axis, keepdims=keepdims))


def Power(k):
    return Lambda(lambda x: K.pow(x, k))


def Exp():
    return Lambda(lambda x: K.exp(x))


def global_model(tile_shape, local_model):

    tiles = Input(shape=(n_tiles,) + tile_shape)
    masks = Input(shape=(n_tiles,))

    local_model_output = local_model(Lambda(lambda x: K.reshape(x, (-1,) + tile_shape))(tiles))
    tile_predictions = Lambda(lambda x: K.reshape(x, (-1, n_tiles)))(local_model_output)
    tile_predictions = Multiply(name='tile_predictions')([tile_predictions, masks])

    sorted_predictions = Lambda(lambda x: tf.nn.top_k(x, n_tiles // 50, sorted=True).values)(tile_predictions)

    prediction = Dense(1, name='prediction', kernel_initializer=keras.initializers.Constant(1 / (n_tiles // 50)))(
        sorted_predictions
    )

    return keras.Model(inputs=[tiles, masks], outputs=[prediction, tile_predictions])


def balanced_criterion(ratio):
    def criterion(target, pred):
        valid = K.cast(K.greater_equal(target, 0), float)
        return 2 * K.abs(target - ratio) * K.binary_crossentropy(target * valid, pred * valid - 1e10 * (1 - valid),
                                                                 from_logits=True)

    return criterion


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
        if validation is not None:
            pred, _ = model.predict_generator(validation)
            pred = 1 / (1 + np.exp(-pred))
            logs['auc'] = roc_auc_score(validation.labels.reshape(-1), pred.reshape(-1))
        else:
            logs['auc'] = 0
        print('Epoch {:02d} | auc: {:.2f}'.format(epoch+1, logs['auc']))
    return compute_auc


def make_model_features(label_ratio, annotation_ratio):
    model = global_model((n_resnet_features,),
                         keras.Sequential((
                             # Dropout(.5),
                             # Dense(64, activation='tanh', kernel_initializer='glorot_uniform'),
                             # LeakyReLU(.01),
                             # Dropout(.1),
                             # Dense(32, activation='tanh', kernel_initializer='glorot_uniform'),
                             # Dense(64, activation='tanh', kernel_initializer='glorot_uniform'),
                             # LeakyReLU(.01),
                             # Dense(64, kernel_initializer='glorot_uniform'),
                             # LeakyReLU(.01),
                             # Dropout(.1),
                             # Dense(32),
                             # LeakyReLU(.01),
                             # Dropout(.5),
                             Dense(1, kernel_initializer='glorot_uniform'),
                         ), name='local_model')
                         )

    model.compile(keras.optimizers.Adam(lr=1e-1, decay=1e1),
                  loss=dict(
                      prediction=balanced_criterion(label_ratio),
                      tile_predictions=balanced_criterion(annotation_ratio)
                  ),
                  loss_weights=dict(
                      prediction=1,
                      tile_predictions=10
                  ),
                  metrics=dict(
                      prediction=[],
                      tile_predictions=[]
                  )
                  )
    return model


def make_model_image(label_ratio, annotation_ratio):
    model = global_model(image_shape,
                         # keras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=1)
                         keras.Sequential((
                             Conv2D(4, 3, activation='relu'),
                             Conv2D(4, 3, activation='relu'),
                             MaxPool2D(2),
                             Conv2D(4, 3, activation='relu'),
                             Conv2D(4, 3, activation='relu'),
                             MaxPool2D(2),
                             Conv2D(4, 3, activation='relu'),
                             Conv2D(4, 3, activation='relu'),
                             MaxPool2D(2),
                             Conv2D(4, 3, activation='relu'),
                             Conv2D(4, 3, activation='relu'),
                             MaxPool2D(2),
                             Conv2D(4, 3, activation='relu'),
                             Conv2D(4, 3, activation='relu'),
                             MaxPool2D(2),
                             Flatten(),
                             Dense(1)
                         ), name='local_model'
                         )
    )

    model.compile(keras.optimizers.Adam(lr=1e-1, decay=1e1),
                  loss=dict(
                      prediction=balanced_criterion(label_ratio),
                      tile_predictions=balanced_criterion(annotation_ratio)
                  ),
                  loss_weights=dict(
                      prediction=1,
                      tile_predictions=10
                  ),
                  metrics=dict(
                      prediction=[],
                      tile_predictions=[]
                  )
                  )
    return model


def train_model():
    train, validation = dataloading.train_loader('../data/train',
                                                 validation_ratio=.2,
                                                 # cross_val=hard_samples,
                                                 train_batch_size=1, validation_batch_size=1,
                                                 image=True)

    model = make_model_image(train.label_ratio, train.annotation_ratio)
    model.summary()

    model_name = 'model_{:02d}'.format(len(os.listdir('../tensorboard')))
    model.fit_generator(train,  # validation_data=validation,
                        callbacks=[
                            keras.callbacks.LambdaCallback(on_epoch_end=auc_callback(model, validation)),
                            keras.callbacks.ReduceLROnPlateau(verbose=1, monitor='loss'),
                            keras.callbacks.TensorBoard(log_dir='../tensorboard/{}'.format(model_name))
                        ],
                        epochs=20, verbose=2)
    model.save('../models/{}.h5'.format(model_name))
    return model, model_name


def cross_val():
    n = 279
    batch_size = 30

    pred = np.zeros(n)

    for i in tqdm.trange(n // batch_size):
        indices = np.arange(i, n, n // batch_size)
        train, validation = dataloading.train_loader('../data/train',
                                                     cross_val=indices,
                                                     train_batch_size=8, validation_batch_size=1)

        model = make_model_image(train.label_ratio, train.annotation_ratio)

        model.fit_generator(train,
                            callbacks=[
                                keras.callbacks.ReduceLROnPlateau(verbose=0, monitor='loss')],
                            epochs=20, verbose=0)

        pred[indices] = model.predict_generator(validation)[0].reshape(-1)
        pred[indices] = 1 / (1 + np.exp(-pred[indices]))
        np.save('../tmp_pred', pred)

    train, _ = dataloading.train_loader('../data/train', train_batch_size=512)
    print(roc_auc_score(train.labels, pred))


if __name__ == '__main__':
    predict_test(*train_model())
    # cross_val()
