import keras
from keras.layers import *
import dataloading
from constants import *
import os
import pandas as pd
import sklearn


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
    return K.sum(greater * mask) / (K.sum(mask) + 1e-5)


def predict_test(model, name):
    test = dataloading.test_loader('../data/test', batch_size=512)
    pred, _ = model.predict_generator(test)
    df = pd.DataFrame(dict(
        ID=test.ids,
        Target=pred.reshape(-1)
    ))
    df.set_index('ID', inplace=True)
    df.to_csv('../predictions/pred_{}.csv'.format(name))

    test = dataloading.test_loader('../data/test', batch_size=512)
    pred, _ = model.predict_generator(test)
    df = pd.DataFrame(dict(
        ID=['{:03d}'.format(i) for i in test.ids],
        Target=pred.reshape(-1)
    ))
    df.set_index('ID', inplace=True)
    df.to_csv('../predictions/pred_wtf_{}.csv'.format(name))


def auc_callback(model, validation):
    def compute_auc(epoch, logs):
        if len(validation) > 0:
            pred_val, _ = model.predict_generator(validation)
            logs['auc'] = sklearn.metrics.roc_auc_score(validation.labels, pred_val)
        else:
            logs['auc'] = 0
        print('Epoch {:02d} | auc: {:.2f}'.format(epoch, logs['auc']))
    return compute_auc


def train_model():
    model = global_model((n_resnet_features,), keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(keras.optimizers.Adam(lr=1e-3, decay=1e-3),
                  loss=dict(
                      prediction='binary_crossentropy',
                      tile_predictions=annotation_criterion
                  ),
                  loss_weights=dict(
                      prediction=1,
                      tile_predictions=1e0
                  )
    )

    np.random.seed(113)
    train, validation = dataloading.train_loader('../data/train', validation_ratio=.5,
                                                 train_batch_size=8, validation_batch_size=512)

    model_name = 'model_{:02d}'.format(len(os.listdir('../tensorboard')))
    model.fit_generator(train, validation_data=validation,
                        callbacks=[
                            keras.callbacks.LambdaCallback(
                              on_epoch_end=auc_callback(model, validation)
                            ),
                            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
                            keras.callbacks.TensorBoard(
                                log_dir='../tensorboard/{}'.format(model_name))],
                        epochs=60, verbose=0)
    model.save('../models/{}.h5'.format(model_name))

    return model, model_name


if __name__ == '__main__':
    # predict_test(keras.models.load_model('model.h5', custom_objects=dict(
    #     annotation_criterion=annotation_criterion,
    #     roc_auc=roc_auc,
    # )))
    predict_test(*train_model())
