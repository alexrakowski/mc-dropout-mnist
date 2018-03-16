import argparse
import os
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras import utils
import numpy as np
from tqdm import tqdm

from models import lenet, lenet_all

TENSORBOARD_DIR = './tensorboard'


def make_dirs():
    if not os.path.isdir(TENSORBOARD_DIR):
        os.makedirs(TENSORBOARD_DIR)


def prepare_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    X_test = X_test.astype(np.float32) / 255.

    y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)


def evalute_mc(model, X_test, y_test, sample_times=50):
    batch_size = 1000
    err = 0.
    for batch_id in tqdm(range(X_test.shape[0] // batch_size)):
        # take batch of data
        x = X_test[batch_id * batch_size: (batch_id + 1) * batch_size]
        # init empty predictions
        y_ = np.zeros((sample_times, batch_size, y_test[0].shape[0]))

        for sample_id in range(sample_times):
            # save predictions from a sample pass
            y_[sample_id] = model.predict(x, batch_size)

        # average over all passes
        mean_y = y_.mean(axis=0)
        # evaluate against labels
        y = y_test[batch_size * batch_id: (batch_id + 1) * batch_size]
        # compute error
        err += np.count_nonzero(np.not_equal(mean_y.argmax(axis=1), y.argmax(axis=1)))

    err = err / X_test.shape[0]

    return 1. - err


def main(args):
    # load the data
    (X_train, y_train), (X_test, y_test) = prepare_data()

    # prepare the model
    model = lenet_all if args.mc else lenet
    model = model(
        input_shape=X_train.shape[1:],
        num_classes=10,
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # train the network
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=[TensorBoard(log_dir=os.path.join(TENSORBOARD_DIR, model.name), write_images=True)]
    )

    # evaluate the model
    if args.mc:
        acc = evalute_mc(model, X_test, y_test)
    else:
        (_, acc) = model.evaluate(
            x=X_test,
            y=y_test,
            batch_size=args.batch_size
        )
    print('Validation accuracy: {}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist experiment')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--mc', action='store_true', help='Whether to use the MC Dropout model', default=True)
    args_ = parser.parse_args()

    make_dirs()

    main(args_)
