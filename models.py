"""
Note that these models are based on 
https://github.com/yaringal/DropoutUncertaintyCaffeModels/blob/master/mnist_uncertainty/lenet_all_dropout_deploy.prototxt
rather than the original LeNet model.
"""
from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


def lenet(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=20, kernel_size=5, strides=1)(inp)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=50, kernel_size=5, strides=1)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inp, x, name='lenet-none')


def lenet_all(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=20, kernel_size=5, strides=1)(inp)
    x = Dropout(0.5)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=50, kernel_size=5, strides=1)(x)
    x = Dropout(0.5)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inp, x, name='lenet-all')
