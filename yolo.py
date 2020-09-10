import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    LocallyConnected2D,
    Dense,
    LeakyReLU,
    Flatten,
    Reshape,
    BatchNormalization,
    Dropout
)


def create_model(input_shape=(448, 448, 3), s=7, b=2, c=20, batchnorm=True, lrelu_alpha=0.1):
    """
    Creates the YOLO model using tf.keras as desribed by the paper You Only Look Once.

    Change the input shape
    """

    output_shape=(s, s, b * 5 + c)
    model = tf.keras.models.Sequential()

    def add_conv_layer(filters, kernel_size, strides):
        model.add(Conv2D(filters, kernel_size, strides, padding='same'))
        model.add(LeakyReLU(alpha=lrelu_alpha))
        if batchnorm:
            model.add(BatchNormalization())
    
    def add_maxpool_layer(pool_size=2, strides=2):
        model.add(MaxPooling2D(pool_size, strides, padding='same'))
    
    def add_local2d_layer(filters, kernel_size, strides):
        model.add(ZeroPadding2D(padding=1))
        model.add(LocallyConnected2D(filters, kernel_size, strides))
        model.add(LeakyReLU(alpha=lrelu_alpha))

    def add_dense_layer(units, lrelu_activation=False):
        model.add(Dense(units))
        if lrelu_activation:
            model.add(LeakyReLU(alpha=lrelu_alpha))

    model.add(InputLayer(input_shape=input_shape))

    add_conv_layer(64, 7, strides=2)
    add_maxpool_layer(2, 2)

    add_conv_layer(192, 3, strides=1)
    add_maxpool_layer(2, 2)

    add_conv_layer(128, 1, strides=1)
    add_conv_layer(256, 3, strides=1)
    add_conv_layer(256, 1, strides=1)
    add_conv_layer(512, 3, strides=1)
    add_maxpool_layer(2, 2)

    for _ in range(4):
        add_conv_layer(256, 1, strides=1)
        add_conv_layer(512, 3, strides=1)
    
    add_conv_layer(512, 1, strides=1)
    add_conv_layer(1024, 3, strides=1)
    add_maxpool_layer(2, 2)

    for _ in range(2):
        add_conv_layer(512, 1, strides=1)
        add_conv_layer(1024, 3, strides=1)

    add_conv_layer(1024, 3, strides=1)
    add_conv_layer(1024, 3, strides=2)

    for _ in range(2):
        add_conv_layer(1024, 3, strides=1)

    add_local2d_layer(256, 3, strides=1)
    model.add(Dropout(0.5))

    model.add(Flatten())

    add_dense_layer(np.prod(output_shape))
    model.add(Reshape(output_shape))

    return model

if __name__ == "__main__":
    yolo_model = create_model()
    yolo_model.summary()