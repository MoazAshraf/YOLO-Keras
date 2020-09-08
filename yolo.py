import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Flatten, Dense, Reshape


def create_model(input_shape=(448, 448, 3), lrelu_alpha=0.1, s=7, b=2, c=20, batchnorm=True):
    """
    Creates the YOLO model using tf.keras as desribed by the paper You Only Look Once.

    The model expects preprocessed images of shape (448, 448, 3) and outputs predictions
    of the shape (7, 7, 30). S = 7, B = 2, C = 20.
    """


    output_shape=(s, s, b * 5 + c)

    model = tf.keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(MaxPooling2D(2, 2, padding='same'))

    model.add(Conv2D(192, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(MaxPooling2D(2, 2, padding='same'))

    model.add(Conv2D(128, kernel_size=1, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(256, kernel_size=1, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(MaxPooling2D(2, 2, padding='same'))

    for _ in range(4):
        model.add(Conv2D(256, kernel_size=1, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=lrelu_alpha))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(512, kernel_size=1, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(1024, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(MaxPooling2D(2, 2, padding='same'))

    for _ in range(2):
        model.add(Conv2D(512, kernel_size=1, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=lrelu_alpha))
        model.add(Conv2D(1024, kernel_size=3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(1024, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Conv2D(1024, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))

    for _ in range(2):
        model.add(Conv2D(1024, kernel_size=3, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=lrelu_alpha))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=lrelu_alpha))

    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=lrelu_alpha))

    model.add(Dense(np.prod(output_shape)))
    model.add(Reshape(output_shape))

    return model

if __name__ == "__main__":
    yolo_model = create_model()
    yolo_model.summary()