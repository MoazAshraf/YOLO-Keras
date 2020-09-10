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


def parse_cfg(filepath):
    """
    Parses a configuration file into a list of dictionaries where each
    dictionary represents a block.
    """

    cfg = []

    with open(filepath) as f:
        # current_section = None
        for line in f:
            line = line.strip()

            # skip empty lines and comments
            if line == '' or line.startswith('#'):
                continue

            # the start of a new section
            if line.startswith('['):
                section_name = line[1:-1]
                current_section = {}
                cfg.append((section_name, current_section))
            else:
                key, values = [s.strip() for s in line.split('=')]
                values = [s.strip() for s in values.split(',')]

                # parse the value
                for i, value in enumerate(values):
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except:
                            # string value
                            pass
                    values[i] = value

                if len(values) == 1:
                    values = values[0]
                
                current_section[key] = values
    
    return cfg


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
    print(parse_cfg('yolov1.cfg'))
    # yolo_model = create_model()
    # yolo_model.summary()