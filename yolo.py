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

def parse_cfg(filepath):
    """
    Parses a configuration file into a list of dictionaries where each
    dictionary represents a block.
    """

    cfg = []

    with open(filepath, 'r') as f:
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

def create_model_from_cfg(cfg):
    """
    Creates a tf.keras model using a configuration dictionray
    as returned by parse_cfg().
    """

    # create the model
    model = tf.keras.models.Sequential()

    def add_conv_or_local(is_conv, section):
        
        filters = section['filters']
        kernel_size = section['size']
        strides = section['stride']

        # add padding
        padding = section['pad']
        if padding == 1:
            pad = (kernel_size - 1) // 2
        else:
            pad = 0
        model.add(ZeroPadding2D(pad))

        # check batch norm
        try:
            batch_normalize = (section['batch_normalize'] == 1)
        except:
            batch_normalize = False
        use_bias = not batch_normalize

        # add the 2d convolution or locally connected layer
        layer_class = Conv2D if is_conv else LocallyConnected2D
        model.add(layer_class(filters, kernel_size, strides, use_bias=use_bias))

        # add batch norm layer
        if batch_normalize:
            model.add(BatchNormalization())

        # add the activation
        activation = section['activation']
        if activation == 'lrelu':
            model.add(LeakyReLU(alpha=0.1))

    for name, section in cfg:
        if name == 'net':
            input_shape = (section['height'], section['width'], section['channels'])
            model.add(InputLayer(input_shape=input_shape))
        elif name == 'convolutional':
            add_conv_or_local(True, section)
        elif name == 'maxpool':
            pool_size = section['size']
            strides = section['stride']
            model.add(MaxPooling2D(pool_size, strides))
        elif name == 'local':
            add_conv_or_local(False, section)
        elif name == 'dropout':
            rate = section['probability']
            model.add(Dropout(rate))
        elif name == 'connected':
            units = section['output']
            activation = section['activation']
            
            model.add(Flatten())
            model.add(Dense(units))
            
            if activation == 'lrelu':
                model.add(LeakyReLU(alpha=0.1))
        elif name == 'detection':
            classes = section['classes']
            grid_size = section['side']
            boxes_per_cell = section['num']

            output_shape = (grid_size, grid_size, 5 * boxes_per_cell + classes)
            model.add(Reshape(output_shape))
    
    return model

def load_pretrained_darknet(cfg_file, weights_file):
    """
    Loads a pretrained darknet model from a cfg file and a weights file
    """

    cfg = parse_cfg(cfg_file)

    with open(weights_file, 'rb') as wf:
        # parse the header
        header = np.fromfile(wf, dtype=np.int32, count=5)
        seen = header[3]

        # load the rest of the weights
        all_weights = np.fromfile(wf, dtype=np.float32)
    
    model = create_model_from_cfg(cfg)
    # TODO: add weights

    return model

if __name__ == "__main__":
    cfg = parse_cfg('yolov1.cfg')
    model = create_model_from_cfg(cfg)
    model.summary()


    # load_pretrained_darknet('yolov1.cfg', None)
    # print(parse_cfg('yolov1.cfg'))
    # yolo_model = create_model()
    # yolo_model.summary()