import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def create_model(input_shape=(448, 448, 3), s=7, b=2, c=20, batchnorm=True, leaky_alpha=0.1):
    """
    Creates the YOLO model using tf.keras as desribed by the paper You Only Look Once.
    """

    output_shape=(s, s, b * 5 + c)
    model = tf.keras.models.Sequential()

    def add_conv_layer(filters, kernel_size, strides):
        model.add(Conv2D(filters, kernel_size, strides, padding='same'))
        model.add(LeakyReLU(alpha=leaky_alpha))
        if batchnorm:
            model.add(BatchNormalization())
    
    def add_maxpool_layer(pool_size=2, strides=2):
        model.add(MaxPooling2D(pool_size, strides, padding='same'))
    
    def add_local2d_layer(filters, kernel_size, strides):
        model.add(ZeroPadding2D(padding=1))
        model.add(LocallyConnected2D(filters, kernel_size, strides))
        model.add(LeakyReLU(alpha=leaky_alpha))

    def add_dense_layer(units, leaky_activation=False):
        model.add(Dense(units))
        if leaky_activation:
            model.add(LeakyReLU(alpha=leaky_alpha))

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
    block_index = 0
    
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
        model.add(ZeroPadding2D(pad, name=f'pad_{block_index}'))

        # check batch norm
        try:
            batch_normalize = (section['batch_normalize'] == 1)
        except:
            batch_normalize = False
        use_bias = not batch_normalize

        # add the 2d convolution or locally connected layer
        if is_conv:
            layer_class = Conv2D
            layer_name = f'conv_{block_index}'
        else:
            layer_class = LocallyConnected2D
            layer_name = f'local_{block_index}'
        model.add(layer_class(filters, kernel_size, strides, use_bias=use_bias, name=layer_name))

        # add batch norm layer
        if batch_normalize:
            model.add(BatchNormalization(name=f'batchnorm_{block_index}'))

        # add the activation
        activation = section['activation']
        if activation == 'leaky':
            model.add(LeakyReLU(alpha=0.1, name=f'leaky_{block_index}'))

    for name, section in cfg:
        if name == 'net':
            input_shape = (section['height'], section['width'], section['channels'])
            model.add(InputLayer(input_shape=input_shape, name='input_0'))
        elif name == 'convolutional':
            add_conv_or_local(True, section)
        elif name == 'maxpool':
            pool_size = section['size']
            strides = section['stride']
            model.add(MaxPooling2D(pool_size, strides, name=f'maxpool_{block_index}'))
        elif name == 'local':
            add_conv_or_local(False, section)
        elif name == 'dropout':
            rate = section['probability']
            model.add(Dropout(rate, name=f'dropout_{block_index}'))
        elif name == 'connected':
            units = section['output']
            activation = section['activation']
            
            model.add(Flatten(name=f'flatten_{block_index}'))
            model.add(Dense(units, name=f'connected_{block_index}'))
            
            if activation == 'leaky':
                model.add(LeakyReLU(alpha=0.1, name=f'leaky_{block_index}'))
        elif name == 'detection':
            classes = section['classes']
            grid_size = section['side']
            boxes_per_cell = section['num']

            output_shape = (grid_size, grid_size, 5 * boxes_per_cell + classes)
            model.add(Reshape(output_shape, name=f'reshape_{block_index}'))
        
        block_index += 1
    
    return model

def load_pretrained_darknet(cfg_file, weights_file):
    """
    Loads a pretrained darknet model from a cfg file and a weights file
    """

    # create the model
    cfg = parse_cfg(cfg_file)
    model = create_model_from_cfg(cfg)

    # load the model weights
    with open(weights_file, 'rb') as wf:

        # parse the header
        major, minor, revision = np.fromfile(wf, dtype=np.int32, count=3)

        if major * 1 + minor >= 2 and major < 1000 and minor < 1000:
            seen_dtype = np.int64
        else:
            seen_dtype = np.int32
        seen = np.fromfile(wf, dtype=seen_dtype, count=1)
    
        def load_array(shape):
            """
            Loads float32 arrays of the specified shape from the weights file
            """

            weights = np.fromfile(wf, dtype=np.float32, count=np.prod(shape))
            weights = weights.reshape(shape)
            return weights

        # load the weights
        for block_index in range(1, len(cfg)):
            name, section = cfg[block_index]

            if name == 'convolutional':
                conv2d_layer = model.get_layer(f'conv_{block_index}')

                # layer hyperparameters
                f = conv2d_layer.filters
                k_size = conv2d_layer.kernel_size[0]
                c_in = conv2d_layer.input_shape[3]
                use_bias = conv2d_layer.use_bias
                try:
                    batch_normalize = section['batch_normalize'] == 1
                except:
                    batch_normalize = False
                
                # load batch norm parameters
                if batch_normalize:
                    # darknet batch norm weights are stored in this order: beta, gamma, running_mean, running_variance
                    bn_beta = load_array((f,))
                    bn_gamma = load_array((f,))
                    bn_running_mean = load_array((f,))
                    bn_running_variance = load_array((f,))

                    # keras batch norm weights should be set as [gamma, beta, running_mean, running_variance]
                    bn_weights = [bn_gamma, bn_beta, bn_running_mean, bn_running_variance]
                    bn_layer = model.get_layer(f'batchnorm_{block_index}')
                    bn_layer.set_weights(bn_weights)

                # load conv2d biases
                if use_bias:
                    conv2d_bias = load_array((f,))

                # load conv2d kernel weights
                darknet_kernel_shape = (f, c_in, k_size, k_size)
                darknet_kernel_weights = load_array(darknet_kernel_shape)

                # Keras Conv2D kernel weights have the shape (k_size, k_size, c_in, f)
                conv2d_kernel_weights = np.transpose(darknet_kernel_weights, [2, 3, 1, 0])

                # set the layer weights
                conv2d_weights = [conv2d_kernel_weights]
                if use_bias:
                    conv2d_weights.append(conv2d_bias)
                
                conv2d_layer.set_weights(conv2d_weights)

            elif name == 'local':

                local2d_layer = model.get_layer(f'local_{block_index}')

                # layer hyperparameters
                f = local2d_layer.filters
                k_size = local2d_layer.kernel_size[0]
                c_in = local2d_layer.input_shape[3]
                _, h_out, w_out, _ = local2d_layer.output_shape
                use_bias = local2d_layer.use_bias

                # load conv2d biases
                if use_bias:
                    local2d_bias_shape = local2d_layer.output_shape[1:]
                    local2d_bias = load_array(local2d_bias_shape)

                # load local2d kernel weights
                # assumption: darknet locally connected layer kernel weights have the shape (c_in, f, k_size, k_size, h_out, w_out)
                darknet_kernel_shape = (c_in, f, k_size, k_size, h_out, w_out)
                darknet_kernel_weights = load_array(darknet_kernel_shape)

                # keras locally connected layer kernel weights have the shape (h*w, c_in*size*size, f)
                local2d_kernel_shape = (h_out*w_out, c_in*k_size*k_size, f)
                local2d_kernel_weights = np.transpose(darknet_kernel_weights, [4, 5, 0, 2, 3, 1])
                local2d_kernel_weights = local2d_kernel_weights.reshape((local2d_kernel_shape))

                # set the layer weights
                local2d_weights = [local2d_kernel_weights]
                if use_bias:
                    local2d_weights.append(local2d_bias)
                local2d_layer.set_weights(local2d_weights)

            elif name == 'connected':
                
                dense_layer = model.get_layer(f'connected_{block_index}')

                # layer hyperparameters
                n_in = dense_layer.input_shape[1]
                units = dense_layer.units
                use_bias = dense_layer.use_bias

                if use_bias:
                    dense_bias_shape = (units,)
                    dense_bias = load_array(dense_bias_shape)

                # assumption: darkent dense layer weights have the shape (units, n_in)
                dense_kernel_shape = (units, n_in)
                dense_kernel_weights = load_array(dense_kernel_shape)

                # keras dense layer weights have the shape (n_in, units)
                dense_kernel_weights = dense_kernel_weights.T

                # set the layer weights
                dense_weights = [dense_kernel_weights]
                if use_bias:
                    dense_weights.append(dense_bias)
                dense_layer.set_weights(dense_weights)                

    return model

if __name__ == "__main__":
    # print(help(BatchNormalization.get_weights))

    # cfg = parse_cfg('yolov1.cfg')
    # model = create_model_from_cfg(cfg)
    # print([a.shape for a in model.get_layer('batchnorm_1').get_weights()])
    # print(model.get_layer('conv_1').output_shape)

    # print("Conv2D:")
    # print([a.shape for a in model.get_layer('conv_28').get_weights()])
    # print(model.get_layer('conv_28').output_shape)
    # print("Local:")
    # print([a.shape for a in model.get_layer('local_29').get_weights()])
    # print(model.get_layer('local_29').output_shape)
    # model.summary()

    model = load_pretrained_darknet('yolov1.cfg', 'yolov1.weights')

    # print(parse_cfg('yolov1.cfg'))
    # yolo_model = create_model()
    # yolo_model.summary()