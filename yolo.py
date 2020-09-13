import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

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

from data_processing import keras_iou, keras_yolo_coords_to_bndboxes, IMAGE_SHAPE


class YOLODetection(tf.keras.layers.Layer):
    """
    Custom detection layer for YOLO

    - Reshapes the input tensor to the shape (s, s, b * 5 + c)
    - Applies softmax to the class probabilites
    - Applies sigmoid to the box confidence scores and the box coordinates
    """

    def __init__(self, s, b, c, softmax_class_probs=True, sigmoid_box_confs=True, sigmoid_box_coords=True, *args, **kwargs):
        super(YOLODetection, self).__init__(*args, **kwargs)
        self.grid_size = s
        self.num_boxes = b
        self.num_classes = c
        self.softmax_class_probs = softmax_class_probs
        self.sigmoid_box_confs = sigmoid_box_confs
        self.sigmoid_box_coords = sigmoid_box_coords
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'grid_size': self.grid_size,
            'num_boxes': self.num_boxes,
            'num_classes': self.num_classes,
            'softmax_class_probs': self.softmax_class_probs,
            'sigmoid_box_confs': self.sigmoid_box_confs,
            'sigmoid_box_coords': self.sigmoid_box_coords
        })
        return config
    
    def call(self, x):
        m = K.shape(x)[0]
        s = self.grid_size
        b = self.num_boxes
        c = self.num_classes

        class_probs_end = s * s * c
        box_confs_end = class_probs_end + s * s * b

        # class probabilities
        class_probs = K.reshape(x[:, :class_probs_end], (m, s, s, c))
        if self.softmax_class_probs:
            class_probs = K.softmax(class_probs)

        # box confidence scores
        box_confs = K.reshape(x[:, class_probs_end:box_confs_end], (m, s, s, b))
        if self.sigmoid_box_confs:
            box_confs = K.sigmoid(box_confs)
        
        # box coordinates
        box_coords = K.reshape(x[:, box_confs_end:], (m, s, s, b * 4))
        if self.sigmoid_box_coords:
            box_coords = K.sigmoid(box_coords)
        
        outputs = K.concatenate([class_probs, box_confs, box_coords])
        return outputs


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
            s = section['side']
            b = section['num']
            c = section['classes']

            model.add(YOLODetection(s, b, c, name=f'detection_{block_index}'))
        
        block_index += 1
    
    return model

def yolo_loss(y_true, y_pred, s=7, b=3, c=20, image_shape=IMAGE_SHAPE[:2], sqrt=True,
              object_scale=1, noobject_scale=.5, class_scale=1, coord_scale=5):
    """
    Custom YOLO loss function
    """

    # truth classes
    truth_classes = y_true[..., :c]   # (m, s, s, c)
    
    # the truth label has one box per cell, if its confidence score is 1 then there is an object
    truth_is_obj = y_true[..., c]
    truth_is_obj = K.expand_dims(truth_is_obj)  # (m, s, s, 1)

    # truth box coordinates
    truth_box_coords = y_true[..., c+b:c+b+4] # the box coordinates of the truth label (zeros if there is no object)
    truth_box_coords = K.reshape(truth_box_coords, (-1, s, s, 1, 4))   # (m, s, s, 1, 4)

    # prediction class probabilites
    pred_class_probs = y_pred[..., :c]    # (m, s, s, c)

    # prediction box confidence scores
    pred_box_confs = y_pred[..., c:c+b]   # (m, s, s, b)

    # prediction box coordinates
    pred_box_coords = y_pred[..., c+b:]
    pred_box_coords = K.reshape(pred_box_coords, (-1, s, s, b, 4))  # (m, s, s, b, 4)

    # convert box coordinates to bounding box limits
    truth_box_min_xy, truth_box_max_xy = keras_yolo_coords_to_bndboxes(truth_box_coords, image_shape)
    pred_box_min_xy, pred_box_max_xy = keras_yolo_coords_to_bndboxes(pred_box_coords, image_shape)

    # find the box with the best IoU with the truth box
    iou_scores = keras_iou(pred_box_min_xy, pred_box_max_xy, truth_box_min_xy, truth_box_max_xy)
    best_ious = K.max(iou_scores, axis=-1, keepdims=True)
    best_box_mask = K.cast(iou_scores >= best_ious, K.dtype(iou_scores))

    # if the jth bounding box predictor in cell i is responsible for that prediction it will be 1; otherwise will be 0
    is_box_responsible = truth_is_obj * best_box_mask   # (m, s, s, b)

    # coord_loss = is_box_responsible * (K.square())


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
    import zipfile
    from io_utils import *

    model = create_model_from_cfg(parse_cfg('yolov1.cfg'))
    model.summary()

    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        image_paths = images_zip.namelist()[1:]
        label_paths = [get_labelpath_from_imagename(get_filename(x)) for x in image_paths]

        data_gen = DataGenerator(image_paths, label_paths, 4, from_zip=True, zip_file=images_zip)

        x, y = data_gen[0]

    print(x.shape)
    y_pred = model.predict(x)
    print(y_pred)
    print(yolo_loss(y, y_pred))
    
    # model = load_pretrained_darknet('yolov1.cfg', 'yolov1.weights')
