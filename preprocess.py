import numpy as np
import cv2


CLASS_NAME_TO_INDEX = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19
}
INDEX_TO_CLASS_NAME = list(CLASS_NAME_TO_INDEX.keys())

def preprocess_image(image, newsize=(448, 448)):
    """
    Resizes and normalizes the image
    """

    image = cv2.resize(image, newsize)
    image = image / 255.
    return image

def bndbox_to_coords(bndbox, img_width, img_height, s):
    """
    Given a bounding box in pixel coordinates, the image dimensions and the grid size (s),
    this function returns:

    - x, y: the center coordinates of the bounding box relative to the cell associated
            with the bounding box where (0, 0) is the top left of the cell and (1, 1)
            is the bottom right.
    - w, h: the size of the bounding box in grid units.
    - cell_x, cell_y: coordinates of the cell associated with the bounding box
    """

    xmin, xmax, ymin, ymax = bndbox

    # absolute position in grid units
    x = (xmin + xmax) / 2 / img_width * s
    y = (ymin + ymax) / 2 / img_height * s

    # size in grid units
    w = (xmax - xmin) / img_width   # * s
    h = (ymax - ymin) / img_height  # * s

    # position relative to cell
    cell_x, cell_y = int(x), int(y)
    x, y = (x - cell_x), (y - cell_y)

    return x, y, w, h, cell_x, cell_y

def coords_to_bndbox(x, y, w, h, cell_x, cell_y, img_width, img_height, s, wh_sqrt=False):
    """
    Given bounding box data in the format specified by bndbox_to_coords() as well as
    the image dimensions and the grid size (s), this function returns the bounding
    box in pixel coordinates as (xmin, xmax, ymin, ymax).

    If wh_sqrt is true, the function expects w and h to be the square roots of the
    predicted width and height respectively.
    """

    if wh_sqrt:
        w, h = w ** 2, h ** 2

    x = (x + cell_x) * img_width / s
    y = (y + cell_y) * img_height / s
    w = w * img_width   # / s
    h = h * img_height  # / s

    xmin = round(x - w / 2)
    xmax = round(x + w / 2)
    ymin = round(y - h / 2)
    ymax = round(y + h / 2)

    return xmin, xmax, ymin, ymax

def preprocess_label(label, s=7, b=3, c=20):
    """
    Creates a target label using a label dictionary

    - s is the size of the grid (there will be s*s cells)
    - b is the number of bounding boxes for each cell
    - c is the number of possible classes

    Note: for now this function only uses one bounding box per cell

    Returns an s * s * (b * 5 + c) such that each cell has a (b * 5 * c) vector with
    the following:
    - the confidence score of the object
    - the bounding box coordinates
    - one-hot encoding of the label classs
    """

    img_width, img_height = label['image-size']['width'], label['image-size']['height']

    label_shape = (s, s, b * 5 + c)
    label_tensor = np.zeros(label_shape)

    for object in label['objects']:
        # get object data
        class_index = CLASS_NAME_TO_INDEX[object['name']]
        x, y, w, h, cell_x, cell_y = bndbox_to_coords(object['bndbox'], img_width, img_height, s)

        # add the data to the tensor
        if label_tensor[cell_y, cell_x, 0] == 0:
            label_tensor[cell_y, cell_x, 0] = 1
            label_tensor[cell_y, cell_x, 1:5] = x, y, w, h
            label_tensor[cell_y, cell_x, 10+class_index] = 1
        
    return label_tensor

def threshold_predictions(class_probs, box_confs, threshold=0.2):
    """
    class_probs is an (s, s, c) tensor of class probabilities.
    box_confs is an (s, s, b) tensor of box confidence scores.

    Returns an (s, s, b, c) tensor of class-specific confidence scores for each box
    after discarding (setting to zero) any score lower than the threshold.
    """

    # TODO: implement using a vectorized method

    class_confs = np.zeros((s, s, b, c))
    for cell_y in range(s):
        for cell_x in range(s):
            for box in range(b):
                box_confidence = box_confs[cell_y, cell_x, box]
                for class_index in range(c):
                    class_probability = class_probs[cell_y, cell_x, class_index]
                    class_confidence = class_probability * box_confidence

                    if class_confidence >= threshold:
                        class_confs[cell_y, cell_x, box, class_index] = class_confidence
    
    return class_confs

def get_area(box):
    """
    Returns the area of a box
    """

    xmin, xmax, ymin, ymax = box
    area = (xmax - xmin) * (ymax - ymin)
    return area

def get_intersection(box1, box2):
    """
    Returns the area of intersection between 2 boxes
    """

    xmin1, xmax1, ymin1, ymax1 = box1
    xmin2, xmax2, ymin2, ymax2 = box2

    inter_xmin = max(xmin1, xmin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymin = max(ymin1, ymin2)
    inter_ymax = min(ymax1, ymax2)

    if inter_xmin > inter_xmax:
        inter_xmin, inter_xmax = inter_xmax, inter_xmin
    if inter_ymin > inter_ymax:
        inter_ymin, inter_ymax = inter_ymax, inter_ymin

    inter_area = get_area([inter_xmin, inter_xmax, inter_ymin, inter_ymax])
    return inter_area

def get_iou(box1, box2):
    """
    Returns the intersection over union for 2 boxes
    """

    inter_area = get_intersection(box1, box2)
    union_area = get_area(box1) + get_area(box2) - inter_area
    iou = inter_area / union_area
    return iou

def non_maximal_suppression(class_confs, box_coords, min_iou=0.5):
    """
    Applies non maximal suppression and returns an (s, s, b, c) tensor representing
    the modified class confidence scores.

    NMS works by first choosing the box with maximum confidence and discarding any
    boxes which have an iou larger than min_iou with that box. Then repeats the
    process for the next box with maximum confidence.
    """

    # TODO
    pass

def get_label_from_prediction(pred, img_width, img_height, threshold=0.2, img_depth=3, s=7, b=3, c=20):
    """
    Converts a prediction tensor (output from the network) to a label in the format
    specified by create_labels.create_object_detection_label() except that the
    'difficult' property is omitted.
    """

    # pred is a vector of length (s * s * (b + 5 + c))
    class_probs_end = s*s*c
    box_confs_end = class_probs_end + s*s*b

    class_probs = pred[:class_probs_end].reshape((s, s, c))
    box_confs = pred[class_probs_end:box_confs_end].reshape((s, s, b))
    box_coords = pred[box_confs_end:].reshape((s, s, b, 4))

    # get thresholded class-specific confidence scores
    class_confs = threshold_predictions(class_probs, box_confs, threshold=threshold)

    # 
    objects = []

    for cell_y in range(s):
        for cell_x in range(s):
            for box in range(b):
                if label_tensor[cell_y, cell_x, box*5+4] >= 0:
                    # get bounding box
                    x, y, w, h = label_tensor[cell_y,cell_x,box*5:box*5+4]
                    w, h = w ** 2, h ** 2
                    xmin, xmax, ymin, ymax = coords_to_bndbox(x, y, w, h, cell_x, cell_y, img_width, img_height, s)

                    # get class name
                    class_index = np.argmax(label_tensor[cell_y, cell_x, b*5:])
                    class_name = INDEX_TO_CLASS_NAME[class_index]

                    # add the object the objects list
                    obj = {'name': class_name, 'bndbox': [xmin, xmax, ymin, ymax]}
                    objects.append(obj)

    label = {
        "image-size": {"depth": img_depth, "width": img_width, "height": img_height},
        "objects": objects
    }

    return label

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import zipfile
    from data_utils import open_example_from_zip, label_image, IMAGES_ZIP_PATH

    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        img, label = open_example_from_zip(images_zip, "JPEG/2011_004301.jpg")
        print(type(img))
        print(img.shape)
        print(label)

        img = preprocess_image(img)
        label_tensor = preprocess_label(label)
        resized_label = tensor_to_label(label_tensor, img.shape[1], img.shape[0], img.shape[2])

        img = label_image(img, resized_label)
        plt.imshow(img)
        plt.show()