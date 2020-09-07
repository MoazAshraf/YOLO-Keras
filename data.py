import os
import zipfile
from PIL import Image 
import matplotlib.pyplot as plt
import xml.dom.minidom
import json
import cv2
import numpy as np


IMGZIP_PATH = "VOC2012/JPEG.zip"
ANNOTATION_DIR = "VOC2012/Annotations"
LABELS_DIR = "VOC2012/Labels"
CLASSES_TO_NUM = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19
}

def open_zipfile(filepath):
    """
    Opens a zip file for reading
    """

    f = open(filepath, 'rb')
    zf = zipfile.ZipFile(f)
    return zf

def open_image_from_zip(zf, imagename):
    """
    Opens a PIL image from a zip file
    """

    info = zf.getinfo(imagename)
    data = zf.open(info)
    img = Image.open(data)
    return img

def parse_annotation(filepath, is_imagepath=False):
    if is_imagepath:
        filepath = os.path.join(ANNOTATION_DIR, filepath.split('/')[-1].split('.')[0] + '.xml')
    
    # with open(filepath, 'r') as f:
    #     print(f.read())

    dom = xml.dom.minidom.parse(filepath)
    root = dom.childNodes[0]

    objects = []
    size = {}
    for child in root.childNodes:
        if child.nodeType == xml.dom.Node.ELEMENT_NODE:
            if child.tagName == 'object':
                object = {}
                object['name'] = child.getElementsByTagName("name")[0].firstChild.data
                object['bndbox'] = {
                    'xmin': float(child.getElementsByTagName("xmin")[0].firstChild.data),
                    'xmax': float(child.getElementsByTagName("xmax")[0].firstChild.data),
                    'ymin': float(child.getElementsByTagName("ymin")[0].firstChild.data),
                    'ymax': float(child.getElementsByTagName("ymax")[0].firstChild.data),
                    }
                object['difficult'] = bool(int(child.getElementsByTagName("difficult")[0].firstChild.data))
                objects.append(object)
            elif child.tagName == 'size':
                size['depth'] = int(child.getElementsByTagName("depth")[0].firstChild.data)
                size['width'] = int(child.getElementsByTagName("width")[0].firstChild.data)
                size['height'] = int(child.getElementsByTagName("height")[0].firstChild.data)
    label = {'size': size, 'objects': objects}
    return label

def convert_annotations(zf):
    if not os.path.isdir(LABELS_DIR):
        os.makedirs(LABELS_DIR)

    for imgname in zf.namelist()[1:]:
        label_filepath = get_labelpath_from_imagepath(imgname)
        label = parse_annotation(imgname, is_imagepath=True)
        with open(label_filepath, 'w') as f:
            json.dump(label, f)

def get_labelpath_from_imagepath(imagepath):
    return os.path.join(LABELS_DIR, imagepath.split('/')[-1].split('.')[0] + '.json')

def preprocess_image(image, newsize=(448, 448)):
    """
    Preprocesses a PIL image by following these steps:
    - Resize the image
    - Normalizes it
    """

    image = np.array(image)
    image = cv2.resize(image, newsize)
    image = image / 255.
    return image

def preprocess_label(label, S=7, B=2, C=20):
    """
    Creates a target output using a label dictionary by following these steps:
    - Create an S * S * (B * 5 + C) tensor
    - Use center position and dimensions
    - Normalize bounding box coordinates and dimensions
    - Add confidence for boxes
    - Add onehot encodings of classes
    """

    img_width, img_height = label['size']['width'], label['size']['height']

    label_shape = (S, S, B * 5 + C)
    label_tensor = np.zeros(label_shape)

    for object in label['objects']:
        # target class
        obj_class = CLASSES_TO_NUM[object['name']]

        # target bounding box coordinates and dimensions
        xmin = object['bndbox']['xmin']
        xmax = object['bndbox']['xmax']
        ymin = object['bndbox']['ymin']
        ymax = object['bndbox']['ymax']
        x = (xmin + xmax) / 2 / img_width
        y = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width * S
        h = (ymax - ymin) / img_height * S

        # add the object's data to its cell label
        loc_x, loc_y = x * S, y * S
        cell_x, cell_y = int(loc_x), int(loc_y)
        x, y = (loc_x - cell_x), (loc_y - cell_y)

        if label_tensor[cell_y, cell_x, 0] == 0:
            label_tensor[cell_y, cell_x, 0] = 1
            label_tensor[cell_y, cell_x, 1:5] = x, y, w, h
            label_tensor[cell_y, cell_x, 10+obj_class] = 1
        
    return label_tensor

def read_example(zf, imagepath):
    # load the image
    img = open_image_from_zip(zf, imagepath)
    
    # load the label
    labelpath = get_labelpath_from_imagepath(imagepath)
    with open(labelpath) as label_file:
        label = json.load(label_file)
    
    target = preprocess_label(label)
    
    show_labelled_image(img, target)

    img = preprocess_image(img)

    return img, label

def show_labelled_image(img, target):
    pass

zf = open_zipfile(IMGZIP_PATH)

# convert_annotations(zf)

imagepath = zf.namelist()[1000]
# label = parse_annotation(file, is_imagepath=True)
img = open_image_from_zip(zf, imagepath)
read_example(zf, imagepath)