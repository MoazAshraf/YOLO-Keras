import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import math
import zipfile
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from data_processing import preprocess_image, get_truth_from_label


IMAGES_ZIP_PATH = "VOC2012/JPEG.zip"
IMAGES_DIR = "VOC2012/JPEG"
LABELS_DIR = "VOC2012/Labels"

def get_filename(filepath):
    """
    Returns the name of the file without the extension
    """

    filename = os.path.split(filepath)[1].split(".")[0]
    return filename

def load_image(image_path):
    """
    Loads an image from a file
    """

    img = Image.open(image_path)
    img = np.array(img)
    return img

def load_image_from_zip(zip_file, image_path_in_zip):
    """
    Loads an image from a zip file as a numpy array.

    zip_file should be a zipfile.ZipFile instance
    image_path_in_zip should be the image's file path relative to the zip file
    """

    info = zip_file.getinfo(image_path_in_zip)
    data = zip_file.open(info)
    img = Image.open(data)
    img = np.array(img)
    return img

def load_label(label_path):
    """
    Loads a label from a JSON file
    """

    with open(label_path, 'r') as label_file:
        label = json.load(label_file)
    
    return label

def get_labelpath_from_imagename(image_name, labels_dir=LABELS_DIR):
    """
    Returns the file path of the .JSON label corresponding to the specified
    image's file name.
    """

    label_path = os.path.join(labels_dir, image_name + ".json")
    return label_path

def draw_object_label_on_image(image, class_name, bndbox, color=(255, 0, 0)):
    """
    Draws an object label (class name and bounding box) on an image. Returns a numpy array representing the RGB image.
    """

    white = (255, 255, 255)

    xmin, xmax, ymin, ymax = [int(x) for x in bndbox]

    tag_width, tag_height = len(class_name) * 10, 20
    text_offset=8

    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    image = cv2.rectangle(image, (xmin, ymin-tag_height), (xmin+tag_width, ymin), color, 2)
    image = cv2.rectangle(image, (xmin, ymin-tag_height), (xmin+tag_width, ymin), color, -1)
    image = cv2.putText(image, class_name, (xmin, ymin-tag_height+text_offset),
                        cv2.FONT_HERSHEY_PLAIN, 1, white, 1)
    
    return image

def label_image(image, label):
    """
    Labels an image with each object class name and bounding box. Returns a numpy array representing the RGB image.
    """

    for obj in label["objects"]:
        image = draw_object_label_on_image(image, obj['name'], obj['bndbox'])

    return image


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, label_paths, batch_size, from_zip=False, zip_file=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.from_zip = from_zip
        self.zip_file = zip_file
    
    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_paths = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for i in range(0, len(batch_image_paths)):
            image_path = batch_image_paths[i]
            label_path = batch_label_paths[i]

            # load the example
            if self.from_zip:
                image = load_image_from_zip(self.zip_file, image_path)
            else:
                image = load_image(image_path)
            label = load_label(label_path)

            # preprocess the example
            x = preprocess_image(image)
            y = get_truth_from_label(label)

            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)


if __name__ == "__main__":
    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        image_paths = images_zip.namelist()[1:]
        label_paths = [get_labelpath_from_imagename(get_filename(x)) for x in image_paths]

        data_gen = DataGenerator(image_paths, label_paths, 4, from_zip=True, zip_file=images_zip)

        print(data_gen[0][0].shape, data_gen[0][1].shape)