import os
import zipfile
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


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

if __name__ == "__main__":
    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        image_path_in_zip = "JPEG/2011_004301.jpg"
        image = load_image_from_zip(images_zip, image_path_in_zip)
        image_name = get_filename(image_path_in_zip)
        label_path = get_labelpath_from_imagename(image_name)
        label = load_label(label_path)

        print(type(image))
        print(image.shape)
        print(label)

        image = label_image(image, label)
        plt.imshow(image)
        plt.show()