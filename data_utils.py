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

def open_image_from_zip(zip_file, filepath):
    """
    Opens an image from a zip file as a numpy array.
    """

    info = zip_file.getinfo(filepath)
    data = zip_file.open(info)
    img = Image.open(data)
    img = np.array(img)
    return img

def open_example_from_zip(image_zip_file, image_filepath, labels_dir=LABELS_DIR):
    """
    Loads an example image stored in a zip file and its label from a JSON file.
    """

    image = open_image_from_zip(image_zip_file, image_filepath)

    image_name = os.path.split(image_filepath)[1].split(".")[0]
    label_filepath = os.path.join(labels_dir, image_name + ".json")

    with open(label_filepath, 'r') as label_file:
        label = json.load(label_file)
    
    return image, label

def draw_label_on_image(image, class_name, bndbox, color=(255, 0, 0)):
    """
    Draws an object label (class name and bounding box) on an image. Returns a numpy array representing the RGB image.
    """

    white = (255, 255, 255)

    xmin, xmax, ymin, ymax = bndbox

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
        image = draw_label_on_image(image, obj['name'], obj['bndbox'])

    return image

if __name__ == "__main__":
    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        img, label = open_example_from_zip(images_zip, "JPEG/2011_004301.jpg")
        print(type(img))
        print(img.shape)
        print(label)

        img = label_image(img, label)
        plt.imshow(img)
        plt.show()