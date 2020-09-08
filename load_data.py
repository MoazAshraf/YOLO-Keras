import os
import zipfile
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        img, label = open_example_from_zip(images_zip, "JPEG/2011_004301.jpg")
        print(type(img))
        print(img.shape)
        print(label)

        plt.imshow(img)
        plt.show()