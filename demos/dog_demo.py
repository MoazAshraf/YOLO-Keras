import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt
from yolo import load_pretrained_darknet
from data_utils import open_image, label_image
from preprocess import preprocess_image, prediction_to_label


CFG_FILE = os.path.join(ROOT_DIR, 'yolov1.cfg')
WEIGHTS_FILE = os.path.join(ROOT_DIR, 'yolov1.weights')
DOG_IMAGE = os.path.join(ROOT_DIR, 'demos/dog.jpg')

# load the pretrained YOLO model
yolo_model = load_pretrained_darknet(CFG_FILE, WEIGHTS_FILE)

# load the dog image
dog_image = open_image(DOG_IMAGE)

# preprocess the image and create a "batch"
x = preprocess_image(dog_image)
x = x[None,:,:,:]

# run the YOLO model on the image
y_pred = yolo_model.predict(x)[0]
print(y_pred.shape)

# label the image
pred_label = prediction_to_label(y_pred, dog_image.shape[1], dog_image.shape[0])
labelled_dog_image = label_image(dog_image, pred_label)
print(pred_label)

plt.imshow(labelled_dog_image)
plt.show()