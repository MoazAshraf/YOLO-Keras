import os
import zipfile
from PIL import Image 
import matplotlib.pyplot as plt
import xml.dom.minidom
import json


IMGZIP_PATH = "VOC2012/JPEG.zip"
ANNOTATION_DIR = "VOC2012/Annotations"
LABELS_DIR = "VOC2012/Labels"

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
        label_filepath = os.path.join(LABELS_DIR, imgname.split('/')[-1].split('.')[0] + '.json')
        label = parse_annotation(imgname, is_imagepath=True)
        with open(label_filepath, 'w') as f:
            json.dump(label, f)

zf = open_zipfile(IMGZIP_PATH)

# convert_annotations(zf)

file = zf.namelist()[1000]
label = parse_annotation(file, is_imagepath=True)
img = open_image_from_zip(zf, file)
print(label)
plt.imshow(img)
plt.show()