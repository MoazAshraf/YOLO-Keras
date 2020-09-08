import os
import xmltodict
import json
from collections import OrderedDict


XML_DIR = "VOC2012/Annotations"
JSON_OUT_DIR = "VOC2012/Labels"

def parse_xml_file(filename, xml_dir=XML_DIR):
    """
    Reads an XML file and parses it to a python dictionary.
    """

    filepath = os.path.join(xml_dir, filename)
    with open(filepath) as f:
        content = xmltodict.parse(f.read())
    return content

def create_object_detection_label(annot_dict):
    """
    Uses an annotation (parsed to a dictionary from an XML file) to create a
    python dictionary that represents a label for the object detection task.

    The resulting dictionary has the following format:
    {
        "image-size": {"depth": <depth:int>, "width": <width:int>, "height": <height:int>}
        "objects": [
            {
                "name": "<object-name>",
                "bndbox": [xmin:float, xmax:float, ymin:float, ymax:float],
                "difficult": <difficult:int>}
            },
            ...,
        ]
    }
    """

    annotation = annot_dict['annotation']

    annotation_objects = annotation["object"]
    if isinstance(annotation_objects, OrderedDict):
        annotation_objects = [annotation_objects]
    
    objects = []
    for annot_object in annotation_objects:
        obj = {"name": annot_object["name"]}
        obj["bndbox"] = [
            annot_object["bndbox"]["xmin"],
            annot_object["bndbox"]["xmax"],
            annot_object["bndbox"]["ymin"],
            annot_object["bndbox"]["ymax"],
            ]
        obj["bndbox"] = [float(x) for x in obj["bndbox"]]
        obj["difficult"] = annot_object["difficult"] == '1'
        objects.append(obj)

    label = {
        "image-size": dict(annotation["size"]),
        "objects": objects
    }

    return label

def write_json_object_detection_labels(xml_dir=XML_DIR, json_out_dir=JSON_OUT_DIR):
    """
    Loads all the XML files in a directory and converts them into JSON labels.
    """

    # create the output directory if it doesn't exist
    if not os.path.isdir(json_out_dir):
        os.makedirs(json_out_dir)

    # get a list of all the filenames in the XML directory
    xml_filenames = os.listdir(xml_dir)
    for filename in xml_filenames:
        # parse the XML annotation file
        content = parse_xml_file(filename, xml_dir=xml_dir)
        label = create_object_detection_label(content)

        # write the label to a JSON label file
        json_filename = filename.split('.')[0] + '.json'
        with open(os.path.join(json_out_dir, json_filename), 'w') as f:
            json.dump(label, f)

if __name__ == "__main__":
    write_json_object_detection_labels()