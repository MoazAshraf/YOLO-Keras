import argparse

import zipfile
from io_utils import DataGenerator, get_labelpath_from_imagename, get_filename, IMAGES_ZIP_PATH
from yolo import parse_cfg, create_model_from_cfg, yolo_loss
from tensorflow.keras.optimizers import Adam


def main():
    argparser = argparse.ArgumentParser('train')
    argparser.add_argument('cfg', metavar='<cfg-file>', help='Model configuration file')

    # load and parse the model configuration file
    args = argparser.parse_args()
    cfg = parse_cfg(args.cfg)

    # create and compile the model
    model = create_model_from_cfg(cfg)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=yolo_loss)

    # open the images zip file
    with zipfile.ZipFile(IMAGES_ZIP_PATH, 'r') as images_zip:
        image_paths = images_zip.namelist()[1:]
        label_paths = [get_labelpath_from_imagename(get_filename(x)) for x in image_paths]

        # training data batch generator
        batch_size = 4
        train_batch_gen = DataGenerator(image_paths, label_paths, batch_size, from_zip=True, zip_file=images_zip)

        # train the model
        model.fit(x=train_batch_gen,
                  steps_per_epoch=(len(image_paths) // batch_size),
                  epochs=3,
                  workers=4)

if __name__ == '__main__':
    main()