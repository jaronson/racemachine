import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import racemachine.config as config
import racemachine.utils as utils
import racemachine.log as log

logger = log.get_logger(__name__)

class Recognizer(object):
    def __init__(self):
        self.model      = cv2.face.LBPHFaceRecognizer_create()
        self.model_path = None
        self.labels     = []
        self.trained    = False

    def load(self, path=None):
        path = path if path else config.get('recognizer.model_path')
        self.model_path = path

        if os.path.isfile(path):
            self.model.read(path)
            self.trained = True

        return False

    def predict_from_image(self, image):
        label, confidence = self.model.predict(np.asarray(image))
        return (label, confidence)

    def save(self, outpath=None):
        outpath = outpath if outpath else config.get('recognizer.model_path')
        self.model.write(outpath)

    def update(self, image, label):
        logger.info('Updating model on label {}'.format(label))
        self.model.update(np.asarray([ image ]), np.asarray([ label ]))
        return label

    def train(self):
        images, labels = self.__read_images()

        # Convert labels to 32bit integers. This is a workaround for 64bit machines.
        labels = np.asarray(labels, dtype=np.int32)

        logger.debug('Training model on {} images'.format(len(images)))
        self.model.train(np.asarray(images), labels)
        self.labels = [ l for l in labels ]
        self.trained = True

    # Load images from a directory with the following format:
    # 001
    #   face_a.png
    #   face_b.png
    # 002
    #   face_a.png
    # The directory numbers are the labels.
    # In the above case, the labels will be [ 1, 1, 2 ].
    def __read_images(self, path=None, limit=None, size=None, ext=None):
        # Set some defaults
        path  = path if path else config.get('recognizer.asset_path')
        limit = int(limit) if limit else config.get('recognizer.face_training_limit')

        if ext is None:
            ext = config.get('recognizer.image_extension') or 'png'

        subdirs = glob.glob('{0}/*'.format(path))
        images  = []
        labels  = []
        count   = 0

        for subdir in subdirs:
            files = glob.glob('{0}/*.{1}'.format(subdir, ext))

            for f in files:
                logger.info('Loading file: {0}'.format(f))
                image = self.__load_image(f, size=size)

                if image is not None:
                    images.append(image)
                    labels.append(int(os.path.basename(subdir)))

            count += 1

            if limit is not None and count >= limit:
                break

        return [images, labels]

    def __load_image(self, filepath, size=None):
        try:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if size is not None:
                image = cv2.resize(image, (size, size))

            return np.asarray(image, dtype=np.uint8)
        except Exception as e:
            print("Unexpected error:", sys.exc_info()[0])
            raise e

singleton = Recognizer()
