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

    def train(self, images, labels):
        # Convert labels to 32bit integers. This is a workaround for 64bit machines.
        labels = np.asarray(labels)

        logger.debug('Training model on {} images'.format(len(images)))
        self.model.train(np.asarray(images), labels)
        self.trained = True

    def update(self, image, label):
        self.model.update(np.asarray([ image ]), np.asarray([ label ]))
        return label
