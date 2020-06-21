import sys
import os
import glob
import re
import math
import numpy as np
import cv2
import random
import simplejson as json

import racemachine.config as config
import racemachine.utils as utils
import racemachine.log as log
import racemachine.recognizer as recognizer
from racemachine.model import Face as FaceModel

logger = log.get_logger(__name__)
recognizer = recognizer.singleton

if config.get('recognizer.persist'):
    recognizer.load()

COLS                 = ['Male', 'Asian', 'White', 'Black']
RACES                = ['Asian', 'Black', 'White']
MALE_THRESHOLD       = 0.62
RECOGNIZER_THRESHOLD = 45.0
MAX_IMAGES_COLLECTED = 10

class Face(object):
    # Count of found faces
    obj_count = 1

    @classmethod
    def find_or_create_face(self, frame, rect, frame_count):
        converted = utils.normalize_rect(frame, rect)
        face = None

        if not recognizer.trained:
            logger.info('recognizer is not trained')
            face = Face(rect, frame_count)
            recognizer.train([ converted ], [ face.id ])
        else:
            label, dist = recognizer.predict_from_image(converted)
            logger.info('Face.find_or_create_face: {}, {}'.format(label, dist))

            if dist <= RECOGNIZER_THRESHOLD:
                return Face(rect, frame_count, id=label, state='matched')
            else:
                face = Face(rect, frame_count)
                recognizer.update(converted, face.id)

        return face

    @classmethod
    def save_recognizer(self):
        recognizer.save()

    def __init__(self, rect, frame_count, id=None, state='new'):
        if id is not None:
            self.id = id
        else:
            self.id = random.randint(0, 999999)
            Face.obj_count += 1

        self.rect = rect
        self.state = state
        self.frame_count = frame_count
        self.__assign_coordinates()
        self.__assign_sex()
        self.__assign_race()

        self.model = FaceModel.get_or_create(recognizer_label=self.id)[0]
        self.model.race = self.race
        self.model.sex = self.sex
        self.model.save()

    def match_from_frame(self, frame, rect):
        converted = utils.normalize_rect(frame, rect)
        label, dist  = recognizer.predict_from_image(converted)
        logger.info('face.match_from_frame {}, {}'.format(label, dist))

        if dist <= RECOGNIZER_THRESHOLD:
            self.__update_recognizer(converted)
            self.state = 'matched'
            return True

        return False

    def update(self, newRect, frame_count):
        self.rect = newRect
        self.__assign_coordinates()
        self.frame_count = frame_count

    def __assign_coordinates(self):
        self.y1, self.x2, self.y2, self.x1 = self.rect[1][4:].astype(int)

    def __assign_race(self):
        index = np.argmax([
            self.rect[1]['Asian'],
            self.rect[1]['Black'],
            self.rect[1]['White']
            ])

        self.race = RACES[index]

    def __assign_sex(self):
        if self.rect[1]['Male'] >= MALE_THRESHOLD:
            self.sex = 'Male'
        else:
            self.sex = 'Female'

    def __update_recognizer(self, image):
        if self.model.images_collected >= MAX_IMAGES_COLLECTED:
            return

        self.model.images_collected += 1
        self.model.save()

        recognizer.update(image, self.id)
