import cv2
import time
import pickle
import numpy as np
import pandas as pd

import racemachine.log as log
import racemachine.utils as utils
import racemachine.face as face
import racemachine.config as config
from racemachine.face import Face
from racemachine.detector import Detector

logger = log.get_logger(__name__)

class VideoTracker(object):
    def __init__(self, opts = None):
        self.options     = opts
        self.frame_in    = None
        self.frame_out   = None
        self.frame_count = 0
        self.faces       = []
        self.rects       = None
        self.detector    = Detector()

        self.detector.load()

    def run(self):
        self.__init_video()

        try:
            while True:
                self.__read_video()

                frame = self.frame_in.copy()
                self.frame_count += 1
                pred, rects = self.detector.predict_from_frame(frame)

                if rects:
                    self.rects = pd.DataFrame(rects, columns = ['top', 'right', 'bottom', 'left'])
                    self.rects = pd.concat([pred, self.rects], axis=1)

                    for row in self.rects.iterrows():
                        # If there are no faces, populate the first
                        if len(self.faces) == 0:
                            self.faces.append(Face.find_or_create_face(self.frame_in, row, self.frame_count))
                        else:
                            matched = False
                            for face in self.faces:
                                matched = face.match_from_frame(frame, row)
                                # If a face is matched, update the face rect
                                # and frame_count
                                if matched:
                                    face.update(row, self.frame_count)
                                    break

                            # Create a face for any unmatched rect
                            if not matched:
                                self.faces.append(Face.find_or_create_face(self.frame_in, row, self.frame_count))

                # Discard faces with no matching rect (not in frame)
                self.faces = [ f for f in self.faces if f.frame_count == self.frame_count ]

                logger.debug('frame_count: {}'.format(self.frame_count))
                logger.debug('face count: {}'.format(len(self.faces)))

                self.__draw_faces(self.frame_out)
                self.__show_frame_out()

                if 0xFF & cv2.waitKey(5) == 27:
                    break
        except Exception as e:
            raise e

        finally:
            if config.get('recognizer.persist'):
                Face.save_recognizer()
            cv2.destroyAllWindows()

    def __match_by_image(self, face, row):
        face.match_from_frame(self.frame_in.copy(), row)

    def __init_video(self):
        self.video     = None
        self.frame_in  = None
        self.frame_out = None

        self.video = cv2.VideoCapture(0)

    def __read_video(self):
        ret, frame_in = self.video.read()
        self.frame_in = frame_in
        self.frame_out = frame_in.copy()

    def __draw_faces(self, image):
        for face in self.faces:
            font        = cv2.FONT_HERSHEY_DUPLEX
            label       = "id: #{}".format(face.id)
            attrs       = "{} {}".format(face.race, face.sex)
            image_width = image.shape[1]

            if face.state == 'new':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            utils.draw_rects(self.frame_out, [ face], color)
            utils.draw_msg(image, face.x1, face.y1, label)
            utils.draw_msg(image, face.x1 + 6, face.y2 - 6, attrs)

    def __show_frame_out(self):
        cv2.imshow('img2', self.frame_out)
        time.sleep(0.01)
