import cv2
import time
import pickle
import face_recognition
import pandas
import numpy as np

import racemachine.log as log
import racemachine.utils as utils
import racemachine.face as face
from racemachine.face import Face

UPSCALE = 1

logger = log.get_logger(__name__)

class VideoTracker(object):
    def __init__(self, opts = None):
        self.options     = opts
        self.frame_in    = None
        self.frame_out   = None
        self.frame_count = 0
        self.faces       = []
        self.rects       = None

    def run(self):
        model_path = 'face_model.pkl'

        # load the model
        with open(model_path, 'rb') as f:
            clf, labels = pickle.load(f)

        self.__init_video()

        try:
            while True:
                self.__read_video()

                frame = self.frame_in.copy()
                self.frame_count += 1
                pred, rects = self.__predict_from_frame(frame, clf, labels)

                if rects:
                    self.rects = pandas.DataFrame(rects, columns = ['top', 'right', 'bottom', 'left'])
                    self.rects = pandas.concat([pred, self.rects], axis=1)

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

                logger.info('frame_count: {}'.format(self.frame_count))
                logger.info('face count: {}'.format(len(self.faces)))

                self.__draw_faces(self.frame_out)
                self.__show_frame_out()

                if 0xFF & cv2.waitKey(5) == 27:
                    break
        except Exception as e:
            raise e

        finally:
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

    def __extract_features(self, image):
        """Exctract 128 dimensional features
        """
        locs = face_recognition.face_locations(image, number_of_times_to_upsample = UPSCALE)
        if len(locs) == 0:
            return None, None
        face_encodings = face_recognition.face_encodings(image, known_face_locations=locs)
        return face_encodings, locs

    def __predict_from_frame(self, frame, clf, labels):
        """Predict face attributes for all detected faces in one image
        """
        face_encodings, locs = self.__extract_features(frame)
        if not face_encodings:
            return None, None
        pred = pandas.DataFrame(clf.predict_proba(face_encodings), columns = labels)
        pred = pred.loc[:, face.COLS]
        return pred, locs

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
