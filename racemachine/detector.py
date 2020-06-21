import os
import pandas as pd
import numpy as np
import pickle
import face_recognition
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import racemachine.config as config
import racemachine.face as face

UPSCALE = 1

class Detector(object):
    def __init__(self):
        self.features_path = config.get('detector.features_path')
        self.labels_path   = config.get('detector.labels_path')
        self.model_path    = config.get('detector.model_path')

    def load(self):
        with open(self.model_path, 'rb') as f:
            self.clf, self.labels = pickle.load(f)

    def predict_from_frame(self, frame):
        # Predict face attributes for all detected faces in one image
        face_encodings, locs = self.__extract_features(frame)
        if not face_encodings:
            return None, None
        pred = pd.DataFrame(self.clf.predict_proba(face_encodings), columns = self.labels)
        pred = pred.loc[:, face.COLS]
        return pred, locs

    def train(self):
        # load features and labels
        print("reading data files from {}, and {}".format(self.features_path, self.labels_path))
        df_feat = pd.read_csv(self.features_path, index_col=0)
        df_label = pd.read_csv(self.labels_path, index_col=0)
        print("splitting train/test set (9:1)")
        # split training/test name
        unique_names = list(set([path.split('/')[0] for path in df_feat.index]))
        name_train, name_test = train_test_split(unique_names, test_size = 0.1, random_state = 0)
        name_train, name_test = set(name_train), set(name_test)
        # split training/test images
        idx_train = [path.split('/')[0] in name_train for path in df_feat.index]
        idx_test = [path.split('/')[0] in name_test for path in df_feat.index]
        X_train, Y_train = df_feat[idx_train], df_label[idx_train]
        X_test, Y_test = df_feat[idx_test], df_label[idx_test]
        print("start training MLP")
        # train models
        clf = MLPClassifier(solver='adam',
                            hidden_layer_sizes=(128, 128),
                            activation='relu',
                            max_iter = 5000,
                            verbose=True,
                            tol=1e-4)
        clf.fit(X_train, Y_train)
        print("saving the trained model to {}".format(save_model))
        with open(self.model_path, 'wb') as f:
            pickle.dump([clf, df_label.columns.tolist()], f)

    def __extract_features(self, image):
        # Exctract 128 dimensional features
        locs = face_recognition.face_locations(image, number_of_times_to_upsample = UPSCALE)
        if len(locs) == 0:
            return None, None
        face_encodings = face_recognition.face_encodings(image, known_face_locations=locs)
        return face_encodings, locs
