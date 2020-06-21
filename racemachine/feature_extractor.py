# coding: utf-8
from __future__ import print_function
import os
import pandas as pd
from tqdm import tqdm

import face_recognition
from face_recognition import face_locations
# I used mat73_to_pickle.py from https://github.com/emanuele/convert_matlab73_hdf5
from racemachine.mat73_to_pickle import *

class FeatureExtractor(object):
    def run(self):
        # set data paths
        data_dir = './lfwa'
        img_dir = './lfwa'
        attr_path = os.path.join(data_dir, 'lfw_att_73.mat')
        save_feature = config.get('detector.features_path')
        save_label = config.get('detector.labels_path')

        # read data
        print("reading data from {}".format(attr_path))
        print("this process might take several minutes")
        f = h5py.File(attr_path, mode='r')
        data = recursive_dict(f)
        df_label = pd.DataFrame(data['label'].T, columns=data['AttrName'], index=data['name'])
        # change "\\" to "/"
        df_label.index = [name.replace('\\', '/') for name in df_label.index]

        # extract face features using face_recognition.face_encodings
        # take about 10 minutes on my pc
        print("extracting face features from images")
        feature_vecs = []
        fnames = []
        for fname in tqdm(df_label.index):
            img_path = os.path.join(img_dir, fname)
            # face detection
            X_img = face_recognition.load_image_file(img_path)
            X_faces_loc = face_locations(X_img)
            # if the number of faces detected in a image is not 1, ignore the image
            if len(X_faces_loc) != 1:
                continue
            # extract 128 dimensional face features
            faces_encoding = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)[0]
            feature_vecs.append(faces_encoding)
            fnames.append(fname)

        df_feat = pd.DataFrame(feature_vecs, index=fnames)
        df_label = df_label[df_label.index.isin(df_feat.index)]
        df_feat.sort_index(inplace=True)
        df_label.sort_index(inplace=True)

        df_feat.to_csv(save_feature)
        df_label.to_csv(save_label)
