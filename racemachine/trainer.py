import racemachine.detector as detector
import racemachine.recognizer as recognizer

class Trainer(object):
    def train_detector(self):
        detector.Detector().train

    def train_recognizer(self):
        recognizer.singleton.train()
        recognizer.singleton.save()
