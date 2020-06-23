import base64
import numpy as np
import scipy as sp
import cv2
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import racemachine.color as color
import racemachine.face as face

def draw_msg(dest, x, y, msg):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(dest, msg, (x + 1, y + 1), font, 1.0, color.BLACK, thickness=2, lineType=cv2.LINE_AA) # Shadow
    cv2.putText(dest, msg, (x, y), font, 1.0, color.WHITE, lineType=cv2.LINE_AA)

def draw_rects(img, faces, color):
    for face in faces:
        cv2.rectangle(img, (face.x1, face.y1), (face.x2, face.y2), color, 2)

def equalize_halves(img):
    w,h  = img.shape[1], img.shape[0]
    midX = int(w / 2)

    # Equalize left and right halves separately
    left  = cv2.equalizeHist(img[0:h, 0:midX])
    right = cv2.equalizeHist(img[0:h, midX:w])

    whole = img

    # Combine the left and right halves
    # smoothing the transition between
    for (x,y), value in np.ndenumerate(img):

        try:
            # left 25%
            if x < w/4:
                v = left[y][x]

            # left 25 - 50%
            elif x < w*2/4:
                lv = left[y][x]
                wv = whole[y][x]
                f  = (float(x) - w*1/4) / (w * 0.25)
                v  = round((1 - f) * lv + (f) * wv)

            # right 50 - 75%
            elif x < w*3/4:
                rv = right[y][x-midX]
                wv = whole[y][x]
                f  = (float(x) - w*2/4) / (w * 0.25)
                v  = round((1 - f) * wv + (f) * rv)

            # right 75 - 100%
            else:
                v = right[y][x-midX]
            img[y][x] = v
        except IndexError:
            pass

    return img

def encode_image(img):
    encoded = msgpack.packb(img)
    return base64.b64encode(encoded)

def decode_image(enc):
    decoded = base64.b64decode(enc)
    return msgpack.unpackb(decoded)

def get_dist(arr_a, arr_b):
    return np.linalg.norm(np.asarray(arr_a) - np.asarray(arr_b))

def normalize_rect(image, rect):
    y1, x2, y2, x1 = rect[1][len(face.COLS):].astype(int)
    image2 = image[y1:y2, x1:x2]
    normalized = safely_to_grayscale(image2.copy())
    normalized = equalize_halves(normalized)
    normalized = cv2.equalizeHist(normalized)
    return normalized

def normalize_face(image, face):
    image2     = image[face.y1:face.y2, face.x1:face.x2]
    normalized = safely_to_grayscale(image2.copy())
    normalized = equalize_halves(normalized)
    normalized = cv2.equalizeHist(normalized)
    return normalized

def rotate_face(self, face_image):
    img   = face_image.copy()
    (h,w) = img.shape[:2]
    rects = Face.eye_detector.find(img)

    if len(rects) != 2:
        return

    (e1, e2) = rects

    if e1[0] < e2[0]:
        (r, l) = (e1, e2)
    else:
        (r, l) = (e2, e1)

    utils.draw_rects(img, rects, (0,255,0))

    direction = (r[0] - l[0], r[1] - l[1])
    rotation  = -math.atan2(float(direction[1]), float(direction[0]))
    mat       = cv2.getRotationMatrix2D((l[0], l[1]), rotation, 1.0)
    rotated   = cv2.warpAffine(img, mat, (w, h))
    return rotated

def safely_to_grayscale(image):
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

