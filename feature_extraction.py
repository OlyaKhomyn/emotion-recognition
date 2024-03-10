import numpy as np
import dlib
import cv2
import torch

from imutils import face_utils

from consts import PRED_PATH
import matplotlib.pyplot as plt
from sobel import horizontal_sobel_filter


def extract_landmarks(image, pred_path, read=True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pred_path)

    if read:
        image = cv2.imread(image, 0)

    rect = detector(image, 0)
    if not rect:
        return image, []

    rect = rect[0]
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    return image, shape


def normalize(image, shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # the angle between eye centroids
    left_eye_center = left_eye.mean(axis=0).astype("int")
    right_eye_center = right_eye.mean(axis=0).astype("int")

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eyes_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                   int((left_eye_center[1] + right_eye_center[1]) // 2))

    M = cv2.getRotationMatrix2D(eyes_center, float(angle), 1)

    s = np.insert(shape, 2, 1, axis=1)
    new_shape = np.matmul(M, s.transpose()).transpose()

    image = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]),
                           flags=cv2.INTER_CUBIC)
    return image, new_shape


def extract_angles(emotion_shape, neutral_shape):
    angles = []

    em_shape = emotion_shape[17:22] + emotion_shape[36:40] + emotion_shape[48:55] + \
               emotion_shape[60:65] + emotion_shape[65:68] + emotion_shape[55:60]
    neu_shape = neutral_shape[17:22] + neutral_shape[36:40] + neutral_shape[48:55] + \
                neutral_shape[60:65] + neutral_shape[65:68] + neutral_shape[55:60]

    for em, neu in zip(em_shape, neu_shape):
        dY = em[1] - neu[1]
        dX = em[0] - neu[0]
        angle = np.arctan2(dY, dX)
        angles.append(angle)

    return angles


class RegionOfInterest:
    def __init__(self, shape):
        self.shape = shape

    def right_eye(self):
        x1 = int(self.shape[17][0])
        y1 = int(self.shape[17][1])

        x2 = int(self.shape[36][0])
        y2 = int(self.shape[36][1])
        h = abs(y2 - y1)
        y2 += h
        # top left, bottom right
        return (x1, y1), (x2, y2)

    def left_eye(self):
        x1 = int(self.shape[45][0])
        y1 = int(self.shape[26][1])

        x2 = int(self.shape[26][0])
        y2 = int(self.shape[45][1])
        h = abs(y2 - y1)
        y2 += h
        # top left, bottom right
        return (x1, y1), (x2, y2)

    def between_eyes(self):
        right_eye = self.shape[39][0]
        left_eye = self.shape[42][0]

        d = abs(left_eye-right_eye) / 4

        x1 = int(self.shape[27][0] - d)
        y1 = int(self.shape[27][1])

        x2 = int(self.shape[27][0] + d)
        y2 = int(self.shape[28][1])
        h = abs(y2 - y1)
        y1 -= h

        return (x1, y1), (x2, y2)

    def show_roi(self, frame_name, image):
        right = self.right_eye()
        left = self.left_eye()
        between = self.between_eyes()

        cv2.rectangle(image, right[0], right[1], (255, 0, 0))
        cv2.rectangle(image, left[0], left[1], (255, 0, 0))
        cv2.rectangle(image, between[0], between[1], (255, 0, 0))

        cv2.imshow(frame_name, image)


class TextureFeatures:
    def __init__(self, image, shape):
        self.image = image
        self.shape = shape

        self.roi = RegionOfInterest(self.shape)

        self.sobel = self.detect_edges()

    def detect_edges(self):
        edges = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3, scale=1)

        edges = cv2.convertScaleAbs(edges)
        edges[edges < 100] = 0

        # edges = horizontal_sobel_filter(self.image)

        self.roi.show_roi("sobel", edges)
        cv2.imshow("sobel", edges)

        return edges

    def calc_density(self, roi):
        (x1, y1), (x2, y2) = roi

        s = 0
        for row in range(y1, y2):
            for col in range(x1, x2):
                s += self.sobel[row][col] / 255

        n = (y2 - y1) * (x2 - x1)
        s /= n

        return s

    def extract_features(self):
        right_eye = self.roi.right_eye()
        right_eye_d = self.calc_density(right_eye)

        left_eye = self.roi.left_eye()
        left_eye_d = self.calc_density(left_eye)

        between = self.roi.between_eyes()
        between_d = self.calc_density(between)

        return right_eye_d, left_eye_d, between_d


def extract_face(image):
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if not list(face):
        return []

    x, y, w, h = face[0]
    image = image[y:y + h, x:x + w]
    return image


def extract_col_face(image):
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(
        im, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if not list(face):
        return []

    x, y, w, h = face[0]
    image = image[y:y + h, x:x + w]
    return image


def get_feature_vector(em_image, neu_shape):
    image = extract_face(em_image)
    if not list(image):
        print("failed extracting face.")
        return []
    image, shape = extract_landmarks(image, PRED_PATH, False)

    for (x, y) in shape:
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)

    if not list(shape):
        print("failed extracting shape.")
        return []

    image, shape = normalize(image, shape)
    features = extract_angles(list(shape), list(neu_shape))

    print(features)

    texture = TextureFeatures(image, shape)
    right_eye_d, left_eye_d, between_d = texture.extract_features()

    # plt.bar(["Right eye", "Left eye", "Between eyes"], [right_eye_d, left_eye_d, between_d], color='maroon', width=0.4)
    # plt.xlabel("Region of interest")
    # plt.ylabel("Density")
    # plt.title("Texture-based features")
    # plt.draw()
    # plt.pause(0.1)

    features.extend([right_eye_d, left_eye_d, between_d])

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    return features
