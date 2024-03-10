import os
import json
import cv2
import torch

from torch.nn.functional import softmax

from consts import *
from feature_extraction import extract_landmarks, normalize, extract_angles, TextureFeatures
from model import EmotionRecModel, resnet


def save_features():
    with open(JSON_FILE, 'r') as f:
        js = f.read()
        data = json.loads(js)
    i = 0

    for d in data:
        print(i)

        em_img, em_shape = extract_landmarks(d['img'], PRED_PATH)

        if not list(em_shape):
            d['features'] = [0]
            continue

        em_img, em_shape = normalize(em_img, em_shape)

        neu_img, neu_shape = extract_landmarks(d['neutral'], PRED_PATH)

        if not list(neu_shape):
            d['features'] = [0]
            continue

        neu_img, neu_shape = normalize(neu_img, neu_shape)

        features = extract_angles(list(em_shape), list(neu_shape))

        texture = TextureFeatures(em_img, em_shape)
        right_eye_d, left_eye_d, between_d = texture.extract_features()
        features.extend([right_eye_d, left_eye_d, between_d])

        d['features'] = features
        i += 1

    js = json.dumps(data)

    with open(JSON_FILE, "w") as f:
        f.write(js)


def load_model():
    model = EmotionRecModel()
    model.load_state_dict(torch.load(TRAINED_MODEL))
    model.eval()

    return model


def load_cnn():
    model = resnet()
    model.load_state_dict(torch.load(CNN_MODEL))
    model.eval()

    return model


def make_prediction(model, features):
    pred = model(features)
    pred = softmax(pred)
    res = torch.argmax(pred, 1)

    return int(res[0])
