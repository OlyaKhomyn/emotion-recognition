import cv2
import torchvision.transforms as T
import time

from consts import LABEL_TO_CLASS
from utils import load_model, make_prediction, load_cnn
from dataset import *


def features_model():
    model = load_model()

    vid = cv2.VideoCapture(0)

    neutral = True

    while True:
        ret, frame = vid.read()

        # capture neutral face
        if neutral:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', image)

            k = cv2.waitKey(1)
            if k % 256 == 32:
                print('capturing neutral face')

                image = extract_face(image)
                if not list(image):
                    print("failed extracting face.")

                neu_img, neu_shape = extract_landmarks(image, PRED_PATH, False)
                if not list(neu_shape):
                    print('failed extracting neutral shape')
                    continue

                neu_img, neu_shape = normalize(neu_img, neu_shape)
                neutral = False
                cv2.destroyAllWindows()
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', image)

            print('start detecting')

            t = time.time()
            features = get_feature_vector(image, neu_shape)

            if not list(features):
                continue

            pred_cl = make_prediction(model, features)
            cl = LABEL_TO_CLASS[pred_cl]

            print("sec {}".format(time.time()-t))
            print(cl)

            cv2.putText(image, cl, (0, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Res', image)

    vid.release()
    cv2.destroyAllWindows()


def cnn():
    vid = cv2.VideoCapture(0)
    model = load_cnn()

    while True:
        ret, frame = vid.read()
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t = time.time()
        im = extract_col_face(frame)
        if not list(im):
            print("failed extracting face.")
            continue

        transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(128, 128)),
        ])

        im = transform(im)
        im = im.unsqueeze(0)

        pred_cl = make_prediction(model, im)
        cl = LABEL_TO_CLASS[pred_cl]

        print("sec {}".format(time.time() - t))

        print(cl)

        cv2.putText(frame, cl, (0, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('Res', frame)

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
