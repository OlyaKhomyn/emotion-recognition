CLASSES = [
    {
        "dir_name": "anger/",
        "label": 0
    },
    {
        "dir_name": "disgust/",
        "label": 1
    },
    {
        "dir_name": "fear/",
        "label": 2
    },
    {
        "dir_name": "happiness/",
        "label": 3
    },
    {
        "dir_name": "neutral/",
        "label": 4
    },
    {
        "dir_name": "sadness/",
        "label": 5
    },
    {
        "dir_name": "surprise/",
        "label": 6
    }
]

LABEL_TO_CLASS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}

TRAINED_MODEL = 'models/trained_sob.pt'
CNN_MODEL = 'models/trained_cnn.pt'

PRED_PATH = 'models/shape_predictor_68_face_landmarks.dat'
