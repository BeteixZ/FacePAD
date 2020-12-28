from __future__ import absolute_import
import numpy as np
import cv2 as cv
from sklearn.externals import joblib
from PAD.scripts.perf_utils import lp_wrapper


def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
                                         # flags = cv.CV_HAAR_SCALE_IMAGE
                                         )
    return faces


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


@lp_wrapper()
def antispoofing_detect(model_path):
    clt = None
    try:
        clt = joblib.load(model_path)
    except IOError as e:
        print("Error loading model <" + model_path + ">: {0}".format(e.strerror))
        return None

    cap = cv.VideoCapture(cv.CAP_DSHOW)

    cascPath = "../models/haarcascade_frontalface_default.xml"
    faceCascade = cv.CascadeClassifier(cascPath)

    sample_number = 1
    count = 0
    measures = np.zeros(sample_number, dtype=np.float)

    while True:
        ret, img = cap.read()
        if ret is False:
            print("Error grabbing frame from camera")
            break

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = detect_face(img_gray, faceCascade)

        measures[count % sample_number] = 0

        point = (0, 0)
        for i, (x, y, w, h) in enumerate(faces):
            roi = img[y:y + h, x:x + w]

            img_ycrcb = cv.cvtColor(roi, cv.COLOR_BGR2YCR_CB)
            img_luv = cv.cvtColor(roi, cv.COLOR_BGR2LUV)

            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)

            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))

            prediction = clt.predict_proba(feature_vector)
            prob = prediction[0][1]

            measures[count % sample_number] = rob

            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            point = (x, y - 5)
            print(measures, np.mean(measures))

            if 0 not in measures:
                text = "True"
                if np.mean(measures) > 0.2:
                    text = "False"
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                               thickness=2, lineType=cv.LINE_AA)
                else:
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9,
                               color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

            count += 1
            cv.imshow('img_rgb', img)

            key = cv.waitKey(1)
            if key & 0xFF == 27:
                break


antispoofing_detect("../models/print-attack_trained_models/print-attack_ycrcb_luv_extraTreesClassifier.pkl")
