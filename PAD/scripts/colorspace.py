import numpy as np
import cv2 as cv


def carema():
    cap = cv.VideoCapture(cv.CAP_DSHOW)

    while True:
        ret, img = cap.read()
        if ret is False:
            print("Error grabbing frame from camera")
            break

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_luv = cv.cvtColor(img,cv.COLOR_RGB2YUV)
        img_crcby = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        img_shv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
        cv.imshow("RGB:",img)
        cv.imshow("GRAY:", img_gray)
        cv.imshow("Luv:", img_luv)
        cv.imshow("YCrCb:",img_crcby)
        cv.imshow("HSV:", img_shv)
        key = cv.waitKey(1)
        if key & 0xFF == 27:
            break



carema()