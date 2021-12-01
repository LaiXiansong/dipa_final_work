import numpy as np
import cv2

cap = cv2.VideoCapture("../resources/eye/eye_closed_01.avi")
while True:
    ret, image = cap.read()
    cv2.imshow("windows", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break