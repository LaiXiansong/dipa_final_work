import numpy as np
import cv2

cap = cv2.VideoCapture("../resources/eye/eye_closed_01.avi")
while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("windows", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destropAllWindows()