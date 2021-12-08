
import numpy as np 
import cv2 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade.load('../opencv/haarcascade_frontalface_alt.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
eye_cascade.load('../opencv/haarcascade_righteye_2splits.xml') 

cap = cv2.VideoCapture("../resources/eye/eye_closed_33.avi") 
fourcc = cv2.VideoWriter_fourcc(*'XVID') 

while(True): 
    ret, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    calhe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    gray = calhe.apply(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

    cv2.imshow('img',img) 
    #键盘输入空格暂停，输入q退出
    key = cv2.waitKey(1) & 0xff
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break
cap.release() 
cv2.destroyAllWindows() 