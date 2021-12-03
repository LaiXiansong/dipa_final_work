import numpy as np
import cv2
import time

cap = cv2.VideoCapture("../resources/eye/eye_closed_05.avi")
start_time = time.time()
counter = 0 
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
while cap.isOpened():
    ret, frame = cap.read() # ret 返回是否读到图片
    #键盘输入空格暂停，输入q退出
    key = cv2.waitKey(1) & 0xff
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break
    counter += 1#计算帧数
    if (time.time() - start_time) != 0:#实时显示帧数
        # cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (500, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
        #             3)
        # 图像处理部分--------------------------------------------

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # xgrad = cv2.Sobel(frame, cv2.CV_16SC1, 1, 0)  # xGrodient
        # ygrad = cv2.Sobel(frame, cv2.CV_16SC1, 0, 1)  # yGrodient
        # frame = cv2.Canny(xgrad, ygrad, 100, 150)  # edge 

        # ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)

        # ------------------------------------------------------
        cv2.imshow('frame', frame)
        # print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    time.sleep(1 / fps)#按原帧率播放

cap.release()
cv2.destropAllWindows()