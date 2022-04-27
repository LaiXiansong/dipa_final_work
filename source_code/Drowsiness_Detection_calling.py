from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy as np
import cv2
import math
import tkinter as tk
from tkinter import filedialog
	
# 判断有没有打电话的平均灰度阈值
thresh = 75
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

file_path = ''
while file_path =='':
	file_path = filedialog.askopenfilenames(title='请选择一个视频',initialdir='resources/calling/')
choose_file = file_path[0]
print(choose_file[-6:])

cap=cv2.VideoCapture(choose_file) #视频路径
# cap=cv2.VideoCapture('resources/eye/eye_closed_21.avi') #视频路径

# 准备输出视频流
fps =cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# width = 960
# height = int((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) * width)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('../results/calling/Phoning_res_'+choose_file[-6:]+'',fourcc, fps, size)

while True:
	ret, frame=cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 图像预处理
	# 中值滤波
	gray = cv2.medianBlur(gray, 5)
	
	# 高斯滤波
	gray = cv2.GaussianBlur(gray,(3, 3),0)

	# 局部直方图
	calhe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
	gray = calhe.apply(gray)


	# 人脸检测
	faces = detect(gray, 0)
	for face in faces:
		# 人脸关键点预测
		face_68_point = predict(gray, face)
		face_68_point = face_utils.shape_to_np(face_68_point) # 转化为numpy数组

		# 找左右脸颊
		left_x = face_68_point[4, 0]
		left_y = face_68_point[4, 1]

		right_x = face_68_point[12, 0]
		right_y = face_68_point[12, 1]

		k = 1.5
		lenth = 100
		top1 = left_y - lenth
		bottom1 = left_y + lenth
		left1 = left_x - int(k * lenth)
		right1 = left_x

		top2 = right_y - lenth
		bottom2 = right_y + lenth
		left2 = right_x 
		right2 = right_x  + int(k * lenth)

		region1  = gray[top1:bottom1, left1:right1]
		region2  = gray[top2:bottom2, left2:right2]

		mean_left = np.sum(region1) / (right1 - left1) / (bottom1 - top1)
		mean_right = np.sum(region2) / (right2 - left2) / (bottom2 - top2)
		# 判断打电话
		if mean_left > thresh:
			cv2.rectangle(frame, (left1, top1), (right1, bottom1), (0, 255, 0), 3)
			cv2.putText(frame, "calling", (left1, top1-15),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
		elif mean_right > thresh:
			cv2.rectangle(frame, (left2, top2), (right2, bottom2), (0, 255, 0), 3)
			cv2.putText(frame, "calling", (left2, top2-15),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "not calling", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
	out.write(frame)
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xff
	if key == ord(" "):
		cv2.waitKey(0)
	if key == ord("q"):
		break

cap.release() 
out.release()
cv2.destroyAllWindows()