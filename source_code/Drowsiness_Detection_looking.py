from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

file_path = ''
while file_path =='':
	file_path = filedialog.askopenfilenames(title='请选择一个视频',initialdir='resources/looking/')
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
out = cv2.VideoWriter('../results/looking/look_sideway_res_'+choose_file[-6:]+'',fourcc, fps, size)


while True:
	ret, frame=cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 图像预处理
	# 中值滤波
	gray = cv2.medianBlur(gray, 3)
	
	# 高斯滤波
	gray = cv2.GaussianBlur(gray,(3, 3),0)

	# 局部直方图
	calhe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
	gray = calhe.apply(gray)

	# 人脸检测
	faces = detect(gray, 0)
	
	if len(faces) == 0:
		cv2.putText(frame, "looking around", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
	else:
		for face in faces:
			# 人脸关键点预测
			face_68_point = predict(gray, face)
			face_68_point = face_utils.shape_to_np(face_68_point) # 转化为numpy数组
			
			# 视频中镜头基本是在人的右前方，即时检测到人脸，根据右眼和左眼的位置关系判断是否在转头
			# 检测到人脸的情况下，如果右眼到鼻梁的距离比上左眼到鼻梁的距离小于一个阈值，判断为转头
			T = 0.7
			# 右眼内侧点序号：39；左眼内侧点序号：42；鼻梁最高点序号27
			dx_r = abs(face_68_point[39][0] - face_68_point[27][0])
			dx_l = abs(face_68_point[42][0] - face_68_point[27][0])
			if dx_r/dx_l < T:
				cv2.putText(frame, "looking around", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
			else:
				cv2.putText(frame, "not looking around", (10, 30),
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