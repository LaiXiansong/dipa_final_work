from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def eyeAspectRatio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
# 判断睁闭眼的ear系数阈值
thresh = 0.32
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
# 眼睛对应的点
(l_eye_start, l_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(r_eye_start, r_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
# 眉毛对应的点
(l_eyeb_start, l_eyeb_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]
(r_eyeb_start, r_eyeb_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]


file_path = ''
while file_path =='':
	file_path = filedialog.askopenfilenames(title='请选择一个视频',initialdir='resources/eye/')
choose_file = file_path[0]

cap=cv2.VideoCapture(choose_file) #视频路径
# cap=cv2.VideoCapture('resources/eye/eye_closed_21.avi') #视频路径

# 准备输出视频流
fps =cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# width = 960
# height = int((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) * width)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('../results/eye/eye_closed_res_'+choose_file[-6:]+'',fourcc, fps, size)

while True:
	ret, frame=cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 图像预处理
	# 中值滤波
	gray = cv2.medianBlur(gray, 3)

	# 均值滤波
	# gray = cv2.blur(gray, (3, 3))
	
	# 高斯滤波
	gray = cv2.GaussianBlur(gray,(3, 3),0)

	# 局部直方图
	calhe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
	gray = calhe.apply(gray)
	# gray = cv2.blur(gray, (3, 3))

	# 人脸检测
	faces = detect(gray, 0)
	for face in faces:
		# 人脸关键点预测
		face_68_point = predict(gray, face)
		face_68_point = face_utils.shape_to_np(face_68_point) # 转化为numpy数组

		# 关键点划分
		left_eye = face_68_point[l_eye_start:l_eye_end] # 左眼
		right_eye = face_68_point[r_eye_start:r_eye_end] # 右眼


		# 左右眼的ear系数计算
		left_EAR = eyeAspectRatio(left_eye)
		right_EAR = eyeAspectRatio(right_eye)
		ear = (left_EAR + right_EAR) / 2.0

		# 找人脸轮廓
		left = face.left()
		top = face.top()
		right = face.right()
		bottom = face.bottom()

		up_offset = int((bottom - top) * 0.2)
		down_offset = int((bottom - top) * 0.4)

		cv2.rectangle(frame, (left, top), (right, bottom - down_offset), (0, 255, 0), 3)

		# 判断睁闭眼
		if ear < thresh:
			cv2.putText(frame, "eye close", (left, top-15),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "eye open", (left, top-15),
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