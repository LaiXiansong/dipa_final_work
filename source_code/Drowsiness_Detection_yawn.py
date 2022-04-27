from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def mouthAspectRatio(mouth):
	A = distance.euclidean(mouth[2], mouth[10])
	B = distance.euclidean(mouth[4], mouth[8])
	C = distance.euclidean(mouth[0], mouth[6])
	mar = (A + B) / (2.0 * C)
	return mar
# 判断张闭嘴的mar系数阈值
thresh = 0.54
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
# 外嘴对应的点
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

file_path = ''
while file_path =='':
	file_path = filedialog.askopenfilenames(title='请选择一个视频',initialdir='resources/mouth/')
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
out = cv2.VideoWriter('../results/mouth/yawn_res_'+choose_file[-6:]+'',fourcc, fps, size)

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
	for face in faces:
		# 人脸关键点预测
		face_68_point = predict(gray, face)
		face_68_point = face_utils.shape_to_np(face_68_point) # 转化为numpy数组

		# 关键点划分
		mouth = face_68_point[mouth_start:mouth_end] # 左眼

		# 左右眼的ear系数计算
		mar = mouthAspectRatio(mouth)

		# 找嘴巴轮廓
		left = mouth[0, 0]
		right = mouth[6, 0]
		top = mouth[3, 1]
		bottom = mouth[9, 1]
		# 框出嘴巴的矩形
		cv2.rectangle(frame, (left-20, top-40), (right+20, bottom + 20), (0, 255, 0), 3)

		# 判断张嘴
		if mar < thresh:
			cv2.putText(frame, "mouth close", (left, top-15),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "mouth open", (left, top-15),
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