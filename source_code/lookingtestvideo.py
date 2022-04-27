from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy as np
import cv2
import tkinter as tk
import math
from tkinter import filedialog

def mouthAspectRatio(mouth):
	A = distance.euclidean(mouth[2], mouth[10])
	B = distance.euclidean(mouth[4], mouth[8])
	C = distance.euclidean(mouth[0], mouth[6])
	mar = (A + B) / (2.0 * C)
	return mar

def cos(array1, array2):
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
    norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
    return sum([array1[i]*array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)

# 判断张闭嘴的mar系数阈值
thresh = 0.54
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
# 外嘴对应的点
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

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
	if len(faces) == 0:
		cv2.putText(frame, "(Large) looking around", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
	else:
		for face in faces:
			# 人脸关键点预测
			face_68_point = predict(gray, face)
			face_68_point = face_utils.shape_to_np(face_68_point) # 转化为numpy数组

			image_points = np.array([
								face_68_point[33],
								face_68_point[8],
								face_68_point[36],
								face_68_point[45],
								face_68_point[48],
								face_68_point[54]
							], dtype="double")

			# 3D model points.
			model_points = np.array([
										(0.0, 0.0, 0.0),             # Nose tip
										(0.0, -330.0, -65.0),        # Chin
										(-225.0, 170.0, -135.0),     # Left eye left corner
										(225.0, 170.0, -135.0),      # Right eye right corne
										(-150.0, -150.0, -125.0),    # Left Mouth corner
										(150.0, -150.0, -125.0)      # Right mouth corner

									])

			# Camera internals

			focal_length = size[1]
			center = (size[1]/2, size[0]/2)
			camera_matrix = np.array(
									[[focal_length, 0, center[0]],
									[0, focal_length, center[1]],
									[0, 0, 1]], dtype = "double"
									)

			# print("Camera Matrix :\n {0}".format(camera_matrix)) 

			dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
			(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
			# (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)

			# print ("Rotation Vector:\n {0}".format(rotation_vector))
			# print ("Translation Vector:\n {0}".format(translation_vector))

			# Project a 3D point (0, 0, 1000.0) onto the image plane.
			# We use this to draw a line sticking out of the nose
			(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

			for p in image_points:
								cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

			p1 = ( int(image_points[0][0]), int(image_points[0][1]))
			p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
			pr1 = (531,593)
			pr2 = (876,655)
			v1 = (p1[0] - p2[0], p1[1] - p2[1] )
			v2 = (pr1[0] - pr2[0], pr1[1] - pr2[1] )
			print('cos=',cos(v1,v2))

			cv2.line(frame, p1, p2, (255,0,0), 2)
			cv2.line(frame, pr1, pr2, (255,255,0),2)

			if cos(v1,v2) < 0.85 :
				cv2.putText(frame, "(small) looking around", (10, 30),
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