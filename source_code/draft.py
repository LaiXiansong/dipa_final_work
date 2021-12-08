import numpy as np
import cv2
import time
from scipy import signal

def createLoGKernel(sigma, size):
    H, W = size
    r, c = np.mgrid[0:H:1.0, 0:W:1.0]
    r -= (H-1)/2
    c -= (W-1)/2
    sigma2 = np.power(sigma, 2.0)
    norm2 = np.power(r, 2.0) + np.power(c, 2.0)
    LoGKernel = (norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))  # 省略掉了常数系数 1\2πσ4

    print(LoGKernel)
    return LoGKernel

def LoG(image, sigma, size, _boundary='symm'):
    LoGKernel = createLoGKernel(sigma, size)
    edge = signal.convolve2d(image, LoGKernel, 'same', boundary=_boundary)
    return edge

#直接采用二维高斯卷积核，进行卷积
def gaussConv2(image, size, sigma):
    H, W = size
    r, c = np.mgrid[0:H:1.0, 0:W:1.0]
    c -= (W - 1.0) / 2.0
    r -= (H - 1.0) / 2.0
    sigma2 = np.power(sigma, 2.0)
    norm2 = np.power(r, 2.0) + np.power(c, 2.0)
    LoGKernel = (1 / (2*np.pi*sigma2)) * np.exp(-norm2 / (2 * sigma2))
    image_conv = signal.convolve2d(image, LoGKernel, 'same','symm')

    return image_conv

def DoG(image, size, sigma, k=1.1):

    Is = gaussConv2(image, size, sigma)
    Isk = gaussConv2(image, size, sigma * k)

    doG = Isk - Is
    doG /= (np.power(sigma, 2.0)*(k-1))
    return doG


def motionDiff(current_frame, diff):
    out = abs(np.sign(current_frame * diff))*255
    return out

##马赛克
def do_mosaic(frame, x, y, w, h, neighbor=9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)


def segment(img, low, high):
    img1 = img.copy()
    img1[img1>high] = 0
    img1[img1<low] = 0
    img1[img1>low] = 1
    return img1

def segment1(img, low, high):
    img1 = img.copy()
    img1[img1>high] = 0
    img1[img1<low] = 0
    return img1

def FillHole(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(img, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
 
    out = sum(contour_list)
    return out


cap = cv2.VideoCapture("../resources/calling/Phoning_09.avi")
start_time = time.time()
counter = 0 
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率

ret, last_frame = cap.read()
last_frame = cv2.cvtColor(last_frame,cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, Mask = cap.read() # ret 返回是否读到图片
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

        gray = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
        # gray = gray[200:520, 0:640]
        frame = gray.copy()
        # frame = cv2.blur(frame, (7, 7))
        # frame = cv2.medianBlur(frame, 3)
        # xgrad = cv2.Sobel(frame, cv2.CV_16SC1, 1, 0)  # xGrodient
        # ygrad = cv2.Sobel(frame, cv2.CV_16SC1, 0, 1)  # yGrodient
        # frame = cv2.Canny(xgrad, ygrad, 100, 150)  # edge 

        # image_gray_blur1 = cv2.GaussianBlur(frame, (3, 3), 0.7)
        # image_gray_blur2 = cv2.GaussianBlur(frame, (3, 3), 0.8)
        # image_gray_dog = image_gray_blur2 - image_gray_blur1

        # sigma = 0.9
        # k = 1.1
        # size = (5, 5)
        # DoG_edge = DoG(frame, size, sigma, k)
        # DoG_edge[DoG_edge>0] = 255
        # DoG_edge[DoG_edge<0] = 0
        # # DoG_edge = DoG_edge / np.max(DoG_edge)
        # # DoG_edge = DoG_edge * 255
        # out_image = DoG_edge.astype(np.uint8)

        # LoG_edge = LoG(frame, 0.4, (5, 5))
        # LoG_edge[LoG_edge>0] = 255
        # # LoG_edge[LoG_edge>255] = 0
        # LoG_edge[LoG_edge<0] = 0
        # out_image = LoG_edge.astype(np.uint8)
        
        # out_image = motionDiff(frame, frame - last_frame)
        # last_frame = frame.copy()
        
        # frame = cv2.blur(frame, (1, 1))

        # frame = cv2.medianBlur(frame, 13)
        # calhe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        # frame = calhe.apply(frame)
        # frame = segment(frame, 0, 100)
        # edge = cv2.Canny(frame,53, 93)
        
        # kernel = np.ones((2, 2),np.uint8)   
        # frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel,iterations=1)

        # out_image = frame.copy()
        # do_mosaic(out_image, 0, 0, 900, 700, 10)
        # out_image = cv2.Canny(out_image,30, 100)
        # edge = FillHole(edge)

        # frame = cv2.medianBlur(frame, 5)
        # frame = segment1(frame, 40, 140)

        # kernel = np.array([[5, 5, 5], [0, 0, 0], [-5, -5, -5]], np.float32)
        # frame = cv2.filter2D(frame, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        
        # frame = segment (frame, 60, 255)

        # frame = cv2.medianBlur(frame, 7)


        # kernel = np.ones((3, 3),np.uint8)   
        # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel,iterations=2)

        # kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)  
        # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel,iterations=3)

        # kernel = np.ones((3, 3),np.uint8)  
        # Mask = cv2.morphologyEx(Mask,cv2.MORPH_OPEN,kernel,iterations=1)
        # --------------------------------------------------------------
        # kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], np.float32)
        # frame = cv2.filter2D(frame, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)

        # kernel = np.ones((3, 3),np.uint8)   
        # frame = cv2.erode(frame, kernel, iterations = 1)

        # kernel = np.ones((5, 5),np.uint8)   
        # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel,iterations=1)

        # kernel = np.ones((3, 3),np.uint8)   
        # frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel,iterations=1)

        # kernel = np.array([[5, 5, 5], [0, 0, 0], [-5, -5, -5]], np.float32)
        # frame = cv2.filter2D(frame, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        
        # kernel = np.ones((5, 5),np.uint8)   
        # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel,iterations=3)
        # -----------------------------------------------------------------------
        # kernel = np.array((
        #         [1, 1, 1, 1]), dtype="int")
        # # kernel = np.array(( 
        # #         [1, 1, 1, 1, 1, 1, 1, 1],
        # #         [0, 0, 1, 1, 1, 1, 0, 0]), dtype="int")
        # frame = cv2.morphologyEx(frame, cv2.MORPH_HITMISS, kernel)

        # result2 = Mask * gray
        # result2 = segment1(result2, 145, 150)
        # ------------------------------------------------------ 
        # 找到眼眶的范围
        # 大范围中值滤波+阈值分割将人从背景中分割出来
        # gray = 源图；frame = 源图的一份复制
        # roi 区域
        # 局部直方图增强
        # calhe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9, 9))
        # frame = calhe.apply(frame)
        # 中值滤波
        frame = cv2.medianBlur(frame, 9)
        # 水平边缘
        kernel = np.array([[5, 5, 5], [0, 0, 0], [-5, -5, -5]], np.float32)
        # kernel = 5 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
        frame = cv2.filter2D(frame, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        # 二值阈值分割
        frame = segment(frame, 60, 255) * 255
        # # 膨胀
        # kernel = np.ones((3, 3),np.uint8)
        # # kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], np.uint8)  
        # frame = cv2.dilate(frame, kernel, iterations = 10)
        # # 闭操作
        # kernel = np.ones((5, 5),np.uint8)   
        # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel,iterations=3)
        # frame = segment(frame, 0, 255)
        # Mask = frame
        # result = gray * Mask

        cv2.imshow('frame', frame)
        # cv2.imshow('frame', edge)

        # print("FPS: ", counter / (time.time() - start_time))

        counter = 0
        start_time = time.time()
    time.sleep(1/fps)#按原帧率播放

cap.release()
cv2.destropAllWindows()