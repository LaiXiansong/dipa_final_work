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
    return img1

def nothing(x):
    pass

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


cv2.namedWindow('res')

cv2.createTrackbar('max','res',0,255,nothing)
cv2.createTrackbar('min','res',0,255,nothing)

frame = cv2.imread("../resources/figure/1.png",0)
# frame = segment(frame, 40, 110)

# calhe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3,3))
# frame = calhe.apply(frame)
frame = cv2.medianBlur(frame, 3)

maxVal=200
minVal=100

while (1):
    
    if cv2.waitKey(20) & 0xFF==27:
        break
    maxVal = cv2.getTrackbarPos('max','res')
    minVal = cv2.getTrackbarPos('min','res')
    # edge = cv2.Canny(frame,minVal,maxVal)
    result = segment(frame, minVal, maxVal)
    # kernel = np.ones((5,5),np.uint8)  
    # edge = cv2.dilate(edge, kernel, iterations = 1)
    # edge = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)
    # edge = FillHole(edge)
    
    # cv2.imshow('res',frame)
    # cv2.imshow('res',edge)
    cv2.imshow('res', result)
cv2.destoryAllWindows()
