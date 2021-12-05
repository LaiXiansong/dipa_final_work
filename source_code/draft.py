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
    # fh, fw = img.shape[0], img.shape[1]
    # for i in range(fh):
    #     for j in range(fw):
    #         if low <= img[i, j] <= high:
    #             img[i, j] = 255
    #         else:
    #             img[i, j] = 0
    img[img>high] = 0
    img[img<low] = 0
    return img

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


cap = cv2.VideoCapture("../resources/eye/eye_closed_02.avi")
start_time = time.time()
counter = 0 
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率

ret, last_frame = cap.read()
last_frame = cv2.cvtColor(last_frame,cv2.COLOR_BGR2GRAY)

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
        # frame = cv2.Canny(frame,30, 100)

        # ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)

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
        frame = segment(frame, 56, 106)
        # edge = cv2.Canny(frame,53, 93)
        
        # kernel = np.ones((5,5),np.uint8)   
        # edge = cv2.dilate(edge, kernel, iterations = 1)
        # edge = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)

        # out_image = frame.copy()
        # do_mosaic(out_image, 0, 0, 900, 700, 10)
        # out_image = cv2.Canny(out_image,30, 100)
        # edge = FillHole(edge)
        # ------------------------------------------------------ 
        cv2.imshow('frame', frame)
        # cv2.imshow('frame', edge)

        # print("FPS: ", counter / (time.time() - start_time))

        counter = 0
        start_time = time.time()
    time.sleep(1/fps)#按原帧率播放

cap.release()
cv2.destropAllWindows()