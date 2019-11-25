# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage import img_as_ubyte
from skimage import img_as_float
from skimage import filters
from skimage.measure import compare_ssim

# from shapedetector import ShapeDetector
# from colorlabeler import ColorLabeler

import imutils
import cv2

def skimage2opencv(img):
    return img_as_ubyte(img)
def opencv2skimage(img):
    return img_as_float(img)

# 高反差保留算法
def highpass(file_name):
    # 读取方法skimage float类型
    img=io.imread(file_name)

    img = img * 1.5
    # 高斯
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    img_out = img - gauss_out + 128.0

    img_out = img_out/255.0

    # 饱和处理
    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2

    # plt.figure()
    # plt.imshow(img/255.0)
    # plt.axis('off')
    # img_out = filters.roberts(img_out)
    plt.axis('off')
    img_out = skimage2opencv(img_out)
    # cv2.imshow('img_out', img_out)
    # print(img_out)
    # img_out = fill_color_demo(img_out)
    return img_out

def ColorSelect(image):
    # 读取图片
    # image = cv2.imread(args["image"])
    # 进行裁剪操作
    resized = imutils.resize(image, width=500)

    # ratio = image.shape[0] / float(resized.shape[0])

    # 进行高斯模糊操作
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    # 进行图片灰度化
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 进行颜色空间的变换
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    # 进行阈值分割
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Thresh", thresh)

    return thresh

def fill_color_demo(img):
    copyImg = img.copy()
    h,w = copyImg.shape[:2]
    # 记住：遮罩mask 要在img的h w 之上加2，基于opencv扫描算法
    # mask必须行和列都加2，且必须为uint8单通道阵列---当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    mask = np.zeros([h+2,w+2],np.uint8)
    # floodFill(image, mask, seedPoint, newVal, flags=None) #种子点(200,500)、填充颜色(0,255,255)、填充区域最低(170,320,100) 最高范围(360,580,30)
    cv2.floodFill(copyImg,mask,(5,25),(0,0,0),(180,180,180),(50,50,50),cv2.FLOODFILL_FIXED_RANGE)
    # cv2.imshow('fill_color_demo',copyImg)
    return copyImg

def DeGrayscaler(image):
    height, width, channels = image.shape
    print("width:%s,height:%s,channels:%s" % (width, height, channels))
    if channels == 3:
        for row in range(height):
            for list in range(width): # BGR
                B = image[row, list, 0]
                G = image[row, list, 1]
                R = image[row, list, 2]
                if R<=165 and G<165 and B<180:
                    if R>40 and G>40 and B>70:
                        image[row, list] = [0,0,0]
                else:
                    image[row, list] = [255, 255, 255]


    return image


def ComputeMinLevel(hist, pnum):
    index = np.add.accumulate(hist)
    return np.argwhere(index>pnum * 20.3 * 0.01)[0][0]


def ComputeMaxLevel(hist, pnum):
    hist_0 = hist[::-1]
    Iter_sum = np.add.accumulate(hist_0)
    index = np.argwhere(Iter_sum > (pnum * 2.2 * 0.01))[0][0]
    return 255-index



def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        index = np.array(list(range(256)))
        screenNum = np.where(index<minlevel,0,index)
        screenNum = np.where(screenNum> maxlevel,255,screenNum)
        for i in range(len(screenNum)):
            if screenNum[i]> 0 and screenNum[i] < 255:
                screenNum[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return screenNum


def CreateNewImg(img):
    h, w, d = img.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imghist = np.bincount(img[:, :, i].reshape(1, -1)[0])
        minlevel = ComputeMinLevel(imghist,  h * w)
        maxlevel = ComputeMaxLevel(imghist, h * w)
        screenNum = LinearMap(minlevel, maxlevel)
        if (screenNum.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = screenNum[img[j, :, i]]
    return newimg

# 反色处理
def access_pixels(image):
    height, width, channels = image.shape
    print("width:%s,height:%s,channels:%s" % (width, height, channels))

    for row in range(height):
        for list in range(width):
            for c in range(channels):
                pv = image[row, list, c]
                image[row, list, c] = 255*2 - pv
    return image

# 颜色还原
def reColor(img, gray):
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    g = gray[:]
    p = 0.2989
    q = 0.5870
    t = 0.1140
    B_new = (g-p*R-q*G)/t
    B_new = np.uint8(B_new)
    img_new = np.zeros((img.shape)).astype("uint8")
    height, width, channels = img.shape
    for row in range(height):
            for line in range(width): # BGR
                if gray[row,line] == 0:
                    img_new[row,line,0] = B_new[row,line]
                    img_new[row,line,1] = G[row,line]
                    img_new[row,line,2] = R[row,line]
                else:
                    img_new[row,line,0] = 0
                    img_new[row,line,1] = 0
                    img_new[row,line,2] = 0
    return img_new

def preprocess(gray):
    resized = imutils.resize(gray, width=600)
    # cv2.imshow('resize', resized)
    # 进行高斯模糊操作
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    # cv2.imshow('blurred', blurred)
    # 进行图片灰度化
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # 进行颜色空间的变换
    # lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    # 进行阈值分割
    thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('result',thresh)
    #
    # for i in range(10):
    #     gray = cv2.fastNlMeansDenoising(gray)
    # cv2.imshow("12", gray)
    # # # 直方图均衡化
    # equ = cv2.equalizeHist(gray)
    #
    #
    #
    # cv2.imshow("3", equ)
    # # 高斯平滑
    # gaussian = cv2.GaussianBlur(equ, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # cv2.imshow("4", gaussian)

    # 中值滤波
    # median = cv2.medianBlur(thresh, 5)
    # Sobel算子，X方向求梯度
    # sobel = cv2.Sobel(thresh, cv2.CV_8U, 0, 1, ksize = 5)

    # 二值化
    # ret, binary = cv2.threshold(sobel, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow("2", binary)
    # # 膨胀和腐蚀操作的核函数
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    # # 膨胀一次，让轮廓突出
    # dilation = cv2.dilate(binary, element2, iterations = 1)
    # # 腐蚀一次，去掉细节
    # erosion = cv2.erode(dilation, element1, iterations = 1)
    # 再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(erosion, element2,iterations = 3)
    # cv2.imshow('erosion',sobel)
    # cv2.waitKey(0)
    return thresh


def detect(img):
    # 转化成灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1",gray)
    #
    # 形态学变换的预处理
    dilation = preprocess(img)


    img_org2 = dilation.copy()
    img_plate = img_org2
    # cv2.imshow('number plate', img_plate)
    # cv2.imwrite('number_plate.jpg', img_plate)
    #
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    #
    # # 带轮廓的图片
    # cv2.imwrite('contours.png', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def colorRegion(image):
    # 定义边界列表 （lower[r, g, b], upper[r, g, b]）
    boundaries = [([60, 140, 50], [180, 240, 160])]
    img_mask = []
    # 循环遍历所有的边界
    for (lower, upper) in boundaries:
        # 创建上边界和下边界
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        # 进行与操作
        output = cv2.bitwise_and(image, image, mask=mask)
        img_mask.append(output)
        # 显示结果
    # cv2.waitKey(0)
    return output

# 通过得到RGB每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    # print("相似度:", sub_data)
    return sub_data

# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def imgdiff(imageA,imageB):
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))



if __name__ == '__main__':
    start_img,end_img = input("请输入验证码图序号开始到结束：（start end）").split()
    start_img_test, end_img_test = input("请输入验证码测试图序号开始到结束：（start end）").split()

    for i in range(int(start_img),int(end_img)+1):
        classify_max = 0
        imgRight = None
        imagePath = './images/%i.jpeg'% i
        imgS = cv2.imread(imagePath)
        imgS = imutils.resize(imgS,width=600)
        img = highpass(imagePath)
        # cv2.imshow('highpass', img)
        thresh = preprocess(img)
        color_img = reColor(imgS,thresh)
        # cv2.imshow("color",color_img)
        for j in range(int(start_img_test), int(end_img_test) + 1):
            test_imagePath = "./images/%i.jpeg" % j
            test_imgS = cv2.imread(test_imagePath)
            test_imgS = imutils.resize(test_imgS,width=600)
            test_img = highpass(test_imagePath)
            # cv2.imshow('test_highpass', test_img)
            test_thresh = preprocess(test_img)
            test_color_img = reColor(test_imgS,test_thresh)
            # cv2.imshow("test_color",test_color_img)

            # 用三通道直方图 对比图片 取相似度（取出每一通道进行直方图对比）
            temp = classify_hist_with_split(color_img, test_color_img)

            if classify_max < temp:
                classify_max = temp
                imgRight = j

        print("测试图 %s.jpg 的正确图片是 %s.jpg" % (i, imgRight))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

