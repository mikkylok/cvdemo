import random

import cv2
import numpy as np


class FontAnalysis():
    def __init__(self):
        pass

    @staticmethod
    def moment(img):
        """
        计算重心

        :param img:
        :return: 重心坐标，画图
        """
        cv_image = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), 1)
        h, w, _ = cv_image.shape
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        _,contours, _ = cv2.findContours(thresh, 0, 1)

        mu = [None] * len(contours)
        rect_area = [None] * len(contours)

        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])

            rect = cv2.boundingRect(contours[i])
            rect_area[i] = rect[2] * rect[3]

        bigindex = np.argmax(rect_area)
        # Get the mass centers
        mc = [None] * len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.drawContours(drawing, contours, i, color, 2)
            cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 3, color, -1)
            mc[i] = (np.round(mc[i][0] / w, 4), np.round(mc[i][1] / h, 4))

        return drawing, mc

    @staticmethod
    def weight(img):
        """
        视觉重量:
        H-用角度度量，取值范围为0°～360°——>[0,180)
        S-取值范围为0.0～1.0——>[0,255)
        V-取值范围为0.0(黑色)～1.0(白色)——>[0,255)
        :param img:上传的图片，数据类型：BytesIO
        :return:
        """
        cv_image = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), 1)
        h, w, _ = cv_image.shape
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])

        mask = cv2.inRange(hsv, lower, upper)
        return mask, np.sum(mask / 255) / w / h

# image = cv2.imread('/Users/snowholy/Desktop/cvdemo/timg.jpeg')
# ff = FontAnalysis()
# ff.moment(image)
