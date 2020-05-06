import cv2
import numpy as np
from PIL import Image, ImageDraw


# widgets

class Common(object):
    def __init__(self):
        pass

    @staticmethod
    def inspect_size(img):
        # 检测尺寸是否符合要求： 30*30
        # img = cv2.imread(img, 1)
        # 将图片从二进制流转换为矩阵
        # img = np.array(img.convert("RGB"))
        from PIL import Image
        img = Image.open(img)
        print (img.size)
        if img.size == (500, 322):
            return True
        else:
            return False

    @staticmethod
    def detect_main_colors(original_image):
        '''
        探测图片前10个颜色
        :param original_image:
        :return:
        '''
        pal_unit_height = 50
        pal_unit_width = 100
        outline_width = 3
        original_image = Image.open(original_image)
        colors = Common.get_main_colors(original_image)  # array of colors in the image
        pal_list = []
        # draw pals
        for per, col in colors:
            pal = Image.new("RGB", (pal_unit_width, pal_unit_height))
            draw = ImageDraw.Draw(pal)
            draw.rectangle([0, 0, pal_unit_width, pal_unit_height], fill=col, width=outline_width,
                           outline="#000")
            pal_list.append((per, col, pal))
        del draw
        return pal_list

    @staticmethod
    def parse_comment(comment_dict):
        comment_map = {
            "img_size": "图片尺寸",
            "img_color": "图片颜色"
        }
        comment = ''
        for key, value in comment_dict.items():
            value = "符合要求" if value else "不符合要求"
            comment += comment_map[key] + " : " + value
        return comment

    @staticmethod
    def object_detect(img):
        '''
        SIFT目标检测
        :param img: 上传的图片，数据类型：BytesIO
        :return:
        '''
        # 内存中的字节数据 转换为opencv的图片矩阵
        img = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), 1)
        # SIFT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.xfeatures2d.SIFT_create()
        keypoints = detector.detect(gray, None)
        cv2.drawKeypoints(gray, keypoints, img)
        points2f = cv2.KeyPoint_convert(keypoints)  # 将KeyPoint格式数据中的xy坐标提取出来。
        # draw rectangle
        sorted_x = sorted([item[0] for item in points2f])
        sorted_y = sorted([item[1] for item in points2f])
        if sorted_x and sorted_y:
            min_x = sorted_x[0]
            max_x = sorted_x[-1]
            min_y = sorted_y[0]
            max_y = sorted_y[-1]
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 238, 0), 1, 4)
        return img

    @staticmethod
    def get_main_colors(original_image):
        '''
        :param image:
        :return: [(89.2%, (255,243,3)), (像素点百分比，rgb色值), ...]
        方法一：可以先压缩（需要，不然像素点太多了），后使用getcolors（）,获取图片中的所有像素
        '''
        # image = original_image.resize((80, 80))
        # colors = image.getcolors(80 * 80)  # array of colors in the image
        width, height = original_image.size
        transparent_pixel_no = 0
        colors = original_image.getcolors(width * height)
        # jpg通常为3通道，png为RGBA四通道+透明通道
        mode = original_image.mode
        if mode == "RGB":
            color_count_list = [color[0] for color in colors]
        elif mode == "RGBA":
            color_count_list = []
            for color in colors:
                if color[1][3] != 0:
                    color_count_list.append(color[0])
                else:
                    transparent_pixel_no += color[0]
        color_count_list.sort(reverse=True)
        top_counts = color_count_list[:10]
        top_colors = []
        for color in colors:
            if color[0] in top_counts:
                top_colors.append(color)
        # get top ten colors
        top_colors = [("%.2f%%" % (float(color[0]) * 100 / (width * height - transparent_pixel_no)), color[1]) for color
                      in Common.quick_sort(top_colors)[:10]]
        return top_colors

    @staticmethod
    def quick_sort(b):
        """快速逆序排序"""
        arr = b
        if len(b) < 2:
            return arr
        # 选取基准，随便选哪个都可以，选中间的便于理解
        mid = arr[len(b) // 2]
        # 定义基准值左右两个数列
        left, right = [], []
        # 从原始数组中移除基准值
        b.remove(mid)
        for item in b:
            # 大于基准值放右边
            if item[0] >= mid[0]:
                left.append(item)
            else:
                # 小于基准值放左边
                right.append(item)
            # 使用迭代进行比较
        return Common.quick_sort(left) + [mid] + Common.quick_sort(right)

    @staticmethod
    def compare_detect(img):
        # 载入标准图
        #img1 = cv2.imread('/Users/tezign/Documents/mikky/projects/百事/百事LOGO标准图/pepsi_en_s.png')
        img1 = cv2.imread('/data/User/lumeixi/pepsi_control_logo/pepsi_en_s.png')
        # 内存中的字节数据 转换为opencv的图片矩阵
        img2 = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), 1)

        kpimg1, kp1, des1 = Common.sift_kp(img1)
        kpimg2, kp2, des2 = Common.sift_kp(img2)

        goodMatch, cnt = Common.get_good_match(des1, des2, kp2)
        # 得到并画出最小外接矩形
        rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        box = np.int0(box)
        # 画出来
        cv2.drawContours(img2, [box], 0, (255, 238, 0), 2)
        # 将标准图和测试图放在同一张图片中进行对比
        all_goodmatch_img = cv2.drawMatches(img1, kp1, img2, kp2, goodMatch, None, flags=2)
        return all_goodmatch_img

    def sift_kp(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        kp_image = cv2.drawKeypoints(gray_image, kp, None)
        return kp_image, kp, des

    def get_good_match(des1, des2, kp2):
        bf = cv2.BFMatcher()
        good = []
        keypoints = []
        # 用knnmatch 返回和原图
        matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
        # DMatch
        # queryIdx：第一个图的特征点描述符的下标序号（第几个特征点描述符）
        # trainIdx：第二个图的特征点描述符的下标
        # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
        # 按照第一个元素的distance和第二个元素的distance比值进行排序，和标准图匹配的两个特征点，距离不能太远
        matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                # m为匹配的特征点，trainIdx为测试图特征描述点的index，kp2为测试图的特征点集，kp2[trainIdx]为测试图上的特征点，pt为获取坐标，返回值为tuple类型
                # queryIdx为标准图特征描述点的index，kp1位标准图的特征点集，kp1[queryIdx]为标准图上的特征点
                p = kp2[m.trainIdx].pt
                p = np.array(p, dtype=float)
                keypoints.append(p)
                good.append(m)
        keypoints = np.array(keypoints, dtype=np.float32)
        return good, keypoints

