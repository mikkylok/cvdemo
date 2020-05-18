import cv2
import numpy as np


class EasyDetect(object):
    '''
    目标检测: 使用传统cv方法
    '''

    def __init__(self, que_img, tra_img):
        # 标准图
        self.que_img = cv2.imdecode(np.frombuffer(que_img.getvalue(), np.uint8), 1)
        # 对比图
        self.tra_img = cv2.imdecode(np.frombuffer(tra_img.getvalue(), np.uint8), 1)

    @staticmethod
    def sift(img):
        '''
        调整参数：
            SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]) -> retval
            1.nfeatures：特征点数目
            2.nOctaveLayers：金字塔中每组的层数 -> 影响图像高斯金字塔的构成
            3.contrastThreshold: 过滤掉较差的特征点的对阈值，越大，检测器检测到的特征越少。->影响在DOG中寻找极值点的过程与结果
            4.edgeThreshold：过滤掉边缘效应的阈值，越大，检测点越多 -> 影响在DOG中寻找极值点的过程与结果
            5.sigma：金字塔第0层图像高斯滤波系数，越小 -> 影响图像高斯金字塔的构成
        标准：
            nfeatures = 0,
            nOctaveLayers = 3,
            contrastThreshold = 0.04,
            edgeThreshold = 10,
            sigma = 1.6
        '''
        #print (img)
        #print (type(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=100, sigma=1.6)
        # kp: 关键点；des：sift特征向量，128维
        kp, des = sift.detectAndCompute(gray, None)
        # img = cv2.drawKeypoints(img, kp, img)
        return kp, des

    def brute_match(self, kp1, des1, kp2, des2):
        '''
            alg: 目标检测算法：SIFT, SURF, ORB, BRIEF, BRISK
        cv2.BFMatcher(normType, crossCheck):
            @param: normType，默认cv.NORM_L2
                1) SIFT, SURF: cv.NORM_L2, cv.NORM_L1
                2) ORB，BRIEF，BRISK: cv.NORM_HAMMING
                3) 如果ORB使用WTA_K == 3或4: cv.NORM_HAMMING2
            @param: crossCheck，默认情况下为false：
                如果为true，则Matcher仅返回具有值(i，j)的那些匹配项，以使集合A中的第i个描述符具有集合B中的第j个描述符为最佳匹配，反之亦然
        调整参数：
            1.normType：不同检测算法，不一样
            2.crossCheck: True/False
        '''
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(des1, des2)
        # 根据距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        # 绘制前100的匹配项
        img = cv2.drawMatches(self.que_img, kp1, self.tra_img, kp2, matches[:100], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img

    def knn_match(self, kp1, des1, kp2, des2):
        '''
        knn聚类算法
        调整参数：
            1.knn_ratio
                标准：0.75
        '''
        bf = cv2.BFMatcher()
        # 获取2个与关键点匹配的2个点
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            # 这个distance分别是哪里到哪里的距离？
            if m.distance < 0.52 * n.distance:
                good.append([m])
        # 连接匹配线
        img = cv2.drawMatchesKnn(self.que_img, kp1, self.tra_img, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img

    def flann_match(self, kp1, des1, kp2, des2):
        '''
        大型数据集，速度比bf快
        调整参数：
            1.index_params:
                根据sift 或者 orb不同
                    sift:
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                    orb:
                        FLANN_INDEX_LSH = 6
                        index_params= dict(algorithm = FLANN_INDEX_LSH,
                                           table_number = 6, # 12
                                           key_size = 12,     # 20
                                           multi_probe_level = 1) #2
            2.search_params:
                索引中树应递归便利的次数，n越大精度越高，但也需要更多时间
                dict(checks=n)
                标准50
            3.flann_ratio
                标准:0.7
        '''
        # Sift参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # 指定索引中的树应递归遍历的次数。较高的值可提供更好的精度，但也需要更多时间。
        search_params = dict(checks=100)  # 或传递一个空字典
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # 只需要绘制好匹配项，因此创建一个掩码
        matchesMask = [[0, 0] for i in range(len(matches))]
        # 根据Lowe的论文进行比例测试
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.5875 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img = cv2.drawMatchesKnn(self.que_img, kp1, self.tra_img, kp2, matches, None, **draw_params)
        return img

    def homography_match(self, kp1, des1, kp2, des2):
        '''
        原因：
            1）之前的匹配算法是在二维空间进行匹配，现在可以通过透视变换，在三维空间进行匹配
            2）二维匹配时可能会出现一些可能影响结果的坏点
        步骤：
            1）sift分别在两张图中找到关键角点
            2）匹配算法找到两张图中特征匹配的点对
            3）通过两图的特征集，找到img1->img2的变换矩阵
            4）img1和变换矩阵做变换，得到在img2中的变换位置，并在img2中框出
        优点：
            1）考虑到了坏点去除，透视变换
        缺点：
            1）透视变换不包括曲面
            2）对于百事这样的圆形图片，特征不明显，可能在一开始匹配的时候就出现错误影响最后的结果
        '''
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        MIN_MATCH_COUNT = 10
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        # 至少有10个匹配项（由MIN_MATCH_COUNT定义）可以找到对象。否则，只需显示一条消息，说明没有足够的匹配项。
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 输入：标准图的好的关键点， 测试图的好的关键点, 5.0是误差值，一般到1-10之间，超过误差就认为是异常值
            # 随机抽取4个点，求得最合适的M变化矩阵
            # 返回值：M为变换矩阵；mask是掩模，在线的点
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
            h, w, d = self.que_img.shape
            # 标准图上确定的4个点
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # 标准图通过变换得到的在测试图中的位置
            dst = np.int32(cv2.perspectiveTransform(pts, M))
            return dst
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            return None
        '''
        draw_params = dict(matchColor = (0,255,0), # 用绿色绘制匹配
                           singlePointColor = None,
                           matchesMask = matchesMask, # 只绘制内部点
                           flags = 2)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        return dst
        '''

    def compare_two_pics_sift(self):
        kp1, des1 = EasyDetect.sift(self.que_img)
        kp2, des2 = EasyDetect.sift(self.tra_img)
        # img = self.brute_match(kp1, des1, kp2, des2)
        # img = self.knn_match(kp1, des1, kp2, des2)
        # img = self.flann_match(kp1, des1, kp2, des2)
        dst = self.homography_match(kp1, des1, kp2, des2)
        if dst is not None:
            img = self.draw_rectangle(dst)
            return img
        else:
            return None

    def draw_rectangle(self, relation):
        """
        openCV提供rectangle 画框 框出长方形
        :param relation: 图片对比的关系
        :param image: 检测图片
        :param path: 输出结果图片path
        """
        xy = self.coordinate(relation)
        pt1, pt2 = xy[0], xy[1]
        result_img = cv2.rectangle(self.tra_img, pt1, pt2, (0, 0, 255), thickness=3)
        return result_img

    def coordinate(self, key):
        """
        寻找最大的坐标和最小的坐标
        :param key: numpy.array的对象
        :return: object 元组
        """
        coordinateX = []
        coordinateY = []
        for coordinate in key.tolist():
            X = tuple(coordinate[0])[0]
            Y = tuple(coordinate[0])[1]
            coordinateX.append(X)
            coordinateY.append(Y)
        minXY = (min(coordinateX), min(coordinateY))
        maxXY = (max(coordinateX), max(coordinateY))
        return minXY, maxXY


if __name__ == '__main__':
    # example = Detect("/Users/tezign/Documents/mikky/projects/百事/百事LOGO标准图/pepsi_en_s.png",
    #                "/Users/tezign/Documents/mikky/projects/百事/pepsi测试图/en_s/en_s_2.jpg")
    # example = Detect("/Users/tezign/Documents/mikky/联合丽华sta.png", "/Users/tezign/Documents/mikky/联合丽华.png")
    example = Detect("/Users/tezign/Downloads/puma1.jpg", "/Users/tezign/Downloads/puma2.jpg")
    # example = Example("/Users/tezign/Downloads/mcdonald.png")
    # example = Example("/Users/tezign/Documents/sanjiao.jpg")
    img = example.compare_two_pics_sift()
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 700, 800)
    cv2.imshow("img", img)
    # img2 = example.compare_two_pics_sift()
    # cv2.imshow("img2", img2)
    # img = example.draw_contours(cnts)
    # print(cnts)
    # img = example.get_complex_outer_rectangle(cnts)
    # print (cnt)
    # print (hie)
    # a = example.get_contours_most_points(cnt[1])
    # print (a)
    # img = example.get_min_circle(cnt[14])
    # get_min_rectangle
    # get_min_circle
    # img = example.zoom_out_pic(600, 50)
    # img = example.move(100, 50, 300, 300)
    # img = example.perspective_transform()
    # img = example.high_filter()
    # contours = example.get_contours()
    # img= example.draw_contours(contours)
    # contours = example.get_contours_complex()
    # img= example.draw_contours(contours)

    # img1 = cv2.imread("/Users/tezign/Downloads/rainbow.jpg")
    # img2 = cv2.imread("/Users/tezign/Documents/opencv.jpg")
    # mg2_small = cv2.resize(img2, (100, 100), interpolation=cv2.INTER_NEAREST)
    # img = Func.stick_two_pics(img1, img2_small)

    # img1 = example.img
    # img2 = example.convert_color_model("RGB")
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.imwrite("/Users/tezign/Documents/mikky/联合丽华comp.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 最好有个工具，可以根据图片的颜色，去选择展示opencv中的rgb和hsv：自己写一个放在机器学习平台上

