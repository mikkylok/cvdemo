import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import io
import cv2
from PIL import Image
from models.common import Common
from models.font import FontAnalysis
from models.detect import EasyDetect

PEPSI_QUERY_PICS = {
    "百事：英文/横": "/Users/tezign/Documents/mikky/projects/百事/百事LOGO标准图/pepsi_en_h.png",
    "百事：英文/竖": "/Users/tezign/Documents/mikky/projects/百事/百事LOGO标准图/pepsi_en_s.png",
    "百事：中文/横": "/Users/tezign/Documents/mikky/projects/百事/百事LOGO标准图/pepsi_cn_h.png",
    "百事：中文/竖": "/Users/tezign/Documents/mikky/projects/百事/百事LOGO标准图/pepsi_cn_s.png",
    "Puma": "/Users/tezign/Documents/puma1.jpg",
    "联合力华": "/Users/tezign/Documents/mikky/联合丽华sta.png",
}

# view
class App(object):
    def __init__(self):
        self.comment = {}

    def logo_inspect(self, option):
        st.title(option)
        # upload image
        st.text("客户要求：\n1. 图片标准尺寸：500 * 322")
        upload_img = st.file_uploader("Choose an image file(only jpg/png/jpeg are supported)", type=["png", "jpg", "jpeg"])
        if upload_img is not None:
            st.image(upload_img, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                     format='JPEG')
            self.comment["img_size"] = Common.inspect_size(upload_img)
            st.text("意见：%s\n" % Common.parse_comment(self.comment))

    def color_detect(self, option):
        '''
        展示前10个色彩和对应rgb值
        :param self:
        :param option: 选项
        :return:
        '''
        st.title(option)
        # upload image
        upload_img = st.file_uploader("Choose an image file(only jpg and png are supported)", type=["png", "jpg"])
        if upload_img is not None:
            # show original image
            st.text("Original image:")
            st.image(upload_img, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                     format='JPEG')
            pal_list = Common.detect_main_colors(upload_img)
            # show pallette
            st.text("Main colors:")
            for i in range(len(pal_list)):
                st.text(str(i + 1) + ". 百分比：" + str(pal_list[i][0]) + " RGB色值： " + str(pal_list[i][1]))
                st.image(pal_list[i][2], caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                         format='JPEG')

    def compare_detect(self, option):
        '''
        目标检测：sift + flann + ransac 传统cv方法进行特征检测
        :param option:
        :return:
        '''
        st.title(option)
        st.subheader("Logo对比方式：")
        img_option = st.radio('', ("已知logo", "自行上传"),  index=1)
        if img_option == "已知logo":
            st.subheader("已知Logo：")
            i = 1
            for key, value in PEPSI_QUERY_PICS.items():
                st.text(str(i) + '. ' + key + ":")
                with open(value, 'rb') as f:
                    img = f.read()
                st.image(img,
                         caption=None,
                         width=100,
                         use_column_width=False,
                         clamp=False,
                         channels='BGR',
                         format='PNG')
                i = i + 1
            st.subheader("请选择logo：")
            query_img_option = st.radio('', tuple(PEPSI_QUERY_PICS.keys()))
            with open(PEPSI_QUERY_PICS[query_img_option], 'rb') as f:
                logo_img = io.BytesIO(f.read())
        else:
            st.subheader("请选择logo：")
            logo_img = st.file_uploader("", type=["png", "jpg"], key=option+"_logo")
        if logo_img is not None:
            st.image(logo_img,
                     caption=None,
                     width=300,
                     use_column_width=False,
                     clamp=False,
                     channels='BGR',
                     format='PNG')
        # upload image
        st.subheader("请选择设计稿：")
        upload_img = st.file_uploader("", type=["png", "jpg"], key=option+"_training_img")
        if upload_img and logo_img:
            easy_detect = EasyDetect(logo_img, upload_img)
            result_img = easy_detect.compare_two_pics_sift()
            if result_img is not None:
                st.image(result_img, caption=None, width=None, use_column_width=True, clamp=False, channels='BGR',
                         format='JPEG')
            else:
                st.text("不匹配！")

    def font_moment(self,option):
        st.title(option)
        upload_img = st.file_uploader("Choose an image file(only jpg and png are supported)", type=["png", "jpg","jpeg"])
        if upload_img is not None:
            # show original image
            st.text("Original image:")
            st.image(upload_img, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                     format='JPEG')
            bound_img,center = FontAnalysis.moment(upload_img)

            st.text("轮廓图")
            st.image(bound_img, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                     format='JPEG')
            for i in range(len(center)):
                st.text("重心位置" + str(i) + ":" + str(center[i]))

    def font_weight(self,option):
        st.title(option)
        # upload image
        upload_img = st.file_uploader("Choose an image file(only jpg and png are supported)", type=["png", "jpg","jpeg"])
        if upload_img is not None:

            st.text("Original image:")
            st.image(upload_img, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                     format='JPEG')
            mask,weight_ratio = FontAnalysis.weight(upload_img)

            st.text("百分比:"+str(np.round(weight_ratio,4)))
            st.image(mask, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                     format='JPEG')



if __name__ == "__main__":
    FUNC_MAP = {
        "Logo合规性检测": 0,
        "主色调颜色检测": 1,
        "SIFT对比特征检测": 2,
        "字体重心计算": 3,
        "字体视觉重量分析": 4,
    }
    app = App()
    option = st.sidebar.radio('', tuple(FUNC_MAP.keys()))
    # view，可以新增demo应用
    if FUNC_MAP[option] == 0:
        app.logo_inspect(option)
    elif FUNC_MAP[option] == 1:
        app.color_detect(option)
    elif FUNC_MAP[option] == 2:
        app.compare_detect(option)
    elif FUNC_MAP[option] == 3:
        app.font_moment(option)
    elif FUNC_MAP[option] == 4:
        app.font_weight(option)
