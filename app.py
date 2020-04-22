import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
from models.common import Common

# view
class App(object):
  def __init__(self):
      self.comment = {}

  def logo_detect(self, option):
      st.title(option)
      # upload image
      upload_img = st.file_uploader("Choose an image file", type="jpg")
      if upload_img is not None:
          st.image(upload_img, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB',
                   format='JPEG')
          # running进度条
          '''
          latest_iteration = st.empty()
          bar = st.progress(0)
          for i in range(10):
              # Update the progress bar with each iteration.
              latest_iteration.text(f'Iteration {i + 1}')
              bar.progress(i + 1)
              time.sleep(0.1)
          '''
          self.comment["img_size"] = Common.detect_size(upload_img)
          st.write("意见：%s" % Common.parse_comment(self.comment))
          st.write("客户要求：xxxxx")

  def text_ocr(self, option):
      st.title(option)

if __name__=="__main__":
    FUNC_MAP = {
      "Logo合规性检测": 0,
      "文字OCR": 1
    }
    app = App()
    option = st.sidebar.radio('', tuple(FUNC_MAP.keys()))
    # view，可以新增demo应用
    if FUNC_MAP[option] == 0:
        app.logo_detect(option)
    elif FUNC_MAP[option] == 1:
        app.text_ocr(option)


