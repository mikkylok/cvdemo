import cv2
import numpy as np
# widgets
class Common(object):
  def __init__(self):
    pass

  @staticmethod
  def detect_size(img):
    # 检测尺寸是否符合要求： 30*30
    #img = cv2.imread(img, 1)
    # 将图片从二进制流转换为矩阵
    #img = np.array(img.convert("RGB"))
    from PIL import Image
    img = Image.open(img)
    print (img.size)
    if img.size == (500, 322):
      return True
    else:
      return False

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

