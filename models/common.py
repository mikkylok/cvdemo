from PIL import Image, ImageDraw
# widgets
class Common(object):
  def __init__(self):
    pass

  @staticmethod
  def inspect_size(img):
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

  '''
  合成调色板
  @staticmethod
  def detect_main_colors(original_image):
      palette_height = 80
      outline_width = 5
      pal_unit_width = 50
      numcolors = 10
      original_image = Image.open(original_image)
      colors = get_colors(original_image)  # array of colors in the image

      pal = Image.new("RGB", (numcolors * (pal_unit_width), palette_height))
      draw = ImageDraw.Draw(pal)
      posx = 0

      # making the palette
      for col in colors:
          draw.rectangle([posx, 0, posx + pal_unit_width, palette_height], fill=col, width=outline_width,
                         outline="#000")
          posx += pal_unit_width

      del draw
      return pal
  '''

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
      colors = get_main_colors(original_image)  # array of colors in the image
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


def get_main_colors(original_image):
    '''
    :param image:
    :return: [(89.2%, (255,243,3)), (像素点百分比，rgb色值), ...]
    方法一：可以先压缩（需要，不然像素点太多了），后使用getcolors（）,获取图片中的所有像素
    '''
    #image = original_image.resize((80, 80))
    #colors = image.getcolors(80 * 80)  # array of colors in the image
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
    top_colors = [("%.2f%%" % (float(color[0]) * 100 / (width * height - transparent_pixel_no)), color[1]) for color in quick_sort(top_colors)[:10]]
    return top_colors

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
    return quick_sort(left) + [mid] + quick_sort(right)