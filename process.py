import os
from PIL import Image
import numpy as np
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt
import time

folder_path = "test_images/png/2307"  # 替换为您实际的文件夹路径

gif_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.gif')]
# 定义图像尺寸
image_width = 256
image_height = 256

for file_name in gif_files:
    file_path = os.path.join(folder_path,file_name)

    # 使用Pillow库打开GIF图片
    gif_image = Image.open(file_path)

    # 获取GIF的第一帧
    gif_first_frame = gif_image.convert('RGB')

    # 计算裁剪的左上角坐标
    left = (gif_first_frame.width - image_width) // 2
    top = (gif_first_frame.height - image_height) // 2

    # 裁剪图像
    cropped_image = gif_first_frame.crop((left, top, left + image_width, top + image_height))

        # 生成新的文件名，以随机数命名
    new_file_name = file_name[:-3]+"png"


    # 拼接新的文件路径
    new_file_path = os.path.join(folder_path, new_file_name)

    # 保存裁剪后的图像
    cropped_image.save(new_file_path)
    
    # 关闭GIF文件
    gif_image.close()

    os.remove(file_path)