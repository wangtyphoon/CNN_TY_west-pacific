import pandas as pd
import numpy as np
import os
from PIL import Image

df_number = pd.read_csv("random_numbers.csv")
randow_number = np.array(df_number['Random Number'])
df_ws = pd.read_csv("typhoon.csv")

# 定义图像尺寸
image_width = 256
image_height = 256

parent_folder_path = "ty\sat"
count = 7835
subfolders = sorted([f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))], key=lambda x: int(x))
number = 0 
new_folder = "images"
for sub_folder in subfolders:
    # print(sub_folder)
    sub_folder_path = os.path.join(parent_folder_path, sub_folder)
    df = pd.read_csv("stat/"+sub_folder_path[-5:]+".csv")
    image_files = os.listdir(sub_folder_path)
    print(len(image_files))
    image_files = sorted(os.listdir(sub_folder_path),  key=lambda x: int(x[:-4]))
    rating_number = 0
    for i, file_name in enumerate(image_files):
        file_path = os.path.join(sub_folder_path, file_name)
        print(file_path)
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
        new_file_name = str(randow_number[count])+".png"
        # 拼接新的文件路径
        new_file_path = os.path.join(new_folder, new_file_name)

        # 保存裁剪后的图像
        cropped_image.save(new_file_path)
        df_ws['ws'][randow_number[count]-1] = df['ws'][rating_number]
        count += 1
        rating_number += 1
        
df_ws.to_csv("ws.csv",index=False)
print("图像裁剪并另存为PNG文件完成！")
print(count)
