import os
from PIL import Image
import numpy as np
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2
import pandas as pd
from sklearn.model_selection import train_test_split
# import visualkeras 
# from PIL import ImageFont 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 设置包含图片文件夹的父文件夹路径
parent_folder_path = "images"

# 定义图像尺寸
image_width = 256
image_height = 256

# 定义批处理大小
batch_size = 60

# 创建一个空列表来存储所有图像的数据和标签
images_data = []
ratings = []
x = []
y = []
# 加载预训练的VGG模型（不包含顶部的全连接层）
vgg_model = EfficientNetV2M(weights=None, include_top=False, input_shape=(image_height, image_width, 3),pooling='max')

# 创建新的回归器模型
regressor = Sequential()
regressor.add(vgg_model)  # 将VGG模型添加到回归器模型中
regressor.add(Flatten())

# 添加 Dropout 层
# regressor.add(Dropout(0.2))  # 设置丢弃率为 0.5

# # # 添加全连接层，并使用 L2 正则化工具
# regressor.add(Dense(256, activation='linear'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(64, activation='linear'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(16, activation='linear'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(4, activation='linear'))  # 设置 L2 正则化参数为 0.01
# # 添加 Dropout 层
# 添加输出层，使用线性激活函数
regressor.add(Dense(1, activation='linear'))

# 编译回归器模型
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# visualkeras.layered_view(vgg_model , legend=True) 

regressor.load_weights('weight\effM_2022.h5')
# 創建 ImageDataGenerator 對象並設置數據增強的參數
datagen = ImageDataGenerator(
    rotation_range=10,         # 隨機旋轉圖片的角度範圍
    width_shift_range=0.1,     # 隨機水平平移圖片的範圍
    height_shift_range=0.1,    # 隨機垂直平移圖片的範圍
    channel_shift_range=10,    # 通道偏移範圍，增加隨機顏色偏移
    zoom_range=0.1,            # 縮放範圍，增加隨機縮放
    brightness_range=[0.9, 1.1],  # 亮度調整範圍，降低或增加亮度
    shear_range=0.05,           # 剪切強度範圍，增加剪切變換
)
df = pd.read_csv("ws.csv")
ws = np.array(df['ws'])
#遍历父文件夹中的所有子文件夹
for i in range(1):
    for sub_folder in os.listdir(parent_folder_path):
        sub_folder_path = os.path.join(parent_folder_path, sub_folder)
        # 读取子文件夹中的所有图片
        image_files = os.listdir(sub_folder_path)
        rating_number = 0
        # # 遍历图片文件列表
        for i, file_name in enumerate(image_files):
            file_path = os.path.join(sub_folder_path, file_name)
            # 使用Pillow库打开GIF图片
            gif_image = Image.open(file_path)

            # 获取GIF的第一帧
            gif_first_frame = gif_image.convert('RGB')

            # 调整图像尺寸并进行裁剪
            # resized_image = gif_first_frame.resize((image_width, image_height))

            # 计算裁剪的左上角坐标
            left = (gif_first_frame.width - image_width) // 2
            top = (gif_first_frame.height - image_height) // 2

            # 裁剪图像
            cropped_image = gif_first_frame.crop((left, top, left + image_width, top + image_height))
            # 将图像数据添加到列表中
            image_data = np.array(cropped_image)
            img_array = np.expand_dims(image_data, axis=0)

            # 生成增強後的圖片
            aug_iter = datagen.flow(img_array, batch_size=1)
            aug_img = next(aug_iter)[0].astype(np.uint8)
            images_data.append(aug_img)
            # image_data = np.array(cropped_image)
            # images_data.append(image_data)
            ratings.append(df['wind_speed'][rating_number])
            rating_number += 1

            if len(images_data) == batch_size : #or i == len(image_files) - 1
                # 将图像数据转换为NumPy数组
                images_data = np.array(images_data,dtype='float')
                ratings = np.array(ratings,dtype='float')
                train_images, val_images, train_ratings, val_ratings = train_test_split(images_data, ratings, test_size=0.3, random_state=42)

                #对图像数据进行预处理
                # processed_images = vgg_model.preprocess_input(images_data)

                # 训练回归器模型
                regressor.fit(images_data, ratings, batch_size=batch_size, epochs=1)
                val_loss, val_mae = regressor.evaluate(val_images, val_ratings)

                y.append(val_loss)
                images_data = []
                ratings = []    

if len(images_data) > 0 : #or i == len(image_files) - 1
    # 将图像数据转换为NumPy数组
    images_data = np.array(images_data,dtype='float')
    ratings = np.array(ratings,dtype='float')
    train_images, val_images, train_ratings, val_ratings = train_test_split(images_data, ratings, test_size=0.2, random_state=3)

    #对图像数据进行预处理
    # processed_images = vgg_model.preprocess_input(images_data)

    # 训练回归器模型
    regressor.fit(images_data, ratings, batch_size=batch_size, epochs=1)


    y.append(val_loss)
    # 绘制散点图：预测值 vs 真实值
    plt.figure(figsize=(8, 6))
    # plt.scatter(x, y, alpha=0.7)
    plt.plot(y)
    # plt.xlabel('True Wind Speed')
    plt.ylabel('error')
    plt.title('Train process')
    plt.grid()
    plt.show()
    # 重置图像数据列表
    images_data = []
    ratings = []    
print("訓練結束")  
regressor.save_weights('./effM_weights(para).h5')

            