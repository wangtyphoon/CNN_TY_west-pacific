import os
from PIL import Image
import numpy as np
from keras.applications.convnext import ConvNeXtBase
from keras.models import Sequential
from keras.layers import Flatten, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from keras.optimizers import Adam
# 设置包含图片文件夹的父文件夹路径
folder_path = "images"

# 定义图像尺寸
image_width = 256
image_height = 256

# 定义批处理大小
batch_size = 30


#訓練資料
df = pd.read_csv("typhoon.csv")
ws = np.array(df['wind_speed'],dtype=float)
image_files = os.listdir(folder_path)

# 创建一个空列表来存储所有图像的数据和标签
images_data = []
ratings = []
x = []
test_error = []
train_error = []

# 加载预训练的VGG模型（不包含顶部的全连接层）
vgg_model = ConvNeXtBase(weights=None, include_top=False, input_shape=(image_height, image_width, 3),pooling='max')

# 创建新的回归器模型
regressor = Sequential()
regressor.add(vgg_model)  # 将VGG模型添加到回归器模型中
regressor.add(Flatten())

# # # 添加全连接层，并使用 L2 正则化工具
regressor.add(Dense(256, activation='relu'))  
regressor.add(Dense(128, activation='relu'))  
regressor.add(Dense(64, activation='relu'))  
regressor.add(Dense(32, activation='relu'))  
regressor.add(Dense(16, activation='relu'))  
regressor.add(Dense(8, activation='linear'))  
regressor.add(Dense(4, activation='linear'))  
regressor.add(Dense(2, activation='linear'))  
regressor.add(Dense(1, activation='linear'))

# 设置自定义的学习率
custom_learning_rate = 0.00001

# 创建一个Adam优化器，设置自定义学习率
custom_optimizer = Adam(learning_rate=custom_learning_rate)
# 编译回归器模型
# 编译回归器模型，使用自定义的优化器
regressor.compile(optimizer=custom_optimizer, loss='mean_squared_error', metrics=['mae'])
# regressor.load_weights('effM_weights(ws)(round2)(part3).h5')

# # 按照文件名中的数字部分进行排序
sorted_image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
part = 0
image_files = sorted_image_files[92580*part:]#30000*part+30000]
#初始設定
epoch = 0
rating_number = 92580*part
for file_name in image_files:
    file_path = os.path.join(folder_path, file_name)
    # print(file_path)
    # 使用PIL库打开PNG图像
    png_image = Image.open(file_path)

    # 将PNG图像转换为数组
    png_array = np.array(png_image)
    images_data.append(png_array)
    ratings.append(ws[rating_number])
    rating_number += 1
    if len(images_data) == batch_size : 
        #記錄輪次
        epoch += 1
        x.append(epoch)
        # 将图像数据转换为NumPy数组
        images_data = np.array(images_data,dtype='float')
        ratings = np.array(ratings,dtype='float')

        # 訓練測試數據分離
        train_images, val_images, train_ratings, val_ratings = train_test_split(images_data, ratings, test_size=0.3, random_state=41)

        # 训练回归器模型
        regressor.fit(images_data, ratings, batch_size=batch_size, epochs=1)
        val_loss, val_mae = regressor.evaluate(val_images, val_ratings)
        train_loss,train_mae = regressor.evaluate(train_images,train_ratings)

        test_error.append(val_loss)
        train_error.append(train_loss)

        images_data = []
        ratings = [] 
        time.sleep(5)