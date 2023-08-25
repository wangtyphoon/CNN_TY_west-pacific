import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

print(tf.test.is_built_with_gpu_support())

# # 设置包含图片文件夹的父文件夹路径
# folder_path = "images"

# # 定义图像尺寸
# image_width = 256
# image_height = 256

# # 定义批处理大小
# batch_size = 100

# #訓練資料
# df = pd.read_csv("ws.csv")
# ws = np.array(df['ws'])
# image_files = os.listdir(folder_path)

# # 创建一个空列表来存储所有图像的数据和标签
# images_data = []
# ratings = []
# x = []
# test_error = []
# train_error = []

# # 加载预训练的VGG模型（不包含顶部的全连接层）
# vgg_model = EfficientNetV2M(weights=None, include_top=False, input_shape=(image_height, image_width, 3),pooling='max')

# # 创建新的回归器模型
# regressor = Sequential()
# regressor.add(vgg_model)  # 将VGG模型添加到回归器模型中
# regressor.add(Flatten())

# # # # 添加全连接层，并使用 L2 正则化工具
# regressor.add(Dense(256, activation='relu'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(128, activation='relu'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(64, activation='relu'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(32, activation='relu'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(16, activation='relu'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(8, activation='linear'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(4, activation='linear'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(2, activation='linear'))  # 设置 L2 正则化参数为 0.01
# regressor.add(Dense(1, activation='linear'))

# # 编译回归器模型
# regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# regressor.load_weights('weight\effM_cmplex\effM_weights(round1)(part4).h5')
# start = 0
# end = 58320
# epoch = 0

# for i in range(start,end):
#     rating_number = i
#     file_name = str(i+1)+".png"  #需要+1編號不同
#     file_path = os.path.join(folder_path, file_name)
#     # 使用PIL库打开PNG图像
#     png_image = Image.open(file_path)

#     # 将PNG图像转换为数组
#     png_array = np.array(png_image)
#     images_data.append(png_array)
#     ratings.append(ws[rating_number])
#     rating_number += 1
#     if len(images_data) == batch_size : 
#         #記錄輪次
#         epoch += 1
#         x.append(epoch)

#         # 将图像数据转换为NumPy数组
#         images_data = np.array(images_data,dtype='float')
#         ratings = np.array(ratings,dtype='float')

#         # 训练回归器模型
#         val_loss, val_mae = regressor.evaluate(images_data, ratings)
#         test_error.append(val_loss)

#         images_data = []
#         ratings = [] 
#         time.sleep(3)      
# # 繪製誤差分布
# plt.figure(figsize=(8, 6))
# plt.plot(x,test_error,color="red",label="test")
# plt.legend()
# plt.ylabel('error')
# plt.title('Test Process')
# plt.grid()
# plt.savefig("weight\effM_cmplex/round1.png",dpi=400)
# plt.show()

# print("測試結束")  
