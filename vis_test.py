
# Display
from IPython.display import Image, display

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt

import tensorflow as tf


# 定义图像尺寸
image_width = 256
image_height = 256

# 加载预训练的VGG模型（不包含顶部的全连接层）
vgg_model = EfficientNetV2M(weights=None, include_top=False, input_shape=(image_height, image_width, 3),pooling='max')
# vgg_model.summary()
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
# 编译回归器模型
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
regressor.load_weights('effM_weights(ws)(round3).h5')
regressor.summary()


