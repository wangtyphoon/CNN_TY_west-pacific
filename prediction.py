import os
from PIL import Image
import numpy as np
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt
import time

ws = [] 
# 定义图像尺寸
image_width = 256
image_height = 256

# 加载预训练的VGG模型（不包含顶部的全连接层）
vgg_model = EfficientNetV2M(weights=None, include_top=False, input_shape=(image_height, image_width, 3),pooling='max')

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
# regressor.load_weights('effM_weights(ws)(round1)(part5).h5')



def prediction(model,image_data,weight):
    model.load_weights(weight)
    ws = model.predict(image_data)
    return ws

folder_path = "test_images/png/2307"

count = 0 
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path,file)
    png_image = Image.open(file_path)

    # 将PNG图像转换为数组
    png_array = np.array(png_image)
    image_data = np.expand_dims(png_array , axis=0)
    # ws1 = prediction(regressor,image_data,"effM_weights(ws)(round1)(part1).h5")
    # ws2 = prediction(regressor,image_data,"effM_weights(ws)(round1)(part2).h5")
    # ws3 = prediction(regressor,image_data,"effM_weights(ws)(round1)(part3).h5")
    # ws4 = prediction(regressor,image_data,"effM_weights(ws)(round1)(part4).h5")
    # ws5 = prediction(regressor,image_data,"effM_weights(ws)(round1)(part5).h5")
    # ws6 = prediction(regressor,image_data,"effM_weights(ws)(round2)(part1).h5")
    # ws7 = prediction(regressor,image_data,"effM_weights(ws)(round2)(part2).h5")
    # ws8 = prediction(regressor,image_data,"effM_weights(ws)(round2)(part3).h5")
    data = prediction(regressor,image_data,"effM_weights(ws)(round3).h5").flatten()[0]
    ws.append(data)
ws_df = pd.DataFrame(ws, columns=['ws'])
plt.plot(ws)
plt.show()
ws_df.to_csv("test_images/prediction/2307.csv",index=False)
    


