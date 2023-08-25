from keras.models import Model
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.efficientnet_v2 import preprocess_input
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
regressor.load_weights('weight\effM_weights(ws)(round3).h5')
# regressor.summary()
target_layer_name = 'efficientnetv2-m'  # 要提取的層的名稱
target_model = regressor.get_layer(target_layer_name)

# 創建一個新的模型，僅包含卷積層
conv_layers = []
for layer in target_model.layers:
    if 'conv' in layer.name:  # 假設卷積層的名稱中包含 'conv'
        conv_layers.append(layer)
# target_layer = target_model.get_layer('block2c_expand_conv')
# 創建一個新的模型，僅包含卷積層
# partial_model = Model(inputs=target_model.input, outputs=target_layer.output)

image_path = 'test_images/2023wp06_4kmirimg_202308080750.png'  # 替換為您的圖像路徑
image = load_img(image_path, target_size=(image_height, image_width))
image_array = img_to_array(image)
image_array = preprocess_input(image_array)  # 預處理圖像
image_array = np.expand_dims(image_array, axis=0)  # 將圖像張量增加一個維度
if not os.path.exists("vis/feature/"+image_path[11:-4]):
    images_folder = "vis/feature/"+image_path[11:-4]
    # 创建文件夹
    os.makedirs(images_folder)
# ... (previous code)
number = 1
for target in conv_layers[:]:
    conv_model = Model(inputs=target_model.input, outputs=target.output)

    # Get all convolutional layer's feature maps
    feature_maps = conv_model.predict(image_array)
    merged_feature_map = np.mean(feature_maps, axis=3)

    # for i in range(feature_maps.shape[-1]):
    plt.figure()
    plt.imshow(merged_feature_map[0,:,:], cmap='viridis')  # 可以根據需要選擇不同的顏色映射
    plt.title(f"Feature Map ")
    plt.savefig(images_folder+"/"+str(number)+".png",dpi=200)
    plt.show()
    number += 1
    
