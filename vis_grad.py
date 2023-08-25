import tensorflow as tf
import numpy as np
import cv2
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.models import Sequential
from keras.layers import Flatten, Dense
import os 

def get_grad_cam_regression(model, image_path, target_layer_name,number):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Use the correct preprocess_input function

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(target_layer_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_output, conv_output = iterate(img_array)
        loss = tf.reduce_sum(model_output)  # Use the regression output as the loss

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = (heatmap * 255).astype(np.float32)  # Convert to float32
    if not os.path.exists("vis/heatmap/"+image_path[11:-4]):
        images_folder = "vis/heatmap/"+image_path[11:-4]
        # 创建文件夹
        os.makedirs(images_folder)

    images_folder = "vis/heatmap/"+image_path[11:-4]
    heatmap_output_path = images_folder+"/"+number+'.jpg'
    cv2.imwrite(heatmap_output_path, heatmap)

    bgr_img = cv2.cvtColor(img_array[0], cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    superimposed_img = cv2.addWeighted(bgr_img, 0.6, heatmap, 0.4, 0)

     # Add legend to the image
    legend = np.zeros((25, superimposed_img.shape[1], 3), dtype=np.uint8)
    legend[:, :, 0] = 255  # Set blue channel to 255 (blue color)
    superimposed_img = np.vstack((legend, superimposed_img))

    if not os.path.exists("vis/gradcam/"+image_path[11:-4]):
        images_folder = "vis/gradcam/"+image_path[11:-4]
        # 创建文件夹
        os.makedirs(images_folder)

    images_folder = "vis/gradcam/"+image_path[11:-4]
    heatmap_output_path = images_folder+"/"+number+'.jpg'
    # Save the visualization
    output_path = 'vis/gradcam/grad_cam_visualization'+number+'.jpg'
    cv2.imwrite(output_path, superimposed_img)

    return superimposed_img

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
regressor.load_weights('weight/effM_weights(ws)(round3).h5')

target_layer_name = 'efficientnetv2-m'  # 要提取的層的名稱
model = regressor.get_layer(target_layer_name)

conv_layers = []
for layer in model.layers:
    if 'conv' in layer.name:  # 假設卷積層的名稱中包含 'conv'
        conv_layers.append(layer.name)
# Set the target layer name
# target_layer_name = 'block2b_project_conv'

# Set the path to the image you want to visualize
image_path = 'test_images/2023wp06_4kmirimg_202308080750.png'
number = 1
print('hi')
for target_layer_name in conv_layers:
    grad_cam = get_grad_cam_regression(model, image_path, target_layer_name,str(number))
    number += 1
