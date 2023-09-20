import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 載入預訓練的 EfficientNetV2M 模型（不含頂層全連接層）
model = EfficientNetV2M(weights='imagenet')

# 選擇目標層（最後一個卷積層）
target_layer = model.get_layer('top_conv')

# 載入並預處理圖像
image_path = 'path_to_your_image.jpg'
image = load_img(image_path, target_size=(224, 224))
x = img_to_array(image)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

# 獲取目標類別的 index（例如，最高預測的類別）
preds = model.predict(x)
predicted_class = np.argmax(preds[0])

# 計算目標類別的梯度
with tf.GradientTape() as tape:
    last_conv_output = target_layer.output
    tape.watch(last_conv_output)
    grads = tape.gradient(preds[:, predicted_class], last_conv_output)

# 計算通道權重（特徵圖上的梯度）
grads_mean = tf.reduce_mean(grads, axis=(1, 2))
grad_cam = tf.reduce_sum(last_conv_output * grads_mean[:, tf.newaxis, tf.newaxis], axis=-1)

# 歸一化 Grad-CAM
grad_cam = np.maximum(grad_cam, 0)
grad_cam = grad_cam / np.max(grad_cam)

# 將 Grad-CAM 轉換為熱力圖
heatmap = cv2.resize(grad_cam, (image.shape[1], image.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

# 融合熱力圖和原始圖像
superimposed_img = heatmap * 0.4 + image

# 顯示結果
plt.imshow(superimposed_img / 255)
plt.axis('off')
plt.show()
