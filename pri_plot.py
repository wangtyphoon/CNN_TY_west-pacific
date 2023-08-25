import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def wind_speed(path):
    df = pd.read_csv(path)
    ws = np.array(df['ws'])
    return ws

folder_path = "test_images\prediction"
csv_list = os.listdir("test_images/prediction")
csv_list.sort()  # 將文件列表按升序排序

# 創建子圖
fig, axes = plt.subplots(3, 2, figsize=(10, 12))  # 3行2列的子圖

# 遍歷每個CSV文件並在子圖中繪製數據
for index, csv in enumerate(csv_list):
    path = os.path.join(folder_path, csv)
    data = wind_speed(path)
    length = len(data)
    x = np.linspace(1, length, length).astype(int)
    
    row = index // 2
    col = index % 2
    ax = axes[row, col]
    
    ax.plot(x, data, label=f'Data {csv[:-4]}')  # 使用文件名作為標籤
    ax.set_title(f'{csv[:-4]}')  # 使用文件名作為子圖標題
    ax.set_ylabel('Wind Speed')
    ax.legend()

# 調整子圖間的間距
plt.tight_layout()
plt.savefig("subplot",dpi=400)
# 顯示圖像
plt.show()


