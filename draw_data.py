import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

df = pd.read_csv("typhoon.csv")
ws = np.array(df['wind_speed'])
pressure = np.array(df['pressure'])
lat = np.array(df['lat'])
lon = np.array(df['lon'])

plt.hist(ws, bins=30, alpha=1, color='blue')

# 设置标题和标签
plt.title('Value Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 设置X轴刻度为10的倍数
x_locator = MultipleLocator(20)
plt.gca().xaxis.set_major_locator(x_locator)

# 显示直方图
plt.show()
plt.scatter(lon,lat,s=0.1)