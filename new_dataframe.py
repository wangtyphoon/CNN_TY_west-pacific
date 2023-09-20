import numpy as np
import pandas as pd

# 创建一个 92580 行、5 列的空矩阵
empty_matrix = np.empty((92580, 5))

# 使用这个空矩阵作为数据源来创建 DataFrame
df = pd.DataFrame(data=empty_matrix, columns=['lat', 'lon', 'pressure', 'wind_speed', 'time'])

# 指定CSV文件名
csv_file = "typhoon.csv"

# 将DataFrame保存为CSV文件
df.to_csv(csv_file, index=False)

print(f"空矩阵已成功写入 {csv_file}")