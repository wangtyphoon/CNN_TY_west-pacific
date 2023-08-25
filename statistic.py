import numpy as np
import pandas as pd
from datetime import datetime
import os

# 定义线性插值函数
def linear_interpolate(value1, value2, time_diff, time_total):
    return value1 + (value2 - value1) * time_diff / time_total

def interpolate_data(csv, bst, reference_time):
    length = len(csv['ws'])
    count = 0

    for i in range(length):
        sat_time = datetime.strptime(csv['IR_time'][i], "%Y-%m-%d %H:%M:%S")
        bst_old = datetime.strptime(bst['Time'][count], "%Y-%m-%d %H:%M:%S")
        bst_new = datetime.strptime(bst['Time'][count + 1], "%Y-%m-%d %H:%M:%S")

        if bst_new > sat_time >= bst_old:
            time_diff = (sat_time - bst_old).total_seconds()
            time_total = (bst_new - bst_old).total_seconds()

            interpolated_wind = linear_interpolate(bst['wind_speed'][count], bst['wind_speed'][count + 1], time_diff, time_total)
            interpolated_pressure = linear_interpolate(bst['pressure'][count], bst['pressure'][count + 1], time_diff, time_total)
            interpolated_lat = linear_interpolate(bst['lat'][count], bst['lat'][count + 1], time_diff, time_total)
            interpolated_lon = linear_interpolate(bst['lon'][count], bst['lon'][count + 1], time_diff, time_total)

            csv['wind_speed'][i] = interpolated_wind
            csv['pressure'][i] = interpolated_pressure
            csv['lat'][i] = interpolated_lat
            csv['lon'][i] = interpolated_lon
        elif sat_time >= bst_new:
            count += 1
            bst_old = datetime.strptime(bst['Time'][count], "%Y-%m-%d %H:%M:%S")
            bst_new = datetime.strptime(bst['Time'][count + 1], "%Y-%m-%d %H:%M:%S")

            time_diff = (sat_time - bst_old).total_seconds()
            time_total = (bst_new - bst_old).total_seconds()

            interpolated_wind = linear_interpolate(bst['wind_speed'][count], bst['wind_speed'][count + 1], time_diff, time_total)
            interpolated_pressure = linear_interpolate(bst['pressure'][count], bst['pressure'][count + 1], time_diff, time_total)
            interpolated_lat = linear_interpolate(bst['lat'][count], bst['lat'][count + 1], time_diff, time_total)
            interpolated_lon = linear_interpolate(bst['lon'][count], bst['lon'][count + 1], time_diff, time_total)

            csv['wind_speed'][i] = interpolated_wind
            csv['pressure'][i] = interpolated_pressure
            csv['lat'][i] = interpolated_lat
            csv['lon'][i] = interpolated_lon
    return csv

def stat(csv_path,bst_path,new_path):
    # 读取数据文件
    csv = pd.read_csv(csv_path)
    csv['lat'] = None
    csv['lon'] = None
    csv['pressure'] = None
    csv['wind_speed'] = None
    bst = pd.read_csv(bst_path)

    # 参考时间（UNIX时间戳起始时间）
    reference_time = datetime(2005, 1, 1)

    # 插值数据
    csv = interpolate_data(csv, bst, reference_time)
    csv.to_csv(new_path,index=False)

parent_folder_path = "copy"
data_path = "bst"
newfolder = "stat"
# 列出資料夾中的所有檔案
files = os.listdir(parent_folder_path)
csv_files = sorted([f for f in files if f.endswith(".csv")], key=lambda x: int(x[:-4]))

for csv in csv_files:
    csv_path = os.path.join(parent_folder_path ,csv)
    bst_path = os.path.join(data_path ,csv)
    new_path = os.path.join(newfolder ,csv)
    stat(csv_path,bst_path,new_path)
    print(csv_path)

