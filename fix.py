import os
import pandas as pd

parent_folder_path = "stat"
files = os.listdir(parent_folder_path)
csv_files = sorted([f for f in files if f.endswith(".csv")], key=lambda x: int(x[:-4]))
count = 0
for csv in csv_files:
    csv_path = os.path.join(parent_folder_path ,csv)
    # print(csv)
    df = pd.read_csv(csv_path)
    length = len(df['ws'])
    print(length)
    count+=length
    # df = df.drop_duplicates(subset='Time', keep='first')
    # # 重設索引
    # df = df.reset_index(drop=True)
    # df.to_csv(csv_path,index=False)

# data =os.listdir("images")
# length =len(data)


# parent_folder_path = "ty\sat"
# files = os.listdir(parent_folder_path)
# subfolders = sorted([f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))], key=lambda x: int(x))
# count = 0
# for sub_folder in subfolders:
#     sub_folder_path = os.path.join(parent_folder_path, sub_folder)
#     length = len(os.listdir(sub_folder_path))
#     print(length)
    # # print(csv_path)
    # df = pd.read_csv(csv_path)
    # length = len(df['ws'])
    # print(length)
    # count+=length