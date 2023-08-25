import random
import csv

# 生成不重复的随机数序列
random_numbers = random.sample(range(1, 92581), 92580)

# 指定CSV文件名
csv_file = "random_numbers.csv"

# 将随机数写入CSV文件
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # 写入表头（可选）
    writer.writerow(["Random Number"])

    # 写入随机数数据
    for number in random_numbers:
        writer.writerow([number])

print(f"随机数已成功写入 {csv_file}")