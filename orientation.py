import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Betsy/Desktop/\ds project/analytics/distance and direction2.csv")
#data = data.head(29)
#print(data)
direction_gap = []
for idx, row in data.iterrows():
    #print(row)
    frame = row["frame_index"]
    closest = row["all_closest_5"][3]
    closest_row = data[(data["frame_index"] == frame) & (data["fish_index"] == closest)]
    if not closest_row.empty:
        closest_direction_change = closest_row.iloc[0]["direction_change"]
        row_direction_change = row["direction_change"]
        row_direction_gap = abs(row_direction_change - closest_direction_change)
        direction_gap.append(row_direction_gap)
data["direction_gap"] = direction_gap
#print(direction_gap)
roo_dis = []
small_gap = data[data["direction_gap"] >= 1.57] #0.175
for idx, row in small_gap.iterrows():
    roo_dis.append(row["list_smallest"])

roo_dis_sorted = sorted(roo_dis)
roo_dis_sorted = roo_dis_sorted[int(len(roo_dis_sorted)*0.05)]
print(roo_dis_sorted)

# 创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列的子图布局

# 绘制直方图
ax1.hist(roo_dis, bins=20, edgecolor='black')  # 指定分箱数和边缘颜色
ax1.set_title("Histogram for roo distribution")
ax1.set_xlabel("Values")
ax1.set_ylabel("Frequency")

# 绘制箱线图
ax2.boxplot(roo_dis)
ax2.set_title("Box Plot for roo")
ax2.set_ylabel("Values")

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()
'''
outputpath = "C:/Users/Betsy/Desktop/\ds project/analytics/direction_change.csv"
data.to_csv(outputpath, sep = ',', index = True, header = True)
'''