import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
import math

data = pd.read_csv('C:/Users/Betsy/ds_csv/transformed_data3.csv')
#print(data)
data.sort_values(['frame_index', 'fish_index'], inplace=True)
def Euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

#print(Euclidean_distance(1,2,3,4))
# 获取所有不重复的frame_index
unique_frame_index = data['frame_index'].unique()
#print(unique_frame_index)
list_avg_all = []
list_avg_5_smallest = []
list_smallest = []
avg_fish_distances = []
avg_smallest_distances = []
centroid_avg_fish_distances = []
centroid_avg_smallest_distances = []
# 遍历每个frame_index
for frame_index in unique_frame_index: #unique_frame_index
    # 获取相同frame_index的所有数据
    frame_data = data[data['frame_index'] == frame_index]
    #print(frame_data)

    all_fish_distances = []
    all_smallest_5_distances = []
    all_smallest_distances = []


    # 遍历每个fish_index
    for idx, row in frame_data.iterrows():
        current_fish_index = row['fish_index']
        current_x = row['x']
        current_y = row['y']
        current_fish_distances = []
        # 计算当前fish_index与其他各项的距离
        for _, other_row in frame_data.iterrows():
            
            if other_row['fish_index'] != current_fish_index:
                other_fish_index = other_row['fish_index']
                other_x = other_row['x']
                other_y = other_row['y']
                
                # 计算欧几里德距离
                distance = Euclidean_distance(current_x, current_y, other_x, other_y)
                current_fish_distances.append(distance)
                sorted_fish_distances = sorted(current_fish_distances)
                smallest_5_distances = sorted_fish_distances[:5]
                smallest_distances = sorted_fish_distances[0]
        #print(smallest_5_distances)
        
        all_fish_distances.append(np.mean(current_fish_distances))
        all_smallest_5_distances.append(np.mean(smallest_5_distances))
        all_smallest_distances.append(smallest_distances)

        #每个frame的均值
    list_avg_all.extend(all_fish_distances)
    list_avg_5_smallest.extend(all_smallest_5_distances)
    list_smallest.extend(all_smallest_distances)
    
    
    sorted_all_fish_distances = sorted(all_fish_distances)
    sorted_smallest_5_distances = sorted(all_smallest_5_distances)
    centroid_fish_distances = sorted_all_fish_distances[0]
    centroid_smallest_5_distances = sorted_smallest_5_distances[0]
    
    avg_fish_distances.append(np.mean(all_fish_distances))
    avg_smallest_distances.append(np.mean(all_smallest_5_distances))
    centroid_avg_fish_distances.append(centroid_fish_distances)
    centroid_avg_smallest_distances.append(centroid_smallest_5_distances)

data["avg_all_distances"] =  list_avg_all
data["avg_closest_distances"] = list_avg_5_smallest
data["list_smallest"] = list_smallest
sorted_smallest = sorted(list_smallest)
smallest_25 = sorted_smallest[int(len(sorted_smallest)*0.05)]
smallest_75 = sorted_smallest[int(len(sorted_smallest)*0.75)]
#print(smallest_25)
#print(smallest_75)
data.to_csv('C:/Users/Betsy/ds_csv/model_distance3.csv', index=False)
'''
plt.boxplot(list_smallest)

# 添加标题和标签
plt.title("Box Plot with Custom Percentiles")
plt.ylabel("Values")



# 计算自定义分位数对应的值
percentiles = np.percentile(list_smallest, [10, 80, 10])

# 在图上添加标签显示分位数对应的值
for i, percentile in enumerate(percentiles):
    plt.text(1.1, percentile, f"{i+1}0% Percentile: {percentile:.2f}", va='center')

# 显示图像
plt.show()

#print(data)

outputpath = "C:/Users/Betsy/Desktop/\ds project/analytics/distance.csv"
data.to_csv(outputpath, sep = ',', index = True, header = True)

print(avg_fish_distances)
print(avg_smallest_distances)
print(centroid_avg_fish_distances)
print(centroid_avg_smallest_distances)


# 创建一个2x2的子图网格
fig = plt.figure(figsize=(9, 6))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# 绘制左上角的数据分布图
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(avg_fish_distances, kde=True, ax=ax1)
ax1.set_title("Distance distribution of all fish")
ax1.set_xlabel("value")
ax1.set_ylabel("num")

# 绘制右上角的数据分布图
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(avg_smallest_distances, kde=True, ax=ax2)
ax2.set_title("Distance distribution of 5 closest fish")
ax2.set_xlabel("value")
ax2.set_ylabel("num")

# 绘制左下角的数据分布图
ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(centroid_avg_fish_distances, kde=True, ax=ax3)
ax3.set_title("Distance distribution of all fish (center)")
ax3.set_xlabel("value")
ax3.set_ylabel("num")

# 绘制右下角的数据分布图
ax4 = fig.add_subplot(gs[1, 1])
sns.histplot(centroid_avg_smallest_distances, kde=True, ax=ax4)
ax4.set_title("Distance distribution of 5 closest fish (center)")
ax4.set_xlabel("value")
ax4.set_ylabel("num")

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()
'''
