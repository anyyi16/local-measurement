import pandas as pd
import numpy as np
import math

data = pd.read_csv("C:/Users/Betsy/ds_csv/model_distance3.csv")
#grouped = data.groupby('fish_index')
unique_frame_index = data['frame_index'].unique()

def Euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_direction(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

data.sort_values(['frame_index', 'fish_index'], inplace=True) 

#print(data)
print(len(unique_frame_index))
# 计算方向和距离变化
for frame_index in range(236):
    frame_data = data[data['frame_index'] == unique_frame_index[frame_index]]
    for index, row in frame_data.iterrows():
        next_frame_index = frame_index + 1
        next_frame_data = data[(data['frame_index'] == unique_frame_index[next_frame_index]) & (data['fish_index'] == row['fish_index'])]
        #print(next_frame_data)
        if len(next_frame_data) > 0:
            next_x = next_frame_data['x'].values[0]
            next_y = next_frame_data['y'].values[0]
            
            direction_change = calculate_direction(row['x'], row['y'], next_x, next_y)
            distance_change = Euclidean_distance(row['x'], row['y'], next_x, next_y)
            
            data.at[index, 'direction_change'] = direction_change
            data.at[index, 'distance_change'] = distance_change

#all
data['avg_direction'] = 0
data['avg_speed'] = 0
for frame_index in unique_frame_index:
    frame_data = data[data['frame_index'] == frame_index]
    avg_direction = frame_data['direction_change'].mean() 
    avg_speed = frame_data['distance_change'].mean() 
    data.loc[data['frame_index'] == frame_index, 'avg_direction'] = avg_direction
    data.loc[data['frame_index'] == frame_index, 'avg_speed'] = avg_speed

list_all_closest_5 = []
avg_speed = []
avg_direction = []
for frame_index in unique_frame_index: #unique_frame_index
    # 获取相同frame_index的所有数据
    frame_data = data[data['frame_index'] == frame_index]
    #print(frame_data)

    all_fish_distances = []
    all_smallest_5_distances = []
    all_closest_5 = []
    # 遍历每个fish_index
    for idx, row in frame_data.iterrows():
        current_fish_index = row['fish_index']
        current_x = row['x']
        current_y = row['y']
        current_fish_distances = {}

        # 计算当前fish_index与其他各项的距离
        for _, other_row in frame_data.iterrows():
            
            if other_row['fish_index'] != current_fish_index:
                other_fish_index = other_row['fish_index']
                other_x = other_row['x']
                other_y = other_row['y']
                
                # 计算欧几里德距离
                distance = Euclidean_distance(current_x, current_y, other_x, other_y)
                current_fish_distances[other_fish_index] = distance
                sorted_fish_distances = sorted(current_fish_distances.items(), key=lambda x: x[1])
                closest_5 = sorted_fish_distances[:5]
        all_closest_5.append(closest_5)
    list_all_closest_5.extend(all_closest_5)
data["all_closest_5"] = list_all_closest_5
#print(list_all_closest_5[0])
avg_closest_direction = []
avg_closest_speed = []
 
# 5 closest
for idx, row in data.iterrows():  # 遍历数据框的每一行
    #print(data["all_closest_5"][idx])
    #break;
    closest_5_rows = data["all_closest_5"][idx]  # 获取对应的最近的5个点的列表
    #print(closest_5_rows)
    #break;
    desired_frame_index = data.iloc[idx]["frame_index"]
    closest_5_direction = []
    closest_5_speed = []
    for i in range(5):
        desired_fish_index = closest_5_rows[i][0]  # 获取距离值
        filtered_row = data[(data['fish_index'] == desired_fish_index) & (data['frame_index'] == desired_frame_index)]
        temp_direction = filtered_row['direction_change']
        temp_speed = filtered_row['distance_change']
        closest_5_direction.append(temp_direction)
        closest_5_speed.append(temp_speed)
    #print(closest_5_direction)
    #break;
    avg_closest_direction.append(np.mean(closest_5_direction))
    avg_closest_speed.append(np.mean(closest_5_speed))
data["avg_closest_direction"] = avg_closest_direction
data["avg_closest_speed"] = avg_closest_speed


#print(data)
outputpath = "C:/Users/Betsy/ds_csv/model_direction3.csv"
data.to_csv(outputpath, sep = ',', index = True, header = True)
