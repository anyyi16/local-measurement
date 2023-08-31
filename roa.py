import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Betsy/Desktop/\ds project/analytics/distance and direction2.csv")
unique_frame_index = list(data['frame_index'].unique())
#print(unique_frame_index)
direction_gap = []
#data = data.head(60)

def Euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

distance_gap = []
for idx, row in data.iterrows():
    #print(row)
    frame = row["frame_index"]
    closest = row["all_closest_5"][3]
    indice = unique_frame_index.index(frame)
    if indice+1 <len(unique_frame_index):
        next_row =  data[(data["frame_index"] == unique_frame_index[indice+1]) & (data["fish_index"] == row["fish_index"])]
        current_closest_row = data[(data["frame_index"] == unique_frame_index[indice]) & (data["fish_index"] == closest)]
        next_closest_row = data[(data["frame_index"] == unique_frame_index[indice+1]) & (data["fish_index"] == closest)]
        if not (next_row.empty or current_closest_row.empty or next_closest_row.empty):
            current_closest_x = current_closest_row.iloc[0]["x"]
            current_closest_y = current_closest_row.iloc[0]["y"]
            next_closest_x = next_closest_row.iloc[0]["x"]
            next_closest_y = next_closest_row.iloc[0]["y"]
            current_x = row["x"]
            current_y = row["y"]
            next_x = next_row.iloc[0]["x"]
            next_y = next_row.iloc[0]["y"]
            current_distance = Euclidean_distance(current_x, current_y, current_closest_x, current_closest_y)
            next_distance = Euclidean_distance(next_x, next_y, next_closest_x, next_closest_y)
            temp_distance_gap = next_distance - current_distance
            if temp_distance_gap < 0:
                distance_gap.append(row["avg_all_distances"])
            #distance_gap.append(temp_distance_gap)

distance_gap_sorted = sorted(distance_gap)
distance_gap_sorted95 = distance_gap_sorted[int(len(distance_gap_sorted)*0.95)]
print(distance_gap_sorted95)
'''
plt.hist(distance_gap)
plt.title("Histogram for distance gap distribution")
plt.show()
'''