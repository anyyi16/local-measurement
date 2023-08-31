import numpy as np
from numpy.linalg import *
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# 从 CSV 文件中读取 positions_history
positions_history = pd.read_csv('C:/Users/Betsy/ds_csv/try3.csv')

# 创建一个空的 DataFrame 列表
frame_index = []
fish = []
x_list = []
y_list = []
# 遍历 positions_history 中的每一行
for index, row in positions_history.iterrows():
    # 创建一个临时的 DataFrame 用于存储当前行的数据
    
    # 遍历每条鱼的索引（0 到 29）
    for fish_index in range(30):
        x = row[f'agent{fish_index}x']
        y = row[f'agent{fish_index}y']
        #print(x)
        frame_index.append(row['times'])
        fish.append(fish_index)
        x_list.append(x)
        y_list.append(y)

result_df = pd.DataFrame()
result_df["frame_index"] = frame_index
result_df["fish_index"] = fish
result_df["x"] = x_list
result_df["y"] = y_list
#print(result_df)
# 将转换后的 DataFrame 保存为 CSV 文件
result_df.to_csv('C:/Users/Betsy/ds_csv/transformed_data3.csv', index=False)