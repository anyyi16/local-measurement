import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
import math

data = pd.read_csv("C:/Users/Betsy/ds_csv/model_direction3.csv")
data_real = pd.read_csv("C:/Users/Betsy/Desktop/ds project/analytics/distance and direction2.csv")
data = data[:-17]
data.dropna(inplace = True)
data_real.dropna(inplace = True)
def histogram_mse(hist1, hist2):

    hist1 = sorted(hist1)
    hist2 = sorted(hist2)
    mean1 = np.mean(hist1)
    mean2 = np.mean(hist2)

    norm_hist1 = hist1 / np.sum(hist1)
    norm_hist2 = hist2 / np.sum(hist2)

    mse = np.mean((norm_hist1 - norm_hist2) ** 2)
    
    return mse

def cross_entropy(h1, h2):
    return -np.sum(h1 * np.log(h2))

def correlation_coefficient(h1, h2):
    cov = np.cov(h1, h2)[0, 1]
    std1 = np.std(h1)
    std2 = np.std(h2)
    return cov / (std1 * std2)

def mutual_information(h1, h2):
    joint_prob = np.outer(h1, h2)
    joint_prob_normalized = joint_prob / np.sum(joint_prob)
    marginal_prob1 = np.sum(joint_prob_normalized, axis=1)
    marginal_prob2 = np.sum(joint_prob_normalized, axis=0)
    mi = np.sum(joint_prob_normalized * np.log(joint_prob_normalized / (marginal_prob1[:, None] * marginal_prob2[None, :])))
    return mi


# 创建一个2x2的子图网格
fig = plt.figure(figsize=(1, 2))
gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])

# 绘制左上角的数据分布图
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(data["avg_all_distances"], kde=True, ax=ax1)
ax1.set_title("Distance for all fish(model)")
ax1.set_xlabel("value")
ax1.set_ylabel("num")

ax1 = fig.add_subplot(gs[0, 1])
sns.histplot(data_real["avg_all_distances"], kde=True, ax=ax1)
ax1.set_title("Real-life")
ax1.set_xlabel("value")
ax1.set_ylabel("num")

mean_value1 = np.mean(data["avg_all_distances"])
std_deviation1 = np.std(data["avg_all_distances"])

mse1 = histogram_mse(data_real["avg_all_distances"], data_real["avg_closest_distances"])
mse1_1 = histogram_mse(data_real["avg_all_distances"], data["avg_all_distances"])
mse1_2 = histogram_mse(data["avg_closest_distances"], data_real["avg_closest_distances"])
'''
ce1 = cross_entropy(data["avg_all_distances"], data["avg_closest_distances"])
corr1 = correlation_coefficient(data["avg_all_distances"], data["avg_closest_distances"])
mi1 = mutual_information(data["avg_all_distances"], data["avg_closest_distances"])
print("mutual_information:", mi1)
print("Correlation Coefficient:", corr1)
print("Cross-Entropy:", ce1)
'''
print("Real MSE:", mse1)
print("Global MSE:", mse1_1)
print("Local MSE:", mse1_2)

# 绘制右上角的数据分布图
ax2 = fig.add_subplot(gs[0, 2])
sns.histplot(data["avg_closest_distances"], kde=True, ax=ax2)
ax2.set_title("Distance for 5 closest fish(model)")
ax2.set_xlabel("value")
ax2.set_ylabel("num")
mean_value2 = np.mean(data["avg_closest_distances"])
std_deviation2 = np.std(data["avg_closest_distances"])

ax2 = fig.add_subplot(gs[0, 3])
sns.histplot(data_real["avg_closest_distances"], kde=True, ax=ax2)
ax2.set_title("Real-life")
ax2.set_xlabel("value")
ax2.set_ylabel("num")
print(f"平均距离{mean_value1}，标准差{std_deviation1}")
print(f"最小平均距离{mean_value2}，标准差{std_deviation2}")

# 绘制左下角的数据分布图
ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(data["avg_direction"], kde=True, ax=ax3)
ax3.set_title("Direction for all fish(model)")
ax3.set_xlabel("value")
ax3.set_ylabel("num")

ax3 = fig.add_subplot(gs[1, 1])
sns.histplot(data_real["avg_direction"], kde=True, ax=ax3)
ax3.set_title("Real-life")
ax3.set_xlabel("value")
ax3.set_ylabel("num")
mean_value3 = np.mean(data["avg_direction"])
std_deviation3 = np.std(data["avg_direction"])
mse2 = histogram_mse(data_real["avg_direction"], data_real["avg_closest_direction"])
mse2_1 = histogram_mse(data_real["avg_direction"], data["avg_direction"])
mse2_2 = histogram_mse(data_real["avg_closest_direction"], data["avg_closest_direction"])
'''
ce2 = cross_entropy(data["avg_direction"], data["avg_closest_direction"])
corr2 = correlation_coefficient(data["avg_direction"], data["avg_closest_direction"])
mi2 = mutual_information(data["avg_direction"], data["avg_closest_direction"])
print("mutual_information:", mi2)
print("Correlation Coefficient:", corr2)
print("Cross-Entropy:", ce2)
'''
print("Real MSE:", mse2)
print("Global MSE:", mse2_1)
print("Local MSE:", mse2_2)
print(f"平均方向{mean_value3}，标准差{std_deviation3}")

# 绘制右下角的数据分布图

ax4 = fig.add_subplot(gs[1, 2])
sns.histplot(data["avg_closest_direction"], kde=True, ax=ax4)
ax4.set_title("Direction for 5 closest fish(model)")
ax4.set_xlabel("value")
ax4.set_ylabel("num")
mean_value4 = np.mean(data["avg_closest_direction"])
std_deviation4 = np.std(data["avg_closest_direction"])

print(f"最小平均方向{mean_value4}，标准差{std_deviation4}")
data["avg_speed"] = sorted(data["avg_speed"])

ax6 = fig.add_subplot(gs[1, 3])
sns.histplot(data_real["avg_closest_direction"], kde=True, ax=ax6)
ax6.set_title("Real-life")
ax6.set_xlabel("value")
ax6.set_ylabel("num")
# 绘制左下角的数据分布图
ax5 = fig.add_subplot(gs[2, 0])
sns.histplot(data["avg_speed"][400:], kde=True, ax=ax5)
ax5.set_title("Speed for all fish(model)")
ax5.set_xlabel("value")
ax5.set_ylabel("num")
mean_value5 = np.mean(data["avg_speed"])
std_deviation5 = np.std(data["avg_speed"])
mse3 = histogram_mse(data_real["avg_speed"], data_real["avg_closest_speed"])
mse3_1 = histogram_mse(data_real["avg_speed"], data["avg_speed"])
mse3_2 = histogram_mse(data["avg_closest_speed"], data_real["avg_closest_speed"])

ax7 = fig.add_subplot(gs[2, 1])
sns.histplot(data_real["avg_speed"], kde=True, ax=ax7)
ax7.set_title("Real-life")
ax7.set_xlabel("value")
ax7.set_ylabel("num")
'''
ce3 = cross_entropy(data["avg_speed"], data["avg_closest_speed"])
corr3 = correlation_coefficient(data["avg_speed"], data["avg_closest_speed"])
mi3 = mutual_information(data["avg_speed"], data["avg_closest_speed"])
print("mutual_information:", mi3)
print("Correlation Coefficient:", corr3)
print("Cross-Entropy:", ce3)
'''
print("Real MSE:", mse3)
print("Global MSE:", mse3_1)
print("Local MSE:", mse3_2)

print(f"平均速度{mean_value5}，标准差{std_deviation5}")

# 绘制右下角的数据分布图
ax9 = fig.add_subplot(gs[2, 2])
sns.histplot(data["avg_closest_speed"], kde=True, ax=ax9)
ax9.set_title("Speed for 5 closest fish(model)")
ax9.set_xlabel("value")
ax9.set_ylabel("num")
mean_value6 = np.mean(data["avg_closest_speed"])
std_deviation6 = np.std(data["avg_closest_speed"])

# 绘制右下角的数据分布图
ax8 = fig.add_subplot(gs[2, 3])
sns.histplot(data_real["avg_closest_speed"], kde=True, ax=ax8)
ax8.set_title("Real-life")
ax8.set_xlabel("value")
ax8.set_ylabel("num")
mean_value6 = np.mean(data["avg_closest_speed"])
std_deviation6 = np.std(data["avg_closest_speed"])

print(f"最小平均速度{mean_value6}，标准差{std_deviation6}")

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.5, hspace = 1)

# 显示图表
plt.show()
'''
#plt.hist(data["distance_change"])
#plt.show()
speed_list = []
speed_chosen = data["distance_change"]
for i in range(30):
    temp = np.random.choice(speed_chosen)
    speed_list.append(temp)

print(speed_list)
'''