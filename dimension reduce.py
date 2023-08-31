import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
import umap

data = pd.read_csv("C:/Users/Betsy/Desktop/\ds project/analytics/distance and direction2.csv")
data.fillna(0, inplace=True)
#print(data.columns)
cleaned_data = data.dropna()
features = cleaned_data[['x', 'y', 'avg_all_distances', 'avg_closest_distances', 'direction_change','distance_change', 'avg_direction', 'avg_speed', 'avg_closest_direction', 'avg_closest_speed']]
standardized_features = (features - features.mean()) / features.std()
standardized_features[['frame_index', 'fish_index']] = data[['frame_index', 'fish_index']]  #x,y
#print(standardized_features)
outputpath = "C:/Users/Betsy/ds_csv/standardized_features3.csv"
standardized_features.to_csv(outputpath, sep = ',', index = True, header = True)

local_features = standardized_features[['frame_index', 'fish_index', 'x', 'y','avg_closest_distances','avg_closest_direction', 'avg_closest_speed']]
global_features = standardized_features[['frame_index', 'fish_index', 'x', 'y','avg_all_distances','avg_direction', 'avg_speed']]
pivoted_local = local_features.pivot(index='fish_index', columns='frame_index', values=['x', 'y','avg_closest_distances','avg_closest_direction', 'avg_closest_speed'])
pivoted_global = global_features.pivot(index='fish_index', columns='frame_index', values=['x', 'y','avg_all_distances','avg_direction', 'avg_speed'])
pivoted_local.columns = [f'Frame {col[1]} {col[0]}' for col in pivoted_local.columns]
pivoted_global.columns = [f'Frame {col[1]} {col[0]}' for col in pivoted_global.columns]
#print(pivoted_local)
#outputpath = "C:/Users/Betsy/ds_csv/pivot_local3.csv"
#pivoted_local.to_csv(outputpath, sep = ',', index = True, header = True)

#outputpath = "C:/Users/Betsy/ds_csv/pivot_global3.csv"
#pivoted_global.to_csv(outputpath, sep = ',', index = True, header = True)

pivoted_local.fillna(0, inplace=True)
pivoted_global.fillna(0, inplace=True)

# pca
pca_local = PCA(n_components=10)
reduced_local_features = pca_local.fit_transform(pivoted_local)
variance_explained1 = pca_local.explained_variance_ratio_
cumulative_variance1 = np.cumsum(variance_explained1)

pca_global = PCA(n_components=10)
reduced_global_features = pca_global.fit_transform(pivoted_global)
variance_explained2 = pca_global.explained_variance_ratio_
cumulative_variance2 = np.cumsum(variance_explained2)

####################################
'''
# 创建Scree Plot
plt.figure(figsize=(12, 5))

# PCA1
plt.subplot(1, 2, 1)
sns.barplot(x=np.arange(1, len(variance_explained1)+1), y=variance_explained1, color='blue', ax=plt.gca())
plt.plot(np.arange(0, len(variance_explained1)), cumulative_variance1, marker='o', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Scree Plot - local')
plt.grid(True)
print(variance_explained1)
# PCA2
plt.subplot(1, 2, 2)
sns.barplot(x=np.arange(1, len(variance_explained2)+1), y=variance_explained2, color='blue', ax=plt.gca())
plt.plot(np.arange(0, len(variance_explained2)), cumulative_variance2, marker='o', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Scree Plot - global')
plt.grid(True)
print(variance_explained2)
plt.tight_layout()
plt.show()



reduction = pd.DataFrame()
# 将降维后的数据添加到data中
reduction.index = pivoted_local.index
reduction['Local_PC1'] = reduced_local_features[:, 0]
reduction['Local_PC2'] = reduced_local_features[:, 1]
reduction['Local_PC3'] = reduced_local_features[:, 2]

reduction['global_PC1'] = reduced_global_features[:, 0]
reduction['global_PC2'] = reduced_global_features[:, 1]
reduction['global_PC3'] = reduced_global_features[:, 2]
#print(reduction.index)
'''
'''
# 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 绘制local数据的散点图
sns.scatterplot(data=reduction, x='Local_PCA1', y='Local_PCA2', hue=reduction.index, marker='o', ax=axes[0], legend=False)
axes[0].set_title('PCA Visualization for Local Data')
axes[0].set_xlabel('Local PCA1')
axes[0].set_ylabel('Local PCA2')

# 绘制global数据的散点图
sns.scatterplot(data=reduction, x='global_PCA1', y='global_PCA2', hue=reduction.index, marker='o', ax=axes[1], legend=False)
axes[1].set_title('PCA Visualization for Global Data')
axes[1].set_xlabel('Global PCA1')
axes[1].set_ylabel('Global PCA2')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
'''
#outputpath = "C:/Users/Betsy/Desktop/\ds project/analytics/pca_reduction3.csv"
#reduction.to_csv(outputpath, sep = ',', index = True, header = True)

#print(pivoted_global)
'''
# tsne
X_global = pivoted_global[1:]
y_global = pivoted_global.index

X_local = pivoted_local[1:]
y_local  = pivoted_local.index
# 创建DataFrame
df_local = pivoted_local
df_global = pivoted_global
#print(df_local)

# 去除含有缺失值的行
#df_local.dropna(inplace=True)

# 使用t-SNE进行降维
tsne_global = TSNE(init='pca', learning_rate='auto')
reduced_features_global = tsne_global.fit_transform(df_global)
tsne_local = TSNE(init='pca', learning_rate='auto')
reduced_features_local = tsne_local.fit_transform(df_local)



# 创建降维后的DataFrame
reduced_df_local = pd.DataFrame(data=reduced_features_local, columns=['t-SNE 1', 't-SNE 2'], index=df_local.index)
reduced_df_global = pd.DataFrame(data=reduced_features_global, columns=['t-SNE 1', 't-SNE 2'], index=df_global.index)
# 绘制散点图
# 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 绘制local数据的t-SNE图
sns.scatterplot(data=reduced_df_local, x='t-SNE 1', y='t-SNE 2', hue=reduced_df_local.index, marker='o', ax=axes[0])
axes[0].set_title('t-SNE Visualization (Local Data)')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
axes[0].legend(title='Index')

# 绘制global数据的t-SNE图
sns.scatterplot(data=reduced_df_global, x='t-SNE 1', y='t-SNE 2', hue=reduced_df_global.index, marker='x', ax=axes[1])
axes[1].set_title('t-SNE Visualization (Global Data)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].legend(title='Index')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
'''

###################################################
'''
# 绘制局部数据的散点图（左侧）
plt.figure(figsize=(5, 3))
plt.scatter(reduction['Local_PCA1'], reduction['Local_PCA2'], color='blue', label='Local Data')

# 绘制整体数据的散点图（右侧）
plt.scatter(reduction['global_PCA1'], reduction['global_PCA2'], color='red', label='Global Data', marker='x')

# 添加标题和标签
plt.title("Scatter Plot of Fishes")
plt.xlabel("PCA1")
plt.ylabel("PCA2")

# 显示图形
plt.show()
'''
###########################################

#the scatter plot for pc2 and 
'''
plt.figure(figsize=(8, 6))
color = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
local_scatter = plt.scatter(reduction['Local_PC2'], reduction['Local_PC3'], c=color, cmap='viridis', marker='o', label='Local Data')

global_scatter = plt.scatter(reduction['global_PC2'], reduction['global_PC3'], c=color, cmap='viridis', marker='x', label='Global Data')

cbar = plt.colorbar(global_scatter)
cbar.set_label("Index")

for index in reduction.index:
    local_point = reduction[reduction.index == index]
    global_point = reduction[reduction.index == index]
    plt.plot([local_point['Local_PC2'].values[0], global_point['global_PC2'].values[0]],
             [local_point['Local_PC3'].values[0], global_point['global_PC3'].values[0]], color='g', linestyle='dashed')

plt.title("Scatter Plot for PC2 and PC3")
plt.xlabel("PC2")
plt.ylabel("PC3")

plt.legend()

plt.show()
'''
########################################################
'''
# the scatter plot for pca1, pca2 and pca3
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i, (local, global_) in enumerate([('PC1', 'PC1'), ('PC2', 'PC2')]):
    ax = axes[i]
    
    ax.scatter(reduction[f'Local_{local}'], reduction[f'global_{global_}'], color='blue', label='Dataset')
    
    slope, intercept, _, _, _ = linregress(reduction[f'Local_{local}'], reduction[f'global_{global_}'])
    
    ax.plot(reduction[f'Local_{local}'], slope * reduction[f'Local_{local}'] + intercept, color='red', linestyle='dashed')
    
    ax.set_title(f"{local} Comparison")# with Outlier
    ax.set_xlabel(f"Local {local}")
    ax.set_ylabel(f"Global {global_}")

plt.tight_layout()

plt.legend()

plt.show()
'''
##################################################
'''
reduction = reduction[reduction['Local_PC1'] < 50]
plt.figure(figsize=(5, 5))
# 绘制散点图
plt.scatter(reduction['Local_PC1'], reduction['global_PC1'], color='blue', label='Dataset')

# 进行线性拟合
slope, intercept, _, _, _ = linregress(reduction['Local_PC1'], reduction['global_PC1'])

# 绘制拟合的直线
plt.plot(reduction['Local_PC1'], slope * reduction['Local_PC1'] + intercept, color='red')

# 添加标题和标签
plt.title("PC1 Comparison without Outlier")
plt.xlabel("Local PC1")
plt.ylabel("Global PC1")

# 显示图形
plt.show()
'''

# UMAP

df_local = pivoted_local
df_global = pivoted_global
# UMAP降维
umap_global = umap.UMAP(n_neighbors=20, min_dist=0.1)
reduced_features_global_umap = umap_global.fit_transform(df_global)

umap_local = umap.UMAP(n_neighbors=20, min_dist=0.1)
reduced_features_local_umap = umap_local.fit_transform(df_local)

# 创建降维后的DataFrame
reduced_df_local_umap = pd.DataFrame(data=reduced_features_local_umap, columns=['UMAP 1', 'UMAP 2'], index=df_local.index)
reduced_df_global_umap = pd.DataFrame(data=reduced_features_global_umap, columns=['UMAP 1', 'UMAP 2'], index=df_global.index)
# 绘制UMAP结果的散点图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# 绘制local数据的UMAP图
sns.scatterplot(data=reduced_df_local_umap, x='UMAP 1', y='UMAP 2', hue=reduced_df_local_umap.index, marker='o', ax=axes[0], legend = False)
for index, row in reduced_df_local_umap.iterrows():
    axes[0].annotate(index, (row['UMAP 1'], row['UMAP 2']), textcoords="offset points", xytext=(0, 10), ha='center')
axes[0].set_title('UMAP Visualization for Local Data')
axes[0].set_xlabel('dimension 1')
axes[0].set_ylabel('dimension 2')
# 绘制global数据的UMAP图
sns.scatterplot(data=reduced_df_global_umap, x='UMAP 1', y='UMAP 2', hue=reduced_df_global_umap.index, marker='o', ax=axes[1], legend = False)
for index, row in reduced_df_global_umap.iterrows():
    axes[1].annotate(index, (row['UMAP 1'], row['UMAP 2']), textcoords="offset points", xytext=(0, 10), ha='center')
axes[1].set_title('UMAP Visualization for Global Data')
axes[1].set_xlabel('dimension 1')
axes[1].set_ylabel('dimension 2')

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
# 计算global数据框中每个点到其他点的距离
global_distances = pd.DataFrame(cdist(reduced_df_global_umap.values, reduced_df_global_umap.values), 
                                index=reduced_df_global_umap.index, columns=reduced_df_global_umap.index)

# 计算local数据框中每个点到其他点的距离
local_distances = pd.DataFrame(cdist(reduced_df_local_umap.values, reduced_df_local_umap.values), 
                               index=reduced_df_local_umap.index, columns=reduced_df_local_umap.index)
'''
# 打印结果
print("Global Distances:")
print(global_distances)

print("Local Distances:")
print(local_distances)
'''
# 对global_distances进行排序
sorted_global_distances = global_distances.rank()
outputpath = "C:/Users/Betsy/Desktop/\ds project/analytics/order_global1.csv"
sorted_global_distances.to_csv(outputpath, sep = ',', index = True, header = True)
print(sorted_global_distances)
# 对local_distances进行排序
sorted_local_distances = local_distances.rank()
outputpath = "C:/Users/Betsy/Desktop/\ds project/analytics/order_local1.csv"
sorted_local_distances.to_csv(outputpath, sep = ',', index = True, header = True)
corr, p_value = spearmanr(global_distances.values.flatten(), local_distances.values.flatten())
#corr = sorted_global_distances.corrwith(sorted_local_distances, axis = 0, method = 'spearman')
# 比较两个表的相似性
comparison = np.zeros_like(global_distances)
print(corr)
# 遍历global_distances和local_distances的每个元素
for i in range(global_distances.shape[0]):
    for j in range(global_distances.shape[1]):
        # 检查local_distances中的距离是否在global_distances的正负2范围内
        if abs(global_distances.iloc[i, j] - local_distances.iloc[i, j]) <= 2:
            # 将相应位置设为1
            comparison[i, j] = 1
#print(comparison)
total_elements = comparison.size
ones_count = np.count_nonzero(comparison == 1)
percentage = ones_count / total_elements * 100
#print(percentage)
# 调整子图之间的间距
plt.tight_layout()
# 显示图形
plt.show()
