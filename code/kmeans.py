from bitir_all_data import kmeans_cluster, kmeans_3d, elbow, silhouette_analysis, correlation_heatmap, boxplot_analysis, histogram_analysis, kmeans_full_analysis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

"""
#görselleştirme
plt.figure(figsize=(15, 4))
k_list = [2, 3, 4]

for idx, k in enumerate(k_list):
    plt.subplot(1, 3, idx+1)
    sns.scatterplot(data=df, x="score", y="ei", hue=f"cluster_{k}", palette="tab10")
    plt.title(f"K-Means Kümeleme (k = {k})")
    plt.xlabel("score")
    plt.ylabel("ei")

plt.tight_layout()
plt.show()
"""
# Veri yükle
# df = pd.read_csv("final_clean_data2.csv")
df = pd.read_csv("filtered_dataset_0.4.csv")
# df, _ = kmeans_cluster(df)
# correlation_heatmap(df)
# boxplot_analysis(df)
# histogram_analysis(df)
#df = kmeans_3d(df)
# df = elbow(df)
# k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# df = silhouette_analysis(df, k_list)

# Sadece score ve ei ile 2D analiz
df = kmeans_full_analysis(df, features=["score", "ei"], k_list=[8], use_3d=False)

# 3D analiz dahil olsun istersen:
df = kmeans_full_analysis(df, features=["score", "ei", "numPMIDs"], k_list=[8], use_3d=True)
