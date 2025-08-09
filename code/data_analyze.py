from bitir_all_data import data_analysis, pairplot_analysis, cluster_summary, pca_visualization, correlation_heatmap, boxplot_analysis, histogram_analysis, kmeans_cluster
import pandas as pd
#df = pd.read_csv("final_clean_data2.csv")
#data_analysis(df)
#df, _ = kmeans_cluster(df)  # k-means çalıştır, df'ye 'cluster_5' alanını ekle
#df.to_csv("final_clean_data_with_clusters.csv", index=False)

# df = pd.read_csv("final_clean_data_with_clusters.csv")

# pairplot_analysis(df)
# cluster_summary(df)
# pca_visualization(df)
# correlation_heatmap(df)
# boxplot_analysis(df)
# histogram_analysis(df)

df = pd.read_csv("filtered_dataset_0.4.csv")
data_analysis(df)