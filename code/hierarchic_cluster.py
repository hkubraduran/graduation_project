from bitir_all_data import hierarchical_clustering, hierarchical_clustering_efficient, dbscan_clustering_analysis
import pandas as pd

# df = pd.read_csv("final_clean_data2.csv")
# df = hierarchical_clustering(df, n_clusters=5)
# df.to_csv("final_with_hc.csv", index=False)
# df = hierarchical_clustering_efficient(df, n_clusters=5)
# df.to_csv("final_clean_data_with_agg_clusters.csv", index=False)
# X = df[["score", "ei", "numPMIDs"]].fillna(0)
# X = df[["score", "ei"]].fillna(0)
# # df = dbscan_clustering_analysis(df, X, eps=0.5, min_samples=5)
# df = dbscan_clustering_analysis(df, X, eps=0.3, min_samples=10)
# df = dbscan_clustering_analysis(df, X, eps=0.3, min_samples=5)
# df = dbscan_clustering_analysis(df, X, eps=0.3, min_samples=10)
#df.to_csv("final_clean_data_with_dbscan.csv", index=False)
#0,3-5 0,25-3 0,35-5

df = pd.read_csv("filtered_dataset_0.4.csv")
X = df[["score", "ei"]].fillna(0)
df = dbscan_clustering_analysis(df, X, eps=0.3, min_samples=5)
df.to_csv("filtered_dbscan_0.3-5.csv", index=False)
# df = hierarchical_clustering(df, n_clusters=8)
# df.to_csv("hierarchic_cluster_c8.csv", index=False)
