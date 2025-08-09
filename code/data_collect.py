
import pandas as pd
from bitir_all_data import query_disgenet_all_pages, batch_all_genes, data_filtering
# api_key = "4b8becb4-3ca2-4726-a9e6-066b36c6fb28"  # Kendi API key'in buraya
# gene_id = 3553
# output_folder = "collected_unfiltered"
# gene_file_path = "gene_list_final_copy.csv"  # Yüklediğin dosya

#q_dg = query_disgenet_top25(gene_id, api_key, output_path=f"gene_{gene_id}_top25.csv")

#batch_query_disgenet(gene_file_path, api_key, output_folder="collected_unfiltered", start_index=746)

#query_disgenet_all_pages(gene_id, api_key, output_folder)

# batch_all_genes(gene_file_path, api_key, output_folder)
df = pd.read_csv("cleaned_dataset_overwrite.csv")
data_filtering(df)