import pandas as pd
df = pd.read_csv("gene_list_final.csv")

df_sampled = df.sample(n=3000, random_state=42)

df_sampled.to_csv("gene_sample_3000.csv", index=False)  
