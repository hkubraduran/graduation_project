import pandas as pd
from bitir_all_data import feature_select, merge_selected_features, clean_null, flatten_list_string, preprocess_pipeline
"""def preprocess_pipeline(df):
    #df = feature_select(df)
    df = pd.read_csv("merged_selected_data2.csv")
    liste_kolonlar = ["diseaseClasses_DO", "diseaseClasses_MSH", "diseaseClasses_UMLS_ST", "diseaseClasses_HPO"]

    for col in liste_kolonlar:
        df[col] = df[col].apply(flatten_list_string)"""
#merge_selected_features("collected_unfiltered", "merged_selected_data2.csv")
#clean_null("merged_selected_data2.csv", "cleaned_file.csv", "null_file.csv")
"""
# Dosyayı oku
df = pd.read_csv("merged_selected_data2.csv")

# Pipeline uygula
df_clean = preprocess_pipeline(df)

# Sonucu kaydet
df_clean.to_csv("merged_selected_data2_unk.csv", index=False)
"""
df = pd.read_csv("merged_selected_data2_unk.csv")

df_clean = preprocess_pipeline(df)

df_clean.to_csv("final_clean_data2.csv", index=False)

print("Preprocessing tamamlandı, final_clean_data.csv oluşturuldu.")
