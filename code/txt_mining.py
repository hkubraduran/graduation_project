from bitir_all_data import clean_text, frequency_analysis, tf_idf, gene_disease, network_analysis
import pandas as pd
"""
##DATA PREPROCESSİNG
# nltk.download('stopwords')
# nltk.download('wordnet')
df = pd.read_csv("final_clean_data2.csv")
# Örnek uygulama
columns_to_clean = [
    "diseaseName",
    "diseaseClasses_DO",
    "diseaseClasses_MSH",
    "diseaseClasses_UMLS_ST",
    "diseaseClasses_HPO"
]
# df["diseaseName_clean"] = df["diseaseName"].apply(clean_text)
# for col in columns_to_clean:
#     df[f"{col}_clean"] = df[col].fillna("").apply(clean_text)

for col in columns_to_clean:
    df[col] = df[col].fillna("").apply(clean_text)
df.to_csv("cleaned_dataset_overwrite.csv", index=False)
print("Temizlenmiş veri başarıyla 'cleaned_dataset_overwrite.csv' dosyasına kaydedildi.")
# Temizlik sonrası ilk 10 satırı göster
# print(df[["diseaseName", "diseaseName_clean"]].head(10))

# (isteğe bağlı) temizlenmiş veriyi dışa aktar
# df.to_csv("cleaned_data_with_diseaseName.csv", index=False)
"""

##FREQUENCY
# Veriyi yükle
df = pd.read_csv("filtered_dataset_0.4.csv")
# frequency_analysis(df)

##TF-IDF
# tf_idf(df)

##CO-OCCURANCE
# gene_disease(df)

##network analysis
network_analysis(df)