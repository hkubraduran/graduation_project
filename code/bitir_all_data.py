import requests
import pandas as pd
import time
import json
import os
import math
import numpy as np
import ast
import seaborn as sns
import matplotlib as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from networkx.algorithms import bipartite 

def query_disgenet_all_pages(gene_id, api_key, output_folder):
    
    base_url = "https://api.disgenet.com/api/v1/gda/summary"

    headers = {
        'Authorization': api_key,
        'accept': 'application/json'
    }

    all_results = []
    page_number = 0
    page_size = 100  # DisGeNET API'nin maksimum desteklediği sayfa boyutu bu, 100 uygundur.

    # İlk sorgu: Kaç kayıt olduğunu öğrenmek için
    params = {
        "gene_ncbi_id": gene_id,
        "page_number": page_number,
        "page_size": page_size
    }

    while True:
        response = requests.get(base_url, params=params, headers=headers, verify=False)

        if response.status_code == 429:
            wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 5))
            print(f"Rate limit aşıldı. {wait_time} saniye bekleniyor...")
            time.sleep(wait_time)
            continue

        if not response.ok:
            print(f"Hata oluştu! Status code: {response.status_code}")
            return None

        break

    data = response.json()
    total_elements = data.get("paging", {}).get("totalElements", 0)
    total_pages = math.ceil(total_elements / page_size)

    print(f"\nGen {gene_id} için toplam {total_elements} ilişki bulundu. Toplam sayfa: {total_pages}")

    # İlk sayfadaki veriyi ekle
    all_results.extend(data.get("payload", []))

    # Kalan sayfaları çek
    for page in range(1, total_pages):
        params["page_number"] = page
        while True:
            response = requests.get(base_url, params=params, headers=headers, verify=False)

            if response.status_code == 429:
                wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 5))
                print(f"Rate limit aşıldı. {wait_time} saniye bekleniyor...")
                time.sleep(wait_time)
                continue

            if not response.ok:
                print(f"Hata oluştu! Status code: {response.status_code}")
                break

            break

        page_data = response.json()
        all_results.extend(page_data.get("payload", []))
        time.sleep(0.3) 
    print(f"Toplam çekilen kayıt: {len(all_results)}")

    if not all_results:
        print(f"Gen {gene_id} için ilişki bulunamadı.")
        return None

    df = pd.DataFrame(all_results)
    os.makedirs(output_folder, exist_ok=True)

    csv_filename = os.path.join(output_folder, f"gene_{gene_id}.csv")
    json_filename = os.path.join(output_folder, f"gene_{gene_id}.json")

    df.to_csv(csv_filename, index=False)
    print(f"{len(df)} kayıt CSV dosyasına yazıldı: {csv_filename}")

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump({"payload": all_results}, f, ensure_ascii=False, indent=4)
    print(f"JSON dosyası kaydedildi: {json_filename}")

    return df

# Toplu işleyen kısım
def batch_all_genes(gene_file_path, api_key, output_folder):
    df = pd.read_csv(gene_file_path)
    
    total_gen_count = len(df)
    print(f"Toplam {total_gen_count} gen bulundu.")
    
    # 'geneID' kolonunu okuyup listeye çeviriyoruz
    gene_ids = df['geneID'].tolist()

    print(f"Toplam {len(gene_ids)} gen bulunuyor. Veri toplanıyor...")

    for idx, gene_id in enumerate(gene_ids, 0):
        #print(f"\n[{idx}/{len(gene_ids)}] Gen ID: {gene_id} işleniyor...")
        print(f"\n[{idx}/{total_gen_count}] Gen ID: {gene_id} işleniyor...")
        try:
            query_disgenet_all_pages(gene_id, api_key, output_folder)
            time.sleep(0.5)  # API'yi çok yormamak için küçük gecikme
        except Exception as e:
            print(f"{gene_id} için hata oluştu: {e}")
            break
def data_filtering(df):
    # Filtreleme
    filtered_df = df[(df['ei'] >= 0.6) & (df['score'] >= 0.4) & (df['numPMIDs'] >= 2)]

    # Kalan veri sayısı
    print("Filtrelenmiş veri sayısı:", len(filtered_df))

    # Kaydetme
    filtered_df.to_csv("filtered_dataset_0.4.csv", index=False)

def feature_select(df):
    columns_to_keep = [
        "symbolOfGene",
        "geneNcbiID",
        "diseaseUMLSCUI",
        "diseaseName",
        "diseaseClasses_DO",
        "diseaseClasses_MSH",
        "diseaseClasses_UMLS_ST",
        "diseaseClasses_HPO",
        "score",
        "ei",
        "numPMIDs"
    ]
    
    available_cols = [col for col in columns_to_keep if col in df.columns]
    return df[available_cols]

def merge_selected_features(folder_path, output_file):
    all_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                df_selected = feature_select(df)
                all_data.append(df_selected)
            except Exception as e:
                print(f"{filename} okunurken hata oluştu: {e}")
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Tüm veriler '{output_file}' dosyasına kaydedildi. Toplam kayıt sayısı: {len(merged_df)}")
    else:
        print("Hiçbir dosya okunamadı veya uygun veri bulunamadı.")

def clean_null(file_path, cleaned_file, null_file):
    """
    Tüm kolonlarda eksik (NaN veya '[]') verileri tespit eder.
    Eksik ve eksiksiz kayıtları ayrı dosyalara kaydeder.
    Her kolon için eksik olup olmadığını if-else yapısıyla raporlar.
    """

    df = pd.read_csv(file_path)

    eksik_mask = df.isna() | (df == "[]")
    is_null = np.any(eksik_mask.values, axis=1)

    null_df = df[is_null]
    clean_df = df[~is_null]

    print("\nKolon bazında eksik kontrolü:\n")
    for col in df.columns:
        null_count = df[col].isna().sum() + (df[col] == "[]").sum()
        
        if null_count > 0:
            print(f"{col} kolonunda {null_count} eksik veya [] değer var.")
        else:
            print(f"{col} kolonunda eksik veya [] değer YOK.")

    if len(null_df) > 0:
        null_df.to_csv(null_file, index=False)
        print(f"\n{len(null_df)} eksik kayıt '{null_file}' dosyasına kaydedildi.")
    else:
        print("\nEksik kayıt bulunmadı. Eksik dosyası oluşturulmadı.")

    if len(clean_df) > 0:
        clean_df.to_csv(cleaned_file, index=False)
        print(f"{len(clean_df)} temiz kayıt '{cleaned_file}' dosyasına kaydedildi.")
    else:
        print("Temiz kayıt bulunmadı.")

# Liste stringlerini düzleştiren fonksiyon
def flatten_list_string(s):
    try:
        liste = ast.literal_eval(s)
        if isinstance(liste, list):
            return ", ".join(liste)
    except:
        pass
    return s

def convert_numeric(df):
    numeric_cols = ["score", "ei", "numPMIDs"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df 

def clean_text(df):
    if "diseaseName" in df.columns:
        df["diseaseName"] = df["diseaseName"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    return df

def preprocess_pipeline(df):
    """
    liste_kolonlar = ["diseaseClasses_DO", "diseaseClasses_MSH", "diseaseClasses_UMLS_ST", "diseaseClasses_HPO"]

    for col in liste_kolonlar:
        if col in df.columns:
            df[col] = df[col].replace("[]", "unknown")
            df[col] = df[col].apply(flatten_list_string)
     """
    #df = convert_numeric(df)
    df = clean_text(df)        
    return df  

def data_analysis(df):
    
    # Sayısal alanların genel dağılımı
    print(df.describe())

    # Eksik veya unknown oranları
    print((df.isna().sum() / len(df)) * 100)
    print((df == "unknown").sum())
    
    # Score dağılımı
    sns.histplot(df["score"], bins=20, kde=True)
    plt.title("Score Dağılımı")
    plt.show()

    # ei dağılımı
    sns.histplot(df["ei"], bins=20, kde=True)
    plt.title("ei Dağılımı")
    plt.show()

    
    # Sık geçen diseaseClasses_DO ilk 20
    print(df["diseaseClasses_DO"].value_counts().head(20))
    
    # Sık geçen diseaseName ilk 20
    top_diseases = df["diseaseName"].value_counts().head(20)
    print(top_diseases)
    sns.barplot(x=top_diseases.values, y=top_diseases.index)
    plt.title("En Sık Geçen DiseaseName'ler")
    plt.xlabel("Frekans")
    plt.ylabel("Hastalık")
    plt.show()

    #yoğunluk haritası 
    plt.figure(figsize=(8,6))
    sns.kdeplot(data=df, x="score", y="ei", fill=True, cmap="Blues", thresh=0.05)
    plt.title("Score - ei Yoğunluk Haritası")
    plt.xlabel("Score")
    plt.ylabel("ei")
    plt.show()
    
    #hexbin plot altıgen yoğunluk grafiği
    plt.figure(figsize=(8,6))
    plt.hexbin(df["score"], df["ei"], gridsize=25, cmap="Purples", mincnt=1)
    plt.colorbar(label="Veri Yoğunluğu")
    plt.title("Score - ei Altıgen Yoğunluk Haritası")
    plt.xlabel("Score")
    plt.ylabel("ei")
    plt.show()
    
    corr = df["score"].corr(df["ei"])
    print(f"Score ile ei arasındaki Pearson korelasyonu: {corr:.3f}")
    
    corr = df[["score", "ei", "numPMIDs"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Korelasyon Matrisi")
    plt.show()

    #dağılım grafiği scatterplot+renk kodlu yoğunluk
    df["log_numPMIDs"] = np.log1p(df["numPMIDs"])
    
    plt.figure(figsize=(8,6))
    # sns.scatterplot(data=df, x="score", y="ei", hue="numPMIDs", palette="viridis", size="log_numPMIDs", sizes=(20, 200))
    sns.scatterplot(data=df, x="score", y="ei", hue="log_numPMIDs", palette="viridis", size="log_numPMIDs", sizes=(20, 200))
    sns.scatterplot(data=df, x="score", y="ei", size="log_numPMIDs", sizes=(20, 200))
    plt.title("Score - ei Dağılımı (log(numPMIDs) göre renk ve büyüklük)")
    plt.xlabel("Score")
    plt.ylabel("ei")
    plt.legend(title="numPMIDs", loc="upper left")
    plt.show()

def kmeans_full_analysis(df, features, k_list, use_3d):
    """
    df: pandas dataframe
    features: kümeleme yapılacak sayısal sütunlar
    k_list: denenmek istenen k değerleri listesi
    use_3d: 3 değişken varsa ve 3D görselleştirme yapılmak isteniyorsa True yapılabilir
    """

    # Sayısal alanları seçip standartlaştır
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for idx, k in enumerate(k_list):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df[f"cluster_{k}"] = clusters

        print(f"\n=== k = {k} için küme dağılımı ===")
        print(df[f"cluster_{k}"].value_counts())
        print("-" * 40)

        # 2D scatter (ilk 2 feature üzerinden)
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="tab10", s=50)
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5, marker="X", label="Merkezler")
        plt.title(f"K-Means 2D Kümeleme (k = {k})")
        plt.xlabel(f"{features[0]} (scaled)")
        plt.ylabel(f"{features[1]} (scaled)")
        plt.legend()
        plt.show()

        # 3D scatter (isteğe bağlı)
        if use_3d and len(features) == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
                                 c=clusters, cmap="tab10", s=50)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
                       c="black", s=200, alpha=0.5, marker="X", label="Merkezler")
            ax.set_xlabel(f"{features[0]} (scaled)")
            ax.set_ylabel(f"{features[1]} (scaled)")
            ax.set_zlabel(f"{features[2]} (scaled)")
            ax.set_title(f"K-Means 3D Kümeleme (k = {k})")
            plt.legend(*scatter.legend_elements(), title="Küme")
            plt.show()

        # Pairplot (eğer 3 değişken varsa)
        if len(features) >= 2:
            prplt = sns.pairplot(df, vars=features, hue=f"cluster_{k}", palette="tab10")
            # Başlığı biraz daha yukarıya al ve font ayarlarını yap
            prplt.fig.suptitle(f"K={k} için Pairplot - Sayısal Değişkenler ve Küme Dağılımı", y=1.03, fontsize=14, fontweight="bold")
            # plt.suptitle(f"K={k} için Pairplot - Sayısal Değişkenler ve Küme Dağılımı", y=1.02)
            # Alt grafikler ile başlık arasında boşluk bırak
            plt.subplots_adjust(top=0.92)
            plt.show()

        # Küme bazlı özet tablo
        print(f"\nKüme Bazlı Ortalama Değerler (k = {k}):\n")
        cluster_summary = df.groupby(f"cluster_{k}")[features].mean().round(3)
        print(cluster_summary)

    return df

def elbow(df):
    # X = df[["score", "ei", "numPMIDs"]].fillna(0)
    X = df[["score", "ei"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # İnertia değerlerini saklayacağımız liste
    inertia_list = []
    k_values = list(range(1, 11))  

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia_list.append(kmeans.inertia_)
        #print(f"k: {k}",end=" ")
        #print(kmeans.inertia_)
    
    inertia_df = pd.DataFrame({"k": k_values, "inertia": inertia_list})
    print("\nInertia Değerleri Tablosu:\n")
    print(inertia_df)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_list, marker="o")
    plt.title("Elbow Yöntemi - Toplam Hata (Inertia) Grafiği")
    plt.xlabel("Küme Sayısı (k)")
    plt.ylabel("Toplam Hata (Inertia)")
    plt.xticks(k_values)
    plt.grid()
    plt.show()
    
    return inertia_df

def silhouette_analysis(df, k_list):
    """
    Silhouette skorlarını hesaplayıp grafik ve tablo oluşturur.

    Parametreler:
    - df: pandas dataframe
    - max_k: denenecek maksimum küme sayısı (minimum 2 olmalı)
    """

    # X = df[["score", "ei", "numPMIDs"]].fillna(0)
    X = df[["score", "ei"]].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_results = []
 
    for k in k_list:
        print(f"\nk = {k} için silhouette hesaplanıyor...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette skoru: {score:.4f}")
        silhouette_results.append({"k": k, "silhouette_score": score})

    silhouette_df = pd.DataFrame(silhouette_results)
    print("\nSilhouette Skor Tablosu:\n")
    print(silhouette_df)

    plt.figure(figsize=(8, 5))
    plt.plot(silhouette_df["k"], silhouette_df["silhouette_score"], marker="o")
    plt.title("Seçilen k Değerleri için Silhouette Skoru")
    plt.xlabel("Küme Sayısı (k)")
    plt.ylabel("Silhouette Skoru")
    plt.grid()
    plt.show()

    return silhouette_df

def hierarchical_clustering(df, n_clusters):

    X = df[["score", "ei"]].fillna(0)
    # X = df[["score", "ei", "numPMIDs"]].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hiyerarşik bağlantı matrisi (ward: varyans-minimizasyonu)
    Z = linkage(X_scaled, method='ward')

    # Dendrogram çizimi
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="level", p=10)
    plt.title("Hiyerarşik Kümeleme Dendrogramı")
    plt.xlabel("Veri Noktası")
    plt.ylabel("Uzaklık")
    plt.tight_layout()
    plt.show()

    df[f"cluster_hc_{n_clusters}"] = fcluster(Z, t=n_clusters, criterion='maxclust')

    print(f"{n_clusters} küme için dağılım:")
    print(df[f"cluster_hc_{n_clusters}"].value_counts())

    return df

def dbscan_clustering_analysis(df, X, eps, min_samples):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("[1/7] Veriler seciliyor ve standardize ediliyor...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[2/7] DBSCAN uygulanıyor...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    df["dbscan_cluster"] = labels  # -1 olanlar "gürültü"

    print("[3/7] DBSCAN Küme Dağılımı:")
    print(df["dbscan_cluster"].value_counts().sort_index())
    print("-" * 40)
    
    # 🔍 Gürültü Analizi
    noise = df[df["dbscan_cluster"] == -1]
    print(f"[3.1] Gürültü (outlier) sayısı: {len(noise)}")
    print(f"[3.2] Gürültü oranı: %{(len(noise) / len(df)) * 100:.4f}")
    if len(noise) > 0:
        print("[3.3] Gürültü verileri (ilk 5 satır):")
        print(noise[["score", "ei"]].head())
        
        print("\n[3.4] Gürültü verilerinin istatistiksel özeti:")
        print(noise[["score", "ei"]].describe())

        noise.to_csv("noise_data.csv", index=False)
        print("\n[3.5] Gürültü verileri 'noise_data.csv' olarak kaydedildi.")
    else:
        print("[3.3] Gürültü verisi bulunamadı.")
    print("-" * 40)
    
    print("[4/7] 2D Scatter Plot (score vs ei)...")
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="score", y="ei", hue="dbscan_cluster", palette="tab10", s=50)
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    plt.xlabel("score")
    plt.ylabel("ei")
    plt.legend(title="Küme")
    plt.tight_layout()
    plt.show()

    print("[5/7] Boxplot'lar çiziliyor...")
    for col in X.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="dbscan_cluster", y=col, palette="Set3")
        plt.title(f"{col} Dağılımı (Boxplot)")
        plt.xlabel("Küme")
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    print("[6/7] Histogramlar çiziliyor...")
    for col in X.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue="dbscan_cluster", kde=True, palette="Set2", multiple="stack")
        plt.title(f"{col} Histogramı (Küme bazlı)")
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()
    """
    print("[7/7] 3d scatter plot çiziliyor...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df["score"], df["ei"], df["numPMIDs"],
                         c=df["dbscan_cluster"], cmap="tab10", s=20)
    ax.set_xlabel("score")
    ax.set_ylabel("ei")
    ax.set_zlabel("numPMIDs")
    ax.set_title(f"DBSCAN Clustering (3D) - eps={eps}, min_samples={min_samples}")
    legend = ax.legend(*scatter.legend_elements(), title="Küme")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.show()
    """
    return df

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Küçük harfe çevir
    text = text.lower()
    # Parantez içini temizle (örneğin "(C17)" gibi)
    text = re.sub(r"\([^\)]*\)", "", text)
    # Noktalama ve sayı temizleme
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize ve lemmatize
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def frequency_analysis(df):

    text_columns = [
        "diseaseName",
        "diseaseClasses_DO",
        "diseaseClasses_MSH",
        "diseaseClasses_UMLS_ST",
        "diseaseClasses_HPO"
    ]

    for col in text_columns:
        print(f"\n--- {col.upper()} sütunu kelime frekansı ---")

        all_text = " ".join(df[col].dropna())
        words = all_text.split()

        word_freq = Counter(words)
        most_common = word_freq.most_common(20)
        print(most_common)

        words, counts = zip(*most_common)
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.title(f"{col} Sütununda En Sık Geçen 20 Kelime")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{col} Sütunu - WordCloud")
        plt.show()

def tf_idf(df):
    
    # TF-IDF vektörleştirici oluştur
    vectorizer = TfidfVectorizer(max_features=1000)  # En sık geçen 1000 terim
    tfidf_matrix = vectorizer.fit_transform(df["diseaseName"])

    feature_names = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_df.to_csv("filtered_tfidf_diseaseName.csv", index=False)

    mean_tfidf = tfidf_df.mean().sort_values(ascending=False)
    top_n = 20
    top_words = mean_tfidf[:top_n]

    plt.figure(figsize=(10, 5))
    top_words.plot(kind="bar", color="skyblue")
    plt.title("En Anlamlı 20 Terim (TF-IDF)")
    plt.ylabel("Ortalama TF-IDF Skoru")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tfidf_top20_terms.png")
    plt.show()
    
def gene_disease(df):
    """co-occurance"""
    # 1. En sık birlikte görülen gen–hastalık çiftleri
    pair_freq = df.groupby(["symbolOfGene", "diseaseName"]).size().reset_index(name="count")
    pair_freq_sorted = pair_freq.sort_values(by="count", ascending=False)
    print(pair_freq_sorted.head(10))

    # 2. Gen başına hastalık sayısı
    gene_disease_count = df.groupby("symbolOfGene")["diseaseName"].nunique().sort_values(ascending=False)
    print(gene_disease_count.head(10))

    # 3. Hastalık başına gen sayısı
    disease_gene_count = df.groupby("diseaseName")["symbolOfGene"].nunique().sort_values(ascending=False)
    print(disease_gene_count.head(10))
    
    top_pairs = pair_freq_sorted.head(20)

    plt.figure(figsize=(10,6))
    sns.barplot(data=top_pairs, x="count", y=top_pairs["symbolOfGene"] + " ↔ " + top_pairs["diseaseName"])
    plt.title("En Sık Görülen Gen-Hastalık Eşleşmeleri")
    plt.xlabel("Frekans")
    plt.ylabel("Gen ↔ Hastalık")
    plt.tight_layout()
    plt.show()

def network_analysis(df):
    edges = df[["symbolOfGene", "diseaseName"]].dropna()
    edges = edges.drop_duplicates()

    # Bipartite graph 
    B = nx.Graph()
    B.add_nodes_from(edges["symbolOfGene"], bipartite="genes")
    B.add_nodes_from(edges["diseaseName"], bipartite="diseases")
    B.add_edges_from(list(edges.itertuples(index=False, name=None)))

    print(f"Toplam düğüm sayısı: {B.number_of_nodes()}")
    print(f"Toplam bağlantı sayısı: {B.number_of_edges()}") 
    
    gene_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == "genes"}
    disease_nodes = set(B) - gene_nodes

    # genler arası bağlantılar
    G_genes = bipartite.projected_graph(B, gene_nodes)
    print(f"Gen projeksiyonundaki düğüm sayısı: {G_genes.number_of_nodes()}")
    print(f"Gen projeksiyonundaki bağlantı sayısı: {G_genes.number_of_edges()}") 
    """
    # --- 1. Etiketsiz tüm ağ yapısı (karmaşık olsa da genel görünüm)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(B, k=0.1, iterations=10)
    nx.draw(B, pos, node_size=5, with_labels=False, edge_color="gray", alpha=0.4)
    plt.title("Tüm Gen-Hastalık Ağı (Etiketsiz Yapı)")
    plt.show()

    # --- 2. En çok geçen 50 genin oluşturduğu alt ağı çiz
    top_genes = df["symbolOfGene"].value_counts().head(50).index.tolist()
    sub_edges = [(g, d) for g, d in B.edges() if g in top_genes]
    subgraph = nx.Graph()
    subgraph.add_edges_from(sub_edges)
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_color="skyblue", node_size=80, with_labels=False, font_size=7)
    plt.title("En Sık Geçen 50 Genin Hastalıklarla Alt Ağı")
    plt.show()
    """
    # Sadece ilk 100 kenarla 
    sub_nodes = list(edges.head(100)["symbolOfGene"]) + list(edges.head(100)["diseaseName"])
    subgraph = B.subgraph(sub_nodes)

    plt.figure(figsize=(12, 8))
    nx.draw(subgraph, with_labels=True, node_color="skyblue", node_size=300, font_size=8)
    plt.title("Gen-Hastalık Bipartite Network (İlk 100 Kenar)")
    plt.show()
