import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath):
    print(f"--- Memulai Proses EDA pada {filepath} ---")
    df = pd.read_csv(filepath)
    os.makedirs('reports', exist_ok=True)
    
    # 1. Sebaran Target (Histogram Harga)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['selling_price'], bins=50, kde=True)
    plt.title('Distribusi Harga Kendaraan (Target)')
    plt.savefig('reports/1_distribusi_target.png')
    plt.close()
    print("Grafik 1 (Distribusi Target) berhasil disimpan.")

    # 2. Visualisasi Data Bolong (Tetap dibuat untuk memenuhi syarat rubrik)
    plt.figure(figsize=(8, 5))
    df.isna().sum().plot(kind='bar')
    plt.title('Jumlah Nilai Hilang per Kolom')
    plt.savefig('reports/2_missing_values.png')
    plt.close()
    print("Grafik 2 (Missing Values) berhasil disimpan.")

    # 3. Fitur Paling Berhubungan (Korelasi Numerik)
    plt.figure(figsize=(8, 6))
    num_cols = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap Korelasi Fitur Numerik')
    plt.savefig('reports/3_heatmap_korelasi.png')
    plt.close()
    print("Grafik 3 (Heatmap Korelasi) berhasil disimpan.")

    # 4. Hubungan Non-linear Umur Kendaraan dan Harga (Wajib Kasus B)
    plt.figure(figsize=(10, 6))
    # Rekayasa fitur: Umur kendaraan (Asumsi tahun scraping data = 2020)
    df['umur_kendaraan'] = 2020 - df['year'] 
    sns.scatterplot(x='umur_kendaraan', y='selling_price', data=df, alpha=0.5)
    plt.title('Hubungan Umur Kendaraan vs Harga')
    plt.savefig('reports/4_scatter_umur_harga.png')
    plt.close()
    print("Grafik 4 (Scatter Umur vs Harga) berhasil disimpan.")
    
    print("\nEDA selesai! Semua grafik aman di dalam folder reports/")

if __name__ == "__main__":
    FILE_PATH = os.path.join("data", "CAR DETAILS FROM CAR DEKHO.csv")
    run_eda(FILE_PATH)