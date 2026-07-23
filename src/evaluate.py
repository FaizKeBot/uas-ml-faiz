import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def run_evaluation():
    print("--- Memulai Evaluasi pada Test Set (Satu Kali Eksekusi) ---")
    
    # 1. Muat Test Set dan Model Pipeline Utuh
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze() # Ubah ke bentuk Series
    pipeline = joblib.load('models/model.joblib')
    
    # 2. Lakukan Prediksi
    y_pred = pipeline.predict(X_test)
    
    # 3. Kalkulasi Metrik Sesuai Justifikasi Bisnis
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print("\nMetrik Evaluasi Akhir (Data Tak Terlihat):")
    print(f"MAE  : {mae:,.2f} INR")
    print(f"RMSE : {rmse:,.2f} INR")
    print(f"R2   : {r2:.4f}")
    
    # 4. Artefak Grafik Sesuai Rubrik
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='blue')
    # Garis ideal merah
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Harga Aktual vs Prediksi (Test Set)')
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    
    # Simpan plot ke laporan
    plot_path = 'reports/5_evaluasi_aktual_vs_prediksi.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafik evaluasi berhasil disimpan ke {plot_path}")
    
    # 5. Eksekusi Syarat Wajib: Analisis 5 Kesalahan Terburuk
    print("\n--- Analisis 5 Kesalahan Terburuk (Wajib Rubrik) ---")
    errors = abs(y_test - y_pred)
    
    # Gabungkan kembali data untuk dianalisis
    error_analysis_df = X_test.copy()
    error_analysis_df['Harga_Aktual'] = y_test
    error_analysis_df['Prediksi_Model'] = y_pred
    error_analysis_df['Selisih_Error'] = errors
    
    # Urutkan berdasarkan error terbesar
    worst_5 = error_analysis_df.sort_values(by='Selisih_Error', ascending=False).head(5)
    
    # Tampilkan kolom kunci agar mudah dianalisis di laporan
    print(worst_5[['brand', 'km_driven', 'umur', 'Harga_Aktual', 'Prediksi_Model', 'Selisih_Error']])
    
    # Rekomendasi untuk laporan
    print("\n[TIPS LAPORAN]: Gunakan tabel di atas di laporan PDF Anda. Biasanya error ekstrem terjadi pada mobil mewah atau mobil sangat tua yang tidak memiliki cukup data serupa saat training.")

if __name__ == "__main__":
    run_evaluation()