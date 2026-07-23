import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import json
import os

def clean_and_prepare(df):
    # 1. Hapus duplikat murni
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Buang Test Drive Car (anomali bisnis yang sudah kita bahas)
    df = df[df['owner'] != 'Test Drive Car'].reset_index(drop=True)
    
    # 3. Ekstrak brand untuk mencegah model sekadar 'menghafal' nama mobil spesifik
    df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
    df = df.drop('name', axis=1) # Buang nama aslinya
    
    # 4. Ubah year menjadi umur (mempermudah interpretasi regresi)
    df['umur'] = 2020 - df['year']
    df = df.drop('year', axis=1)
    
    return df

def run_training():
    print("--- Memulai Tahap Training (Cross-Validation 5-Fold) ---")
    df = pd.read_csv('data/CAR DETAILS FROM CAR DEKHO.csv')
    df = clean_and_prepare(df)
    
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    
    # ATURAN EMAS: Split SEBELUM preprocessing pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Kunci Test Set! Simpan dan JANGAN disentuh sampai skrip evaluate.py
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print("Data test berhasil diisolasi ke data/X_test.csv dan y_test.csv")
    
    # --- Setup Preprocessing di dalam ColumnTransformer ---
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    num_cols = ['km_driven', 'umur']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
    
    # --- Komparasi 3 Model ---
    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    best_score = -np.inf
    best_model_name = ""
    best_pipeline = None
    
    print("\nHasil 5-Fold Cross Validation (R2 Score):")
    for name, model in models.items():
        # Bungkus preprocessing dan algoritma ke dalam satu PIPELINE UTUH
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Evaluasi menggunakan R-squared
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        mean_score = scores.mean()
        print(f"- {name}: {mean_score:.4f} (std: {scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nModel terbaik terpilih: {best_model_name} dengan R2 = {best_score:.4f}")
    
    # Latih model terbaik secara utuh menggunakan seluruh data training
    print(f"Melatih {best_model_name} final...")
    best_pipeline.fit(X_train, y_train)
    
    # Simpan Artefak Pipeline (bukan model telanjang)
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, 'models/model.joblib')
    
    # Simpan Metadata (Syarat rubrik)
    metadata = {
        "model_terbaik": best_model_name,
        "r2_cv_score": round(best_score, 4),
        "metrik_utama": "MAE dan R2",
        "justifikasi_bisnis": "MAE dipilih karena lebih mudah dipahami oleh penjual mobil bekas sebagai selisih harga absolut dalam satuan mata uang asli. R2 untuk melihat seberapa baik fitur menjelaskan variansi harga."
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Pipeline utuh dan metadata berhasil disimpan ke models/!")

if __name__ == "__main__":
    run_training()