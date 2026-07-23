import pandas as pd
import os

def load_and_inspect_data(filepath):
    print(f"--- Memuat data dari {filepath} ---")
    df = pd.read_csv(filepath)
    
    # 1. Jumlah baris dan kolom
    print(f"\nJumlah Baris: {df.shape[0]}")
    print(f"Jumlah Kolom: {df.shape[1]}")
    
    # 2. Tipe tiap kolom
    print("\nTipe Tiap Kolom:")
    print(df.dtypes)
    
    # 3. Jumlah nilai hilang per kolom
    print("\nJumlah Nilai Hilang Per Kolom:")
    print(df.isna().sum())
    
    return df

if __name__ == "__main__":
    # Mengamankan path menggunakan os.path.join agar tahan banting di OS manapun
    FILE_PATH = os.path.join("data", "CAR DETAILS FROM CAR DEKHO.csv")
    load_and_inspect_data(FILE_PATH)