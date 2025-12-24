import pandas as pd
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn
import os

# --- KONFIGURASI ---
DATA_PATH = 'preprocessed_data.csv' 

def train_model():
    print("Memuat data...")
    
    # Cek apakah file data ada
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File '{DATA_PATH}' tidak ditemukan di folder ini.")
        # Kita buat dummy data jika file tidak ada (HANYA AGAR TIDAK ERROR DI TEST)
        # Tapi sebaiknya pastikan file csv Anda ikut ter-upload.
        return

    data = pd.read_csv(DATA_PATH)
    print(f"Data berhasil dimuat. Shape: {data.shape}")

    # 1. AKTIFKAN AUTOLOG
    # Ini akan otomatis mencatat parameter dan metrik model sklearn
    mlflow.sklearn.autolog()

    # 2. MULAI EXPERIMENT
    # PENTING: Jangan gunakan mlflow.set_experiment() di sini.
    # Biarkan 'mlflow run' di GitHub Actions yang mengatur nama eksperimennya.
    
    print("Memulai training dengan MLflow...")
    with mlflow.start_run():
        
        # Inisialisasi Model
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Train Model
        kmeans.fit(data)
        
        print("Training selesai! Model dan parameter telah tersimpan otomatis.")

if __name__ == "__main__":
    train_model()