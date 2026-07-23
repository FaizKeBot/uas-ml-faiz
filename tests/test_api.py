import pytest
from fastapi.testclient import TestClient
from app.main import app

# 1. KUNCI PERBAIKAN: Gunakan fixture dengan context manager ('with')
# Ini memaksa FastAPI untuk menjalankan 'lifespan' (memuat model) sebelum test dimulai.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ==========================================
# 1. MEKANIKAL TESTS (Minimal 4 Sesuai Rubrik)
# ==========================================

# Tambahkan 'client' sebagai argumen di setiap fungsi test
def test_health_check_mengembalikan_200(client):
    """Test mekanis 1: Endpoint /health harus merespons 200 OK dan status sehat."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediksi_input_valid_mengembalikan_200(client):
    """Test mekanis 2: Input yang valid harus berhasil diprediksi."""
    payload = {
        "name": "Hyundai Verna 1.6 SX",
        "year": 2015,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200
    assert "harga_estimasi" in response.json()

def test_prediksi_field_hilang_mengembalikan_422(client):
    """Test mekanis 3: Menghilangkan field wajib (year) harus ditolak."""
    payload = {
        "name": "Hyundai Verna",
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_prediksi_enum_tidak_dikenal_mengembalikan_422(client):
    """Test mekanis 4: Tipe bahan bakar yang tidak ada di Enum harus ditolak."""
    payload = {
        "name": "Hyundai Verna",
        "year": 2015,
        "km_driven": 45000,
        "fuel": "Bahan Bakar Nuklir", # Enum ngawur
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422


# ==========================================
# 2. BEHAVIORAL TESTS (Minimal 2 Sesuai Rubrik)
# ==========================================

def test_behavioral_mobil_tua_lebih_murah(client):
    """
    Behavioral Test 1: Mobil yang lebih tua dengan spesifikasi identik 
    harus diprediksi dengan harga yang lebih murah.
    """
    mobil_baru = {
        "name": "Maruti Swift",
        "year": 2018,
        "km_driven": 50000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
    
    mobil_tua = mobil_baru.copy()
    mobil_tua["year"] = 2010 
    
    resp_baru = client.post("/predict-harga", json=mobil_baru).json()
    resp_tua = client.post("/predict-harga", json=mobil_tua).json()
    
    assert resp_tua["harga_estimasi"] < resp_baru["harga_estimasi"], "Mobil tua seharunya lebih murah!"

def test_behavioral_kilometer_tinggi_lebih_murah(client):
    """
    Behavioral Test 2: Mobil yang sudah menempuh kilometer sangat jauh
    harus lebih murah dibandingkan mobil simpanan (low KM).
    """
    mobil_simpanan = {
        "name": "Honda City",
        "year": 2016,
        "km_driven": 20000, 
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
    
    mobil_capek = mobil_simpanan.copy()
    mobil_capek["km_driven"] = 200000 
    
    resp_simpanan = client.post("/predict-harga", json=mobil_simpanan).json()
    resp_capek = client.post("/predict-harga", json=mobil_capek).json()
    
    assert resp_capek["harga_estimasi"] < resp_simpanan["harga_estimasi"], "Mobil kilometer tinggi seharusnya lebih murah!"