# 🚀 Automated Preprocessing Pipeline - Telco Customer Churn

Proyek ini merupakan **Submission Akhir MLOps - Dicoding (2026)** oleh **Aqil Afif**.

Repository ini berisi otomatisasi tahapan preprocessing untuk dataset Telco Customer Churn. Seluruh tahapan dibungkus ke dalam Class `DataPreprocessor` berbasis Python agar terstruktur dan dapat dieksekusi secara otomatis melalui CI/CD GitHub Actions untuk menghasilkan dataset siap latih.

---

## 📌 Alur Pipeline (Preprocessing Steps)

Script `automate_Aqil-Afif.py` menjalankan 7 tahapan utama secara berurutan:

1. **Memuat Dataset** : Mengambil raw dataset secara langsung dari repositori IBM.
2. **Validasi Data** : Melakukan pengecekan awal terhadap jumlah missing values, duplikat, dan melihat distribusi kelas target `Churn`.
3. **Pembersihan Data** :
   - Menghapus kolom ID yang tidak relevan yaitu `customerID`.
   - Memperbaiki tipe data pada kolom `TotalCharges` dari string menjadi float numerik.
   - Mengisi (imputasi) data yang kosong (NaN) dengan nilai median.
   - Menghapus baris yang terduplikasi.
4. **Feature Engineering & Encoding** : Menerapkan Label Encoding pada target (`Churn`: Yes→1, No→0) dan One-Hot Encoding (OHE) pada fitur kategorik.
5. **Normalisasi** : Menerapkan `StandardScaler` pada fitur numerik (`tenure`, `MonthlyCharges`, dan `TotalCharges`).
6. **Train-Test Split** : Membagi data secara berstrata (stratified) dengan rasio 80% data latih dan 20% data uji (`test_size=0.2`) dengan random state 42.
7. **Menyimpan Output** : Menyimpan hasil preprocessing (`train.csv` dan `test.csv`) ke direktori lokal.

---

## 🛠️ Persyaratan Library (Dependencies)

Library yang digunakan dalam environment ini terdaftar dalam `requirements.txt`, yang meliputi:

| Kategori | Library |
|---|---|
| **Data Science Core** | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` |
| **Machine Learning** | `xgboost` *(opsional untuk eksperimen)* |
| **Tracking & Logging** | `mlflow`, `dagshub` |
| **Serving & Monitoring** | `fastapi`, `uvicorn`, `prometheus-client`, `psutil` |

---

## 🚀 Cara Menjalankan Secara Lokal

**1. Install semua dependensi:**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**2. Jalankan script pipeline:**

```bash
cd preprocessing
python automate_Aqil-Afif.py
```

---

## ⚙️ CI/CD Otomatisasi (GitHub Actions)

Proyek ini telah dilengkapi dengan GitHub Actions (`.github/workflows/preprocessing.yml`) untuk menjalankan pipeline preprocessing secara otomatis di environment **Ubuntu** dengan **Python 3.10**.

- **Trigger** : Workflow ini akan berjalan secara otomatis saat terdapat:
  - Push ke branch `main`, `master`, dan `develop`
  - Pull Request ke branch `main` dan `master`

- **Verifikasi & Artifacts** : GitHub Actions akan secara otomatis mengunggah artefak (`train.csv`, `test.csv`, file mentah, dan file `preprocessing.log`) yang akan tersimpan sebagai Artifact selama **30 hari**, sedangkan file log tersendiri akan disimpan selama **14 hari**.

---

## 📁 Struktur Output Direktori

Apabila script dijalankan secara sukses, script akan menghasilkan folder dan file berikut:

```
📦 output/
├── 📂 telco_raw/
│   └── Telco-Customer-Churn.csv     # Raw data salinan dari data asli
├── 📂 logs/
│   └── preprocessing.log            # Rekaman log mendetail pipeline
└── 📂 telco_preprocessing/
    ├── train.csv                    # Dataset untuk melatih model
    └── test.csv                     # Dataset untuk pengujian model
```

| File / Folder | Deskripsi |
|---|---|
| `telco_raw/Telco-Customer-Churn.csv` | Berisi raw data salinan dari data asli |
| `logs/preprocessing.log` | Rekaman log mendetail terkait berjalannya pipeline |
| `telco_preprocessing/train.csv` | Dataset hasil preprocessing untuk melatih model |
| `telco_preprocessing/test.csv` | Dataset hasil preprocessing untuk pengujian model |
