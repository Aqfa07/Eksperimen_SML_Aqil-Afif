"""
automate_Aqil-Afif.py
=====================
Automated Preprocessing Pipeline - Telco Customer Churn Dataset
Author  : Aqil Afif
Project : Submission Akhir MLOps - Dicoding
Date    : 2026

Deskripsi:
    Script ini membungkus seluruh tahapan preprocessing dari notebook
    Eksperimen_Aqil-Afif.ipynb ke dalam sebuah class yang terstruktur,
    dapat dijalankan secara otomatis via GitHub Actions, dan menghasilkan
    dataset siap latih yang tersimpan di direktori data/processed/.
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# URL dataset publik
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

# Kolom numerik yang akan di-scale
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Kolom target
TARGET_COL = "Churn"

# Kolom ID yang tidak relevan untuk model
DROP_COLS = ["customerID"]

# Proporsi data uji
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Direktori output
OUTPUT_DIR = "telco_preprocessing"


# ─────────────────────────────────────────────────────────────────────────────
# SETUP LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    """Konfigurasi logger dengan format yang informatif."""
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("PreprocessingPipeline")
    logger.setLevel(logging.INFO)

    # Formatter
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler ke konsol
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    # Handler ke file
    fh = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"), encoding="utf-8")
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# CLASS UTAMA: DataPreprocessor
# ─────────────────────────────────────────────────────────────────────────────
class DataPreprocessor:
    """
    Kelas yang mengenkapsulasi seluruh pipeline preprocessing
    dataset Telco Customer Churn.

    Atribut:
        df_raw      : DataFrame mentah hasil pemuatan
        df_clean    : DataFrame setelah dibersihkan
        X_train     : Fitur data latih
        X_test      : Fitur data uji
        y_train     : Label data latih
        y_test      : Label data uji
        scaler      : StandardScaler yang sudah di-fit
        label_enc   : LabelEncoder untuk kolom target

    Contoh penggunaan:
        >>> pipeline = DataPreprocessor()
        >>> X_train, X_test, y_train, y_test = pipeline.run()
    """

    def __init__(
        self,
        dataset_url: str = DATASET_URL,
        numerical_cols: list = None,
        target_col: str = TARGET_COL,
        drop_cols: list = None,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        output_dir: str = OUTPUT_DIR,
    ):
        self.dataset_url   = dataset_url
        self.numerical_cols = numerical_cols or NUMERICAL_COLS
        self.target_col    = target_col
        self.drop_cols     = drop_cols or DROP_COLS
        self.test_size     = test_size
        self.random_state  = random_state
        self.output_dir    = output_dir

        self.df_raw    = None
        self.df_clean  = None
        self.X_train   = None
        self.X_test    = None
        self.y_train   = None
        self.y_test    = None
        self.scaler    = StandardScaler()
        self.label_enc = LabelEncoder()

        self.logger = setup_logger()

    # ── STEP 1: Memuat Dataset ────────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        """Memuat dataset dari URL dan menyimpan salinan mentahnya."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Memuat Dataset")
        self.logger.info("=" * 60)
        self.logger.info(f"Sumber  : {self.dataset_url}")

        self.df_raw = pd.read_csv(self.dataset_url)

        self.logger.info(
            f"Dataset berhasil dimuat: {self.df_raw.shape[0]} baris, "
            f"{self.df_raw.shape[1]} kolom."
        )

        # Simpan raw data
        raw_dir = "../telco_raw"
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, "Telco-Customer-Churn.csv")
        self.df_raw.to_csv(raw_path, index=False)
        self.logger.info(f"Raw data disimpan ke: {raw_path}")

        return self.df_raw

    # ── STEP 2: Validasi Data ─────────────────────────────────────────────────
    def validate_data(self) -> None:
        """Melakukan pengecekan awal kualitas data (missing, duplikat, tipe)."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Validasi Data")
        self.logger.info("=" * 60)

        df = self.df_raw.copy()

        # Cek missing values
        missing = df.isnull().sum()
        total_missing = missing.sum()
        self.logger.info(f"Total missing values (NaN): {total_missing}")
        if total_missing > 0:
            self.logger.info(f"Detail:\n{missing[missing > 0]}")

        # Cek duplikat
        n_dup = df.duplicated().sum()
        self.logger.info(f"Jumlah baris duplikat: {n_dup}")

        # Distribusi target
        churn_dist = df[self.target_col].value_counts()
        self.logger.info(f"Distribusi target '{self.target_col}':\n{churn_dist.to_string()}")

        # Rangkuman statistik fitur numerik
        self.logger.info(
            f"Statistik fitur numerik:\n"
            f"{df[self.numerical_cols].describe().to_string()}"
        )

    # ── STEP 3: Pembersihan Data ──────────────────────────────────────────────
    def clean_data(self) -> pd.DataFrame:
        """
        Membersihkan data:
        - Menghapus kolom ID yang tidak relevan
        - Memperbaiki tipe data TotalCharges (string → float)
        - Mengisi missing value dengan median
        - Menghapus duplikat
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Pembersihan Data")
        self.logger.info("=" * 60)

        df = self.df_raw.copy()

        # 3a. Hapus kolom tidak relevan
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df.drop(cols_to_drop, axis=1, inplace=True)
            self.logger.info(f"Kolom dihapus: {cols_to_drop}")

        # 3b. Konversi TotalCharges ke numerik (ada spasi kosong di dataset asli)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_coerced = df["TotalCharges"].isnull().sum()
        self.logger.info(
            f"TotalCharges: {n_coerced} nilai non-numerik dikonversi ke NaN."
        )

        # 3c. Imputasi missing value dengan median
        for col in self.numerical_cols:
            if col in df.columns:
                median_val = df[col].median()
                n_fill = df[col].isnull().sum()
                if n_fill > 0:
                    df[col].fillna(median_val, inplace=True)
                    self.logger.info(
                        f"Kolom '{col}': {n_fill} NaN diisi dengan median ({median_val:.4f})."
                    )

        # 3d. Hapus duplikat (jika ada)
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        if before != after:
            self.logger.info(f"{before - after} baris duplikat dihapus.")

        self.logger.info(
            f"Data setelah pembersihan: {df.shape[0]} baris, {df.shape[1]} kolom."
        )
        self.df_clean = df
        return self.df_clean

    # ── STEP 4: Feature Engineering & Encoding ────────────────────────────────
    def encode_features(self):
        """
        Encoding fitur:
        - Label Encoding pada target (Churn: Yes→1, No→0)
        - One-Hot Encoding (get_dummies) pada fitur kategorik
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: Feature Engineering & Encoding")
        self.logger.info("=" * 60)

        df = self.df_clean.copy()

        # Pisahkan fitur dan target
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        # Label Encoding pada target
        y_encoded = self.label_enc.fit_transform(y)
        classes = dict(zip(self.label_enc.classes_, self.label_enc.transform(self.label_enc.classes_)))
        self.logger.info(f"Label Encoding target '{self.target_col}': {classes}")

        # One-Hot Encoding pada fitur kategorik
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.logger.info(f"Kolom kategorik yang di-encode ({len(cat_cols)}): {cat_cols}")
        X_encoded = pd.get_dummies(X, drop_first=True)
        self.logger.info(
            f"Jumlah fitur setelah One-Hot Encoding: {X_encoded.shape[1]}"
        )

        return X_encoded, y_encoded

    # ── STEP 5: Normalisasi / Scaling ─────────────────────────────────────────
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Menerapkan StandardScaler pada kolom numerik."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: Normalisasi Fitur Numerik (StandardScaler)")
        self.logger.info("=" * 60)

        num_cols_present = [c for c in self.numerical_cols if c in X.columns]
        self.logger.info(f"Kolom yang di-scale: {num_cols_present}")

        X_scaled = X.copy()
        X_scaled[num_cols_present] = self.scaler.fit_transform(X[num_cols_present])

        self.logger.info("Scaling selesai.")
        return X_scaled

    # ── STEP 6: Train-Test Split ──────────────────────────────────────────────
    def split_data(self, X: pd.DataFrame, y: np.ndarray):
        """Membagi data menjadi data latih dan data uji dengan stratifikasi."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: Train-Test Split")
        self.logger.info("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,          # ← Stratifikasi untuk menjaga proporsi kelas
        )

        self.logger.info(f"Test size   : {self.test_size * 100:.0f}%")
        self.logger.info(f"Random state: {self.random_state}")
        self.logger.info(f"Data Latih  : {X_train.shape[0]} baris, {X_train.shape[1]} fitur")
        self.logger.info(f"Data Uji    : {X_test.shape[0]} baris, {X_test.shape[1]} fitur")

        churn_train = y_train.sum() / len(y_train) * 100
        churn_test  = y_test.sum()  / len(y_test)  * 100
        self.logger.info(f"Churn rate train : {churn_train:.2f}%")
        self.logger.info(f"Churn rate test  : {churn_test:.2f}%")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        return X_train, X_test, y_train, y_test

    # ── STEP 7: Simpan Output ─────────────────────────────────────────────────
    def save_outputs(self) -> None:
        """Menyimpan dataset yang sudah diproses ke direktori output."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 7: Menyimpan Output")
        self.logger.info("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)

        # Gabungkan fitur + label untuk disimpan sebagai CSV
        train_df = self.X_train.copy()
        train_df[self.target_col] = self.y_train
        test_df  = self.X_test.copy()
        test_df[self.target_col]  = self.y_test

        train_path = os.path.join(self.output_dir, "train.csv")
        test_path  = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,   index=False)

        self.logger.info(f"Data latih disimpan ke : {train_path}")
        self.logger.info(f"Data uji disimpan ke   : {test_path}")

    # ── PIPELINE UTAMA ────────────────────────────────────────────────────────
    def run(self):
        """
        Menjalankan seluruh pipeline preprocessing secara berurutan.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
                   DataFrame/array siap digunakan untuk pelatihan model.
        """
        self.logger.info("╔══════════════════════════════════════════════════════╗")
        self.logger.info("║     PIPELINE PREPROCESSING - TELCO CUSTOMER CHURN   ║")
        self.logger.info("║               Aqil Afif - MLOps Dicoding             ║")
        self.logger.info("╚══════════════════════════════════════════════════════╝")

        self.load_data()
        self.validate_data()
        self.clean_data()
        X_encoded, y_encoded = self.encode_features()
        X_scaled = self.scale_features(X_encoded)
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y_encoded)
        self.save_outputs()

        self.logger.info("=" * 60)
        self.logger.info("✅ PIPELINE SELESAI! Dataset siap untuk pelatihan model.")
        self.logger.info("=" * 60)

        return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = DataPreprocessor()
    X_train, X_test, y_train, y_test = pipeline.run()

    print("\n" + "=" * 60)
    print("RINGKASAN HASIL PREPROCESSING")
    print("=" * 60)
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  y_test  shape : {y_test.shape}")
    print(f"  Fitur         : {list(X_train.columns[:5])} ...")
    print("=" * 60)
    print("Output tersimpan di -> telco_preprocessing/")
