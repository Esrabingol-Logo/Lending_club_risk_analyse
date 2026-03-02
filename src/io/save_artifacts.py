"""
save_artifacts.py
─────────────────
Görev : İşlenmiş DataFrame ve model artifact'larını kaydeder.
        - Parquet (hızlı okuma)
        - CSV (gerekirse)
        - Klasör oluşturma

Kullanım:
    from src.io.save_artifacts import save_parquet, save_csv, create_dirs
"""

import pandas as pd
import os
import time


# 1. KLASÖR OLUŞTUR 

def create_dirs(*paths: str) -> None:
    """
    Verilen klasör yollarını oluşturur.
    Zaten varsa hata vermez.

    Args:
        *paths : Oluşturulacak klasör yolları
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"📁 Klasör hazır: {path}")


# 2. PARQUET KAYDET 

def save_parquet(df: pd.DataFrame,
                 filepath: str,
                 index: bool = False) -> None:
    """
    DataFrame'i parquet formatında kaydeder.
    Parquet → CSV'den 5-10x hızlı okuma, %70 daha az yer.

    Args:
        df       : Kaydedilecek DataFrame
        filepath : Kayıt yolu (.parquet uzantılı)
        index    : Index'i kaydet (genelde False)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    print(f"💾 Parquet kaydediliyor: {filepath}")
    baslangic = time.time()

    df.to_parquet(filepath, index=index)

    boyut_mb = os.path.getsize(filepath) / 1024 ** 2
    sure = time.time() - baslangic

    print(f"✅ Kaydedildi!")
    print(f"   Satır  : {df.shape[0]:,}")
    print(f"   Sütun  : {df.shape[1]:,}")
    print(f"   Boyut  : {boyut_mb:.1f} MB")
    print(f"   Süre   : {sure:.1f} saniye")


# 3. CSV KAYDET 

def save_csv(df: pd.DataFrame,
             filepath: str,
             index: bool = False) -> None:
    """
    DataFrame'i CSV formatında kaydeder.

    Args:
        df       : Kaydedilecek DataFrame
        filepath : Kayıt yolu (.csv uzantılı)
        index    : Index'i kaydet
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    print(f"💾 CSV kaydediliyor: {filepath}")
    baslangic = time.time()

    df.to_csv(filepath, index=index)

    boyut_mb = os.path.getsize(filepath) / 1024 ** 2
    sure = time.time() - baslangic

    print(f"✅ Kaydedildi!")
    print(f"   Boyut : {boyut_mb:.1f} MB")
    print(f"   Süre  : {sure:.1f} saniye")


# 4. PARQUET OKU 

def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Parquet dosyasını okur.

    Args:
        filepath : Parquet dosya yolu

    Returns:
        DataFrame
    """
    if not os.path.exists(filepath):
        print(f"❌ Dosya bulunamadı: {filepath}")
        return pd.DataFrame()

    print(f"📂 Parquet okunuyor: {filepath}")
    baslangic = time.time()

    df = pd.read_parquet(filepath)

    sure = time.time() - baslangic
    print(f"✅ Okundu!")
    print(f"   Satır : {df.shape[0]:,}")
    print(f"   Sütun : {df.shape[1]:,}")
    print(f"   Süre  : {sure:.1f} saniye")

    return df