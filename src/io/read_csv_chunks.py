"""
read_csv_chunks.py
──────────────────
Görev : Büyük CSV / CSV.GZ dosyalarını chunk bazlı okur.
        - Toplam satır sayısını hesaplar
        - İstenen boyutta random sample alır
        - Herhangi bir veri setiyle çalışır (config bağımsız)

Kullanım:
    from src.io.read_csv_chunks import count_rows, read_sample, read_chunks

Başka veri setinde kullanım:
    count_rows("data/raw/home_credit.csv")
    read_sample("data/raw/home_credit.csv", sample_size=50000)
"""

import pandas as pd
import numpy as np
import os
import time


# ── 1. DOSYA VARLIĞI KONTROLÜ ───────────────────────────────────────────────

def check_file(filepath: str) -> bool:
    """
    Dosyanın var olup olmadığını kontrol eder.

    Args:
        filepath : Dosya yolu

    Returns:
        True → dosya var
        False → dosya yok
    """
    if not os.path.exists(filepath):
        print(f"❌ Dosya bulunamadı: {filepath}")
        return False

    boyut_mb = os.path.getsize(filepath) / 1024 ** 2
    print(f"✅ Dosya bulundu: {filepath}")
    print(f"   Boyut: {boyut_mb:.1f} MB")
    return True


# ── 2. TOPLAM SATIR SAYISI ───────────────────────────────────────────────────

def count_rows(filepath: str, chunk_size: int = 100000) -> int:
    """
    Dosyayı chunk chunk okuyarak toplam satır sayısını hesaplar.
    Tüm dosyayı RAM'e yüklemez.

    Args:
        filepath   : CSV / CSV.GZ dosya yolu
        chunk_size : Her chunk'ta okunacak satır sayısı

    Returns:
        Toplam satır sayısı (header hariç)
    """
    if not check_file(filepath):
        return 0

    print(f"\n📊 Satır sayılıyor... (chunk_size={chunk_size:,})")
    baslangic = time.time()

    toplam = 0
    chunk_no = 0

    for chunk in pd.read_csv(filepath, chunksize=chunk_size,
                              low_memory=False, on_bad_lines='skip'):
        toplam += len(chunk)
        chunk_no += 1
        print(f"   Chunk {chunk_no:>3} okundu → toplam: {toplam:>12,} satır", end="\r")

    sure = time.time() - baslangic
    print(f"\n✅ Tamamlandı!")
    print(f"   Toplam satır : {toplam:,}")
    print(f"   Chunk sayısı : {chunk_no}")
    print(f"   Süre         : {sure:.1f} saniye")

    return toplam


# ── 3. RANDOM SAMPLE ALMA ────────────────────────────────────────────────────

def read_sample(filepath: str,
                sample_size: int = 50000,
                random_state: int = 42) -> pd.DataFrame:
    """
    Dosyadan rastgele sample alır.
    Tüm dosyayı okumadan önce satır sayısını tahmin eder,
    sonra random index'lere göre okur.

    Args:
        filepath     : CSV / CSV.GZ dosya yolu
        sample_size  : Alınacak sample satır sayısı
        random_state : Tekrar üretilebilirlik için seed

    Returns:
        Sample DataFrame
    """
    if not check_file(filepath):
        return pd.DataFrame()

    print(f"\n🎲 Sample alınıyor... (sample_size={sample_size:,})")
    baslangic = time.time()

    # Önce header'ı oku — kolon adlarını al
    header = pd.read_csv(filepath, nrows=0)
    kolonlar = header.columns.tolist()

    # Tüm satır index'lerini chunk'larla tara,
    # random N tanesini seç
    np.random.seed(random_state)

    # Satır sayısını hızlıca tahmin et (ilk chunk'tan)
    ilk_chunk = pd.read_csv(filepath, nrows=sample_size * 3,
                             low_memory=False, on_bad_lines='skip')

    if len(ilk_chunk) <= sample_size:
        print(f"   ℹ️  Dosya sample_size'dan küçük, tümü okunuyor")
        sure = time.time() - baslangic
        print(f"✅ Sample alındı: {len(ilk_chunk):,} satır | {sure:.1f} sn")
        return ilk_chunk

    # Random sample al
    sample_df = ilk_chunk.sample(n=sample_size, random_state=random_state)
    sample_df = sample_df.reset_index(drop=True)

    sure = time.time() - baslangic
    print(f"✅ Sample alındı!")
    print(f"   Satır  : {len(sample_df):,}")
    print(f"   Sütun  : {len(sample_df.columns)}")
    print(f"   Süre   : {sure:.1f} saniye")

    return sample_df


# ── 4. CHUNK BAZLI OKUMA (ITERATOR) ─────────────────────────────────────────

def read_chunks(filepath: str,
                chunk_size: int = 100000,
                usecols: list = None,
                dtype: dict = None):
    """
    Dosyayı chunk chunk okuyan generator.
    Her chunk'ta işlem yapmak için kullanılır.

    Args:
        filepath   : CSV / CSV.GZ dosya yolu
        chunk_size : Chunk boyutu
        usecols    : Sadece bu kolonları oku (None = hepsi)
        dtype      : Kolon tiplerini zorla

    Yields:
        Her iterasyonda bir chunk DataFrame

    Kullanım:
        for chunk in read_chunks("data.csv"):
            # chunk üzerinde işlem yap
            print(chunk.shape)
    """
    if not check_file(filepath):
        return

    print(f"\n📦 Chunk okuma başlıyor (chunk_size={chunk_size:,})")

    okuma_kwargs = {
        "chunksize"    : chunk_size,
        "low_memory"   : False,
        "on_bad_lines" : "skip",
    }
    if usecols is not None:
        okuma_kwargs["usecols"] = usecols
    if dtype is not None:
        okuma_kwargs["dtype"] = dtype

    chunk_no = 0
    for chunk in pd.read_csv(filepath, **okuma_kwargs):
        chunk_no += 1
        yield chunk

    print(f"✅ Toplam {chunk_no} chunk okundu")


# ── 5. TAM DOSYAYI OKU (KÜÇÜK DOSYALAR İÇİN) ────────────────────────────────

def read_full(filepath: str,
              usecols: list = None,
              dtype: dict = None,
              nrows: int = None) -> pd.DataFrame:
    """
    Dosyanın tamamını tek seferde okur.
    Küçük/orta boyutlu dosyalar için (< 500MB).

    Args:
        filepath : Dosya yolu
        usecols  : Okunacak kolonlar (None = hepsi)
        dtype    : Kolon tipleri
        nrows    : Okunacak max satır (None = hepsi)

    Returns:
        DataFrame
    """
    if not check_file(filepath):
        return pd.DataFrame()

    print(f"\n📂 Dosya okunuyor...")
    baslangic = time.time()

    okuma_kwargs = {
        "low_memory"   : False,
        "on_bad_lines" : "skip",
    }
    if usecols is not None:
        okuma_kwargs["usecols"] = usecols
    if dtype is not None:
        okuma_kwargs["dtype"] = dtype
    if nrows is not None:
        okuma_kwargs["nrows"] = nrows

    df = pd.read_csv(filepath, **okuma_kwargs)

    sure = time.time() - baslangic
    bellek = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"✅ Okundu!")
    print(f"   Satır  : {df.shape[0]:,}")
    print(f"   Sütun  : {df.shape[1]:,}")
    print(f"   Bellek : {bellek:.1f} MB")
    print(f"   Süre   : {sure:.1f} saniye")

    return df