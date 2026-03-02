"""
handle_missing.py
─────────────────
Görev : Eksik değerleri missing_strategies.json'a göre doldurur.
        Median/mode değerleri sadece train setinden hesaplanır.
        Global leakage önlenir.
"""

import pandas as pd
import numpy as np
import json


def compute_fill_values(df_train: pd.DataFrame,
                        kolonlar: list) -> dict:
   
    fill_values = {}

    for kolon in kolonlar:
        if kolon not in df_train.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_train[kolon]):
            fill_values[kolon] = df_train[kolon].median()
        else:
            mode_seri = df_train[kolon].mode()
            if len(mode_seri) > 0:
                fill_values[kolon] = mode_seri[0]
            else:
                fill_values[kolon] = None

    return fill_values


def handle_missing(df: pd.DataFrame,
                   missing_json: str,
                   fill_values: dict = None) -> pd.DataFrame:
    
    with open(missing_json, encoding='utf-8') as f:
        stratejiler = json.load(f)

    sil_listesi    = stratejiler.get('sil', [])
    flag_listesi   = stratejiler.get('flag_ve_median', [])
    doldur_listesi = stratejiler.get('median_doldur', [])

    # Tüm doldurulacak kolonlar
    tum_doldurulacak = flag_listesi + doldur_listesi

    # fill_values yoksa df'ten hesapla (sadece EDA için)
    if fill_values is None:
        print("   ⚠️  fill_values verilmedi — df üzerinden hesaplanıyor")
        print("   ⚠️  Production'da compute_fill_values(train) kullanın!")
        fill_values = compute_fill_values(df, tum_doldurulacak)

    # ── 1. SİL 
    mevcut_sil = [c for c in sil_listesi if c in df.columns]
    if mevcut_sil:
        df = df.drop(columns=mevcut_sil)
        print(f"   ❌ Silindi          : {len(mevcut_sil)} kolon")

    # ── 2. FLAG + MEDIAN 
    flag_sayac = 0
    for kolon in flag_listesi:
        if kolon not in df.columns:
            continue

        # Flag ekle
        df[f"{kolon}_MISSING"] = df[kolon].isnull().astype('int8')

        # fill_values'dan doldur
        deger = fill_values.get(kolon)
        if deger is not None:
            df[kolon] = df[kolon].fillna(deger)

        flag_sayac += 1

    if flag_sayac:
        print(f"   🚩 Flag + Median   : {flag_sayac} kolon")

    # ── 3. MEDIAN / MODE 
    doldur_sayac = 0
    for kolon in doldur_listesi:
        if kolon not in df.columns:
            continue

        deger = fill_values.get(kolon)
        if deger is not None:
            df[kolon] = df[kolon].fillna(deger)
            doldur_sayac += 1

    if doldur_sayac:
        print(f"   🔧 Median/Mode     : {doldur_sayac} kolon")

    kalan = df.isnull().sum().sum()
    print(f"   Kalan eksik değer  : {kalan:,}")

    return df