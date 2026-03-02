"""
outlier_rules.py
────────────────
Görev : Sayısal kolonlardaki uç değerleri sınırlandırır.
        Percentile değerleri sadece train setinden hesaplanır.
        Global leakage önlenir.
"""

import pandas as pd
import numpy as np


# LC'ye özel sentinel/anomali kuralları
OZEL_KURALLAR = {
    'annual_inc' : {'min': 0,   'max': None},
    'dti'        : {'min': 0,   'max': 100},
    'revol_util' : {'min': 0,   'max': 100},
    'loan_amnt'  : {'min': 500, 'max': None},
}


def compute_clip_bounds(df_train: pd.DataFrame,
                        kolonlar: list,
                        alt_percentile: float = 0.01,
                        ust_percentile: float = 0.99) -> dict:
  
    sinirlar = {}
    for kolon in kolonlar:
        if kolon not in df_train.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_train[kolon]):
            continue

        alt = df_train[kolon].quantile(alt_percentile)
        ust = df_train[kolon].quantile(ust_percentile)

        # Özel kurallari uygula
        if kolon in OZEL_KURALLAR:
            kural = OZEL_KURALLAR[kolon]
            if kural.get('min') is not None:
                alt = max(alt, kural['min'])
            if kural.get('max') is not None:
                ust = min(ust, kural['max'])

        sinirlar[kolon] = {'alt': alt, 'ust': ust}

    return sinirlar


def apply_outlier_rules(df: pd.DataFrame,
                        kolonlar: list = None,
                        clip_bounds: dict = None,
                        alt_percentile: float = 0.01,
                        ust_percentile: float = 0.99) -> pd.DataFrame:
    
    if kolonlar is None:
        print("⚠️  Kolon listesi verilmedi")
        return df

    # clip_bounds yoksa df'ten hesapla (sadece EDA için)
    if clip_bounds is None:
        print("   ⚠️  clip_bounds verilmedi — df üzerinden hesaplanıyor")
        print("   ⚠️  Production'da compute_clip_bounds(train) kullanın!")
        clip_bounds = compute_clip_bounds(
            df, kolonlar, alt_percentile, ust_percentile)

    mevcut = [c for c in kolonlar
              if c in df.columns and c in clip_bounds]
    sayac  = 0

    print(f"── Uç Değer Düzeltme (clip) ─────────────────")

    for kolon in mevcut:
        if not pd.api.types.is_numeric_dtype(df[kolon]):
            continue
        alt = clip_bounds[kolon]['alt']
        ust = clip_bounds[kolon]['ust']
        df[kolon] = df[kolon].clip(lower=alt, upper=ust)\
                              .astype('float32')
        sayac += 1

    print(f"   ✅ {sayac} kolona uygulandı")
    print(f"   Alt: %{alt_percentile*100:.0f}  "
          f"Üst: %{ust_percentile*100:.0f}")

    return df