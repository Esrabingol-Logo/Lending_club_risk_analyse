"""
parse_dates.py
──────────────
Görev : Tarih kolonlarını parse eder ve
        sayısal özelliklere dönüştürür.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.cleaning.parse_dates import parse_dates
"""

import pandas as pd
import numpy as np


def parse_dates(df: pd.DataFrame,
                kolonlar: list,
                referans_tarih: str = None,
                format: str = None) -> pd.DataFrame:
   
    print(" Tarih Dönüşümü ")

    ref = pd.to_datetime(referans_tarih) \
          if referans_tarih else None

    for kolon in kolonlar:
        if kolon not in df.columns:
            print(f"   ⚠️  {kolon} bulunamadı, atlandı")
            continue

        try:
            if format:
                tarih = pd.to_datetime(df[kolon],
                                       format=format,
                                       errors='coerce')
            else:
                tarih = pd.to_datetime(df[kolon],
                                       errors='coerce')

            # Yıl ve ay
            df[f"{kolon}_YIL"] = tarih.dt.year\
                                       .astype('float32')
            df[f"{kolon}_AY"]  = tarih.dt.month\
                                       .astype('float32')

            # Referans ile ay farkı
            if ref is not None:
                df[f"{kolon}_AY_FARK"] = (
                    (ref - tarih).dt.days / 30
                ).astype('float32')

            # Orijinal kolonu düşür
            df = df.drop(columns=[kolon])

            print(f"   ✅ {kolon} → "
                  f"{kolon}_YIL, {kolon}_AY"
                  + (f", {kolon}_AY_FARK"
                     if ref else ""))

        except Exception as e:
            print(f"   {kolon} dönüştürülemedi: {e}")

    return df