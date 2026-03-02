"""
deduplicate.py
──────────────
Görev : Duplicate satırları tespit eder ve kaldırır.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.cleaning.deduplicate import deduplicate
"""

import pandas as pd


def deduplicate(df: pd.DataFrame,
                subset: list = None,
                keep: str = 'first') -> pd.DataFrame:
    onceki = len(df)

    dup_sayisi = df.duplicated(subset=subset).sum()

    if dup_sayisi == 0:
        print(f"✅ Duplicate yok — {onceki:,} satır temiz")
        return df

    df = df.drop_duplicates(subset=subset, keep=keep)
    df = df.reset_index(drop=True)

    print(f"✅ Duplicate temizlendi:")
    print(f"   Önceki : {onceki:,}")
    print(f"   Silinen: {dup_sayisi:,}")
    print(f"   Sonraki: {len(df):,}")

    return df