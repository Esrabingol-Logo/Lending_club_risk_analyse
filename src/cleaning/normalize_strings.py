"""
normalize_strings.py
────────────────────
Görev : String kolonlarda trim, lowercase yapar.
        NaN değerleri 'nan' string'ine dönüşmez.
"""

import pandas as pd


def normalize_strings(df: pd.DataFrame,
                      kolonlar: list = None,
                      lower: bool = True,
                      strip: bool = True) -> pd.DataFrame:
   
    if kolonlar is None:
        kolonlar = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

    sayac = 0
    for kolon in kolonlar:
        if kolon not in df.columns:
            continue
        if df[kolon].dtype not in ['object', 'category']:
            continue

        seri = df[kolon].copy()
        mask = seri.notna()          
        if strip:
            seri[mask] = seri[mask].astype(str).str.strip()
        if lower:
            seri[mask] = seri[mask].str.lower()

        df[kolon] = seri
        sayac += 1

    print(f"✅ String normalize: {sayac} kolon işlendi")
    return df