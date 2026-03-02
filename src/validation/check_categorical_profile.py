"""
check_categorical_profile.py
─────────────────────────────
Görev : Kategorik kolonların profilini çıkarır.
        - Benzersiz değer sayısı
        - En sık görülen değerler
        - Default oranı (target ile ilişki)
        - Kardinalite seviyesi
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.validation.check_categorical_profile import check_categorical_profile
"""

import pandas as pd
import numpy as np


def check_categorical_profile(df: pd.DataFrame,
                               hedef_kolon: str = 'target',
                               max_unique: int = 50) -> pd.DataFrame:
    
    kategorik = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    kategorik = [c for c in kategorik if c != hedef_kolon]

    rapor_listesi = []

    for kolon in kategorik:
        seri        = df[kolon]
        n_unique    = seri.nunique()
        eksik_oran  = seri.isnull().mean() * 100
        en_sik      = seri.value_counts().index[0] \
                      if seri.notna().any() else None
        en_sik_oran = seri.value_counts(normalize=True).iloc[0] * 100 \
                      if seri.notna().any() else None

        # Kardinalite seviyesi
        if n_unique <= 5:
            kardinalite = 'düşük'
        elif n_unique <= max_unique:
            kardinalite = 'orta'
        else:
            kardinalite = 'yüksek'

        # Default oranı (target varsa)
        if hedef_kolon in df.columns:
            default_orani = df.groupby(kolon)[hedef_kolon]\
                             .mean().to_dict()
        else:
            default_orani = {}

        rapor_listesi.append({
            'kolon'          : kolon,
            'n_unique'       : n_unique,
            'kardinalite'    : kardinalite,
            'eksik_oran'     : round(eksik_oran, 2),
            'en_sik_deger'   : en_sik,
            'en_sik_oran'    : round(en_sik_oran, 1)
                               if en_sik_oran else None,
        })

    rapor = pd.DataFrame(rapor_listesi)\
              .sort_values('n_unique', ascending=False)\
              .reset_index(drop=True)

    print("── Kategorik Kolon Profili ───────────────────")
    print(f"   Toplam kategorik     : {len(rapor)}")
    print(f"   Düşük kardinalite   : "
          f"{(rapor['kardinalite']=='düşük').sum()}  (≤5 benzersiz)")
    print(f"   Orta kardinalite    : "
          f"{(rapor['kardinalite']=='orta').sum()}   (6-{max_unique})")
    print(f"   Yüksek kardinalite  : "
          f"{(rapor['kardinalite']=='yüksek').sum()}  (>{max_unique})")

    return rapor