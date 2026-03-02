"""
drop_leakage.py
───────────────
Görev : Data leakage kolonlarını ve gereksiz kolonları siler.
        LEAKAGE_DEFAULTS her zaman otomatik uygulanır.
        encoding_decisions.json'daki 'dusur' listesi eklenir.
"""

import pandas as pd
import json

# Her zaman silinecek leakage kolonları — unutulmaya karşı sabit liste
LEAKAGE_DEFAULTS = [
    'total_pymnt', 'total_rec_prncp', 'recoveries',
    'total_rec_int', 'total_rec_late_fee', 'last_pymnt_amnt',
    'collection_recovery_fee', 'debt_settlement_flag',
    'hardship_flag', 'hardship_status', 'out_prncp',
    'out_prncp_inv', 'total_pymnt_inv', 'loan_status',
    'last_pymnt_d', 'next_pymnt_d'
]


def drop_leakage(df: pd.DataFrame,
                 encoding_json: str = None,
                 ekstra_kolonlar: list = None) -> pd.DataFrame:
   
    silinecek = list(LEAKAGE_DEFAULTS)  

    # JSON'dan dusur listesini ekle
    if encoding_json:
        with open(encoding_json, encoding='utf-8') as f:
            enc = json.load(f)
        for kategori, liste in enc.get('dusur', {}).items():
            silinecek.extend(liste)

    # Ekstra kolonlar
    if ekstra_kolonlar:
        silinecek.extend(ekstra_kolonlar)

    silinecek   = list(set(silinecek))
    mevcut      = [c for c in silinecek if c in df.columns]
    bulunamayan = [c for c in silinecek if c not in df.columns]

    onceki = df.shape[1]
    df = df.drop(columns=mevcut)

    print("── Leakage & Gereksiz Kolon Temizleme ───────")
    print(f"   LEAKAGE_DEFAULTS     : {len(LEAKAGE_DEFAULTS)}")
    print(f"   Toplam silinecek     : {len(silinecek)}")
    print(f"   Silinen              : {len(mevcut)}")
    print(f"   Bulunamayan          : {len(bulunamayan)}")
    print(f"   Kolon: {onceki} → {df.shape[1]}")

    return df