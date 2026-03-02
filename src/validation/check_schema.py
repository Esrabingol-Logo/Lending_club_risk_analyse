"""
check_schema.py
───────────────
Görev : Beklenen kolonların DataFrame'de var olup olmadığını kontrol eder.
        Eksik kolonları raporlar.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.validation.check_schema import check_schema
"""

import pandas as pd


def check_schema(df: pd.DataFrame,
                 zorunlu_kolonlar: list) -> dict:
    """
    Zorunlu kolonların varlığını kontrol eder.

    Args:
        df               : DataFrame
        zorunlu_kolonlar : Bulunması gereken kolon listesi

    Returns:
        {
          'gecti'         : True/False,
          'mevcut'        : [...],
          'eksik'         : [...],
          'toplam_kolon'  : int
        }
    """
    mevcut   = [c for c in zorunlu_kolonlar if c in df.columns]
    eksik    = [c for c in zorunlu_kolonlar if c not in df.columns]

    gecti = len(eksik) == 0

    print("── Schema Kontrolü ──────────────────────────")
    print(f"   Toplam kolon      : {df.shape[1]}")
    print(f"   Kontrol edilen    : {len(zorunlu_kolonlar)}")
    print(f"   Bulunan           : {len(mevcut)}")

    if eksik:
        print(f"   ❌ Eksik kolon    : {len(eksik)}")
        for e in eksik:
            print(f"      → {e}")
    else:
        print(f"   ✅ Tüm kolonlar mevcut")

    return {
        'gecti'        : gecti,
        'mevcut'       : mevcut,
        'eksik'        : eksik,
        'toplam_kolon' : df.shape[1]
    }