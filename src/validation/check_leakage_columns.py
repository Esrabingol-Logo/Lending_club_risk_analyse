"""
check_leakage_columns.py
────────────────────────
Görev : Data leakage kolonlarını tespit eder ve raporlar.
        Silmez — sadece raporlar. Silme işlemi cleaning/drop_leakage.py'da.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.validation.check_leakage_columns import check_leakage_columns
"""

import pandas as pd


def check_leakage_columns(df: pd.DataFrame,
                           leakage_listesi: list) -> dict:
    """
    Leakage kolonlarını tespit eder, raporlar.

    Args:
        df              : DataFrame
        leakage_listesi : Leakage olduğu bilinen kolon listesi

    Returns:
        {
          'bulunan'    : [...],
          'bulunamayan': [...],
          'oran'       : float
        }
    """
    bulunan     = [c for c in leakage_listesi if c in df.columns]
    bulunamayan = [c for c in leakage_listesi if c not in df.columns]

    print("── Leakage Kolon Kontrolü ───────────────────")
    print(f"   Kontrol edilen : {len(leakage_listesi)}")
    print(f"   Veri setinde   : {len(bulunan)}")
    print(f"   Bulunamayan    : {len(bulunamayan)}")

    if bulunan:
        print(f"\n   ⛔ Silinmesi gereken leakage kolonlar:")
        for k in bulunan:
            print(f"      ❌ {k}")
        print(f"\n   → cleaning/drop_leakage.py ile silinecek")
    else:
        print(f"   ✅ Leakage kolon bulunamadı")

    return {
        'bulunan'    : bulunan,
        'bulunamayan': bulunamayan,
        'oran'       : len(bulunan) / df.shape[1] * 100
    }