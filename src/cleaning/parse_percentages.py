"""
parse_percentages.py
────────────────────
Görev : "13.56%" formatındaki string kolonları
        float'a dönüştürür.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.cleaning.parse_percentages import parse_percentages
"""

import pandas as pd


def parse_percentages(df: pd.DataFrame,
                      kolonlar: list,
                      bolme: bool = False) -> pd.DataFrame:
    

    print(" Yüzde Dönüşümü ")

    for kolon in kolonlar:
        if kolon not in df.columns:
            print(f"   ⚠️  {kolon} bulunamadı, atlandı")
            continue

        onceki_tip = df[kolon].dtype

        # String temizle ve dönüştür
        df[kolon] = (
            df[kolon]
            .astype(str)
            .str.strip()
            .str.replace('%', '', regex=False)
            .pipe(pd.to_numeric, errors='coerce')
        )

        if bolme:
            df[kolon] = df[kolon] / 100

        print(f"   ✅ {kolon:<20} "
              f"{str(onceki_tip):<10} → float32")
        df[kolon] = df[kolon].astype('float32')

    return df