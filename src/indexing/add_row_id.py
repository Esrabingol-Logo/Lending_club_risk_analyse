"""
add_row_id.py
─────────────
Görev : Her satıra stabil, tekrar üretilebilir row_id ekler.
"""

import pandas as pd


def add_row_id(df: pd.DataFrame,
               id_col: str = 'row_id',
               start: int = 0) -> pd.DataFrame:
    if id_col in df.columns:
        print(f"ℹ️  '{id_col}' zaten mevcut, atlanıyor")
        return df

    df = df.reset_index(drop=True)

    # PerformanceWarning'i önlemek için concat kullanıyoruz
    id_series = pd.Series(range(start, start + len(df)),
                          name=id_col, dtype='int32')
    df = pd.concat([id_series, df], axis=1)

    print(f"✅ row_id eklendi: '{id_col}'")
    print(f"   Aralık: {start} → {start + len(df) - 1}")

    return df