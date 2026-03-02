"""
set_primary_key.py
──────────────────
Görev : Veri setinin kendi id kolonunu standartlaştırır.
        - Duplicate kontrolü
        - Null kontrolü
        - Standart 'pk' adına rename

Kullanım:
    from src.indexing.set_primary_key import set_primary_key
"""

import pandas as pd


def set_primary_key(df: pd.DataFrame,
                    id_kolon: str,
                    yeni_ad: str = 'pk') -> pd.DataFrame:
    """
    Veri setinin id kolonunu standartlaştırır.

    Args:
        df       : DataFrame
        id_kolon : Mevcut id kolon adı
        yeni_ad  : Standart kolon adı

    Returns:
        Standartlaştırılmış DataFrame
    """
    if id_kolon not in df.columns:
        print(f"⚠️  '{id_kolon}' kolonu bulunamadı")
        return df

    # Null kontrolü
    null_sayisi = df[id_kolon].isnull().sum()
    if null_sayisi > 0:
        print(f"⚠️  {null_sayisi} adet null id bulundu")

    # Duplicate kontrolü
    dup_sayisi = df[id_kolon].duplicated().sum()
    if dup_sayisi > 0:
        print(f"⚠️  {dup_sayisi} adet duplicate id bulundu")
    else:
        print(f"✅ Tüm id'ler benzersiz")

    # Rename
    if id_kolon != yeni_ad:
        df = df.rename(columns={id_kolon: yeni_ad})
        print(f"✅ '{id_kolon}' → '{yeni_ad}' olarak yeniden adlandırıldı")

    return df