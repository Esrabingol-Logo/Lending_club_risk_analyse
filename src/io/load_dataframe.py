"""
load_dataframe.py
─────────────────
Görev : infer_dtypes ile tahmin edilen dtype'ları kullanarak
        final DataFrame'i yükler.
        - Gereksiz kolonları okumaz (usecols)
        - Optimize dtype ile yükler
        - Herhangi bir veri setiyle çalışır

Kullanım:
    from src.io.load_dataframe import load_dataframe
"""

import pandas as pd
import time
from src.io.infer_dtypes import optimize_dtypes


# FINAL DATAFRAME YÜKLEME 

def load_dataframe(filepath: str,
                   usecols: list = None,
                   drop_cols: list = None,
                   optimize: bool = True,
                   nrows: int = None) -> pd.DataFrame:
    """
    CSV / CSV.GZ dosyasını optimize şekilde yükler.

    Args:
        filepath  : Dosya yolu
        usecols   : Sadece bu kolonları yükle (None = hepsi)
        drop_cols : Yükledikten sonra bu kolonları düşür
        optimize  : True → dtype optimizasyonu uygula
        nrows     : Max satır sayısı (None = hepsi)

    Returns:
        Yüklenmiş DataFrame
    """
    print(f"📂 DataFrame yükleniyor: {filepath}")
    baslangic = time.time()

    okuma_kwargs = {
        "low_memory"   : False,
        "on_bad_lines" : "skip",
    }
    if usecols:
        okuma_kwargs["usecols"] = usecols
    if nrows:
        okuma_kwargs["nrows"] = nrows

    df = pd.read_csv(filepath, **okuma_kwargs)

    
    if drop_cols:
        mevcutlar = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=mevcutlar)
        print(f"   {len(mevcutlar)} kolon düşürüldü")

    
    if optimize:
        df = optimize_dtypes(df, verbose=False)
        bellek = df.memory_usage(deep=True).sum() / 1024 ** 2
    else:
        bellek = df.memory_usage(deep=True).sum() / 1024 ** 2

    sure = time.time() - baslangic
    print(f"✅ Yüklendi!")
    print(f"   Satır  : {df.shape[0]:,}")
    print(f"   Sütun  : {df.shape[1]:,}")
    print(f"   Bellek : {bellek:.1f} MB")
    print(f"   Süre   : {sure:.1f} saniye")

    return df