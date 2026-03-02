import pandas as pd
import numpy as np


def infer_dtypes(df: pd.DataFrame) -> dict:
    dtype_map = {}
    for kolon in df.columns:
        seri = df[kolon]
        mevcut_tip = seri.dtype
        if pd.api.types.is_integer_dtype(mevcut_tip):
            min_val = seri.min()
            max_val = seri.max()
            if min_val >= -32768 and max_val <= 32767:
                dtype_map[kolon] = 'int16'
            elif min_val >= -2147483648 and max_val <= 2147483647:
                dtype_map[kolon] = 'int32'
            else:
                dtype_map[kolon] = 'int64'
        elif pd.api.types.is_float_dtype(mevcut_tip):
            dtype_map[kolon] = 'float32'
        elif pd.api.types.is_object_dtype(mevcut_tip):
            benzersiz_oran = seri.nunique() / len(seri)
            if benzersiz_oran < 0.50:
                dtype_map[kolon] = 'category'
            else:
                dtype_map[kolon] = 'object'
        else:
            dtype_map[kolon] = str(mevcut_tip)
    return dtype_map


def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    onceki_bellek = df.memory_usage(deep=True).sum() / 1024 ** 2
    dtype_map = infer_dtypes(df)
    for kolon, yeni_tip in dtype_map.items():
        try:
            if yeni_tip == 'category':
                df[kolon] = df[kolon].astype('category')
            elif yeni_tip in ['int16', 'int32', 'int64']:
                df[kolon] = pd.to_numeric(df[kolon], errors='coerce').astype(yeni_tip)
            elif yeni_tip == 'float32':
                df[kolon] = pd.to_numeric(df[kolon], errors='coerce').astype('float32')
        except Exception:
            pass
    sonraki_bellek = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        azalma = (1 - sonraki_bellek / onceki_bellek) * 100
        print(f"✅ Dtype optimizasyonu tamamlandı")
        print(f"   Önceki bellek : {onceki_bellek:.1f} MB")
        print(f"   Sonraki bellek: {sonraki_bellek:.1f} MB")
        print(f"   Tasarruf      : %{azalma:.1f}")
    return df


def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    rapor = pd.DataFrame({
        'kolon'      : df.columns,
        'dtype'      : df.dtypes.values,
        'benzersiz'  : [df[c].nunique() for c in df.columns],
        'eksik_oran' : (df.isnull().sum() / len(df) * 100).values.round(2),
        'ornek_deger': [df[c].dropna().iloc[0]
                        if df[c].notna().any() else None
                        for c in df.columns]
    })
    rapor = rapor.sort_values('eksik_oran', ascending=False).reset_index(drop=True)
    print(f"✅ Dtype raporu hazır — {len(rapor)} kolon")
    return rapor