"""
check_numeric_profile.py
────────────────────────
Görev : Sayısal kolonların istatistiksel profilini çıkarır.
        - Dağılım istatistikleri
        - Uç değer tespiti
        - Çarpıklık analizi
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.validation.check_numeric_profile import check_numeric_profile
"""

import pandas as pd
import numpy as np


def check_numeric_profile(df: pd.DataFrame,
                           hedef_kolon: str = 'target') -> pd.DataFrame:
    """
    Sayısal kolonların profilini çıkarır.

    Args:
        df           : DataFrame
        hedef_kolon  : Hedef kolon (profile dahil edilmez)

    Returns:
        Profil DataFrame
    """
    # Sayısal kolonları seç
    numerik = df.select_dtypes(include=[np.number]).columns.tolist()
    numerik = [c for c in numerik if c != hedef_kolon]

    rapor_listesi = []

    for kolon in numerik:
        seri = df[kolon].dropna()

        if len(seri) == 0:
            continue

        # IQR ile uç değer tespiti
        q1  = seri.quantile(0.25)
        q3  = seri.quantile(0.75)
        iqr = q3 - q1
        alt_sinir = q1 - 1.5 * iqr
        ust_sinir = q3 + 1.5 * iqr
        uc_deger_sayisi = ((seri < alt_sinir) | (seri > ust_sinir)).sum()
        uc_deger_orani  = uc_deger_sayisi / len(seri) * 100

        # Çarpıklık
        carpiklik = seri.skew()
        if abs(carpiklik) < 0.5:
            carpiklik_etiket = 'simetrik'
        elif abs(carpiklik) < 1.0:
            carpiklik_etiket = 'orta çarpık'
        else:
            carpiklik_etiket = 'yüksek çarpık'

        rapor_listesi.append({
            'kolon'          : kolon,
            'dtype'          : str(df[kolon].dtype),
            'eksik_oran'     : round(df[kolon].isnull().mean() * 100, 2),
            'ortalama'       : round(seri.mean(), 4),
            'medyan'         : round(seri.median(), 4),
            'std'            : round(seri.std(), 4),
            'min'            : round(seri.min(), 4),
            'max'            : round(seri.max(), 4),
            'q1'             : round(q1, 4),
            'q3'             : round(q3, 4),
            'carpiklik'      : round(carpiklik, 3),
            'carpiklik_tip'  : carpiklik_etiket,
            'uc_deger_oran'  : round(uc_deger_orani, 2),
        })

    rapor = pd.DataFrame(rapor_listesi)

    print("── Sayısal Kolon Profili ─────────────────────")
    print(f"   Toplam sayısal kolon : {len(rapor)}")
    print(f"   Simetrik             : {(rapor['carpiklik_tip']=='simetrik').sum()}")
    print(f"   Orta çarpık          : {(rapor['carpiklik_tip']=='orta çarpık').sum()}")
    print(f"   Yüksek çarpık        : {(rapor['carpiklik_tip']=='yüksek çarpık').sum()}")
    print(f"   Uç değer >%5 olan    : {(rapor['uc_deger_oran'] > 5).sum()}")

    return rapor