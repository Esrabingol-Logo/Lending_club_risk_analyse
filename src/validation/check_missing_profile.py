"""
check_missing_profile.py
────────────────────────
Görev : Eksik değer profilini çıkarır.
        Her kolon için eksik oran ve strateji önerir.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.validation.check_missing_profile import check_missing_profile
"""

import pandas as pd


def check_missing_profile(df: pd.DataFrame,
                           esik_sil: float = 0.60,
                           esik_flag: float = 0.20) -> pd.DataFrame:
    """
    Eksik değer profilini çıkarır ve strateji önerir.

    Args:
        df         : DataFrame
        esik_sil   : Bu oranın üzerinde → sütunu sil
        esik_flag  : Bu ile esik_sil arası → flag + median

    Returns:
        Eksik değer rapor DataFrame
    """
    eksik_sayisi = df.isnull().sum()
    eksik_oran   = eksik_sayisi / len(df) * 100

    rapor = pd.DataFrame({
        'kolon'      : df.columns,
        'dtype'      : df.dtypes.values,
        'eksik_sayi' : eksik_sayisi.values,
        'eksik_oran' : eksik_oran.values.round(2)
    })

    # Strateji ata
    def strateji_belirle(oran):
        if oran > esik_sil * 100:
            return 'SİL'
        elif oran > esik_flag * 100:
            return 'FLAG + MEDIAN'
        elif oran > 0:
            return 'MEDIAN/MODE'
        else:
            return 'TAM'

    rapor['strateji'] = rapor['eksik_oran'].apply(strateji_belirle)
    rapor = rapor[rapor['eksik_sayi'] > 0]\
            .sort_values('eksik_oran', ascending=False)\
            .reset_index(drop=True)

    # Özet
    tam_kolon    = (eksik_sayisi == 0).sum()
    sil_kolon    = (rapor['strateji'] == 'SİL').sum()
    flag_kolon   = (rapor['strateji'] == 'FLAG + MEDIAN').sum()
    doldur_kolon = (rapor['strateji'] == 'MEDIAN/MODE').sum()

    print("── Eksik Değer Profili ──────────────────────")
    print(f"   Toplam kolon      : {df.shape[1]}")
    print(f"   ✅ Tam (eksik yok): {tam_kolon}")
    print(f"   ❌ Silinecek      : {sil_kolon}  (>%{esik_sil*100:.0f})")
    print(f"   🚩 Flag+Median    : {flag_kolon}  (%{esik_flag*100:.0f}-%{esik_sil*100:.0f})")
    print(f"   🔧 Doldurulacak   : {doldur_kolon}  (<%{esik_flag*100:.0f})")

    if len(rapor) > 0:
        print(f"\n   En çok eksik 10 kolon:")
        print(rapor[['kolon','eksik_oran','strateji']].head(10).to_string(index=False))

    return rapor