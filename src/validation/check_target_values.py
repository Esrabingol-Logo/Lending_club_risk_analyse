"""
check_target_values.py
──────────────────────
Görev : Hedef değişkenin beklenen değerleri içerip içermediğini kontrol eder.
        Sınıf dağılımını raporlar.
        Herhangi bir veri setiyle çalışır.

Kullanım:
    from src.validation.check_target_values import check_target_values
"""

import pandas as pd


def check_target_values(df: pd.DataFrame,
                        hedef_kolon: str,
                        pozitif_sinif: list,
                        negatif_sinif: list,
                        gozard_et: list = None) -> dict:
    """
    Hedef değişkeni kontrol eder ve binary target oluşturur.

    Args:
        df            : DataFrame
        hedef_kolon   : Hedef kolon adı
        pozitif_sinif : 1 olarak kodlanacak değerler
        negatif_sinif : 0 olarak kodlanacak değerler
        gozard_et     : Veri setinden çıkarılacak değerler

    Returns:
        {
          'gecti'         : True/False,
          'sinif_dagilimi': {...},
          'default_orani' : float,
          'df'            : temizlenmiş DataFrame
        }
    """
    if hedef_kolon not in df.columns:
        print(f"❌ Hedef kolon bulunamadı: {hedef_kolon}")
        return {'gecti': False, 'df': df}

    print("── Hedef Değişken Kontrolü ──────────────────")

    # Mevcut değerleri göster
    mevcut_degerler = df[hedef_kolon].value_counts()
    print(f"\n   Mevcut değerler:")
    for val, cnt in mevcut_degerler.items():
        oran = cnt / len(df) * 100
        print(f"   {str(val):<45} : {cnt:>10,}  (%{oran:.1f})")

    # Göz ardı edilecekleri çıkar
    if gozard_et:
        onceki = len(df)
        df = df[~df[hedef_kolon].isin(gozard_et)]
        sonraki = len(df)
        print(f"\n   Çıkarılan satır  : {onceki - sonraki:,} "
              f"({gozard_et})")

    # Binary target oluştur
    tum_beklenen = pozitif_sinif + negatif_sinif
    beklenmeyen  = [v for v in df[hedef_kolon].unique()
                    if v not in tum_beklenen]

    if beklenmeyen:
        print(f"\n   ⚠️  Beklenmeyen değerler: {beklenmeyen}")

    df['target'] = df[hedef_kolon].apply(
        lambda x: 1 if x in pozitif_sinif else
                  0 if x in negatif_sinif else None
    )

    df = df[df['target'].notna()].copy()
    df['target'] = df['target'].astype('int8')

    # Sınıf dağılımı
    toplam        = len(df)
    pozitif_sayı  = (df['target'] == 1).sum()
    negatif_sayı  = (df['target'] == 0).sum()
    default_orani = pozitif_sayı / toplam * 100

    print(f"\n   Binary target oluşturuldu:")
    print(f"   0 (negatif) : {negatif_sayı:>10,}  (%{100-default_orani:.1f})")
    print(f"   1 (pozitif) : {pozitif_sayı:>10,}  (%{default_orani:.1f})")
    print(f"\n   ⚠️  Default oranı: %{default_orani:.2f}")

    if default_orani < 5:
        print(f"   ⚠️  Çok dengesiz veri! SMOTE veya class_weight gerekebilir")
    elif default_orani < 30:
        print(f"   ℹ️  Dengesiz veri — class_weight önerilir")
    else:
        print(f"   ✅ Dengeli veri")

    return {
        'gecti'         : True,
        'sinif_dagilimi': {'pozitif': int(pozitif_sayı),
                           'negatif': int(negatif_sayı)},
        'default_orani' : round(default_orani, 2),
        'df'            : df
    }