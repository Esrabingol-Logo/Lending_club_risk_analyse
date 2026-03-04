"""
encode_categoricals.py
──────────────────────
Görev:
    Kategorik kolonları encoding_decisions.json'a göre sayısala dönüştürür.

Desteklenen encoding tipleri:
    - ordinal              : mapping dict ile sıralı encoding
    - one_hot              : pd.get_dummies ile binary kolonlar
    - grupla_sonra_one_hot : önce gruplama, sonra one-hot

Notlar:
    - NaN değerler astype(str) ile "nan"e dönüştürülmez;
      pandas StringDtype kullanılarak korunur.
    - OOV (out-of-vocabulary) değerler -1 ile kodlanır ve loglanır.
    - One-hot encoding'de max_categories koruması vardır.
    - dummy_na=True ile NaN bilgisi one-hot'ta korunur.
    - drop_first=True multicollinearity önler (LR için önerilir,
      tree modeller için gerekli değil).
    - align_columns ile train/val/test kolon uyumsuzluğu önlenir.
    - "tarih_donustur" ve "dusur" JSON anahtarları bu modülde
      işlenmez; ilgili cleaning modüllerine devredilir.

Kullanım:
    from src.features.encode_categoricals import encode_categoricals
    from src.features.encode_categoricals import align_columns
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# One-hot'ta izin verilen maksimum benzersiz değer sayısı
_MAX_CATEGORIES: int = 50


# YARDIMCI FONKSİYONLAR 

def _ensure_dataframe(df: pd.DataFrame) -> None:
    """
    DataFrame tip ve None kontrolü.

    Raises:
        ValueError : df None ise
        TypeError  : df DataFrame değilse
    """
    if df is None:
        raise ValueError("df is None. DataFrame bekleniyor.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got: {type(df)}"
        )


def _load_encoding_json(path: str) -> Dict[str, Any]:
   
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"encoding_decisions.json bulunamadı: '{path}'. "
            f"Dosya yolunu kontrol edin."
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"encoding_decisions.json geçersiz JSON: {e}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"encoding_decisions.json dict bekleniyor, "
            f"got: {type(data)}"
        )
    return data


def _strip_preserve_na(seri: pd.Series) -> pd.Series:
   
    return seri.astype('string').str.strip()


# 1. ORDİNAL ENCODİNG 

def _apply_ordinal(
    df: pd.DataFrame,
    kolon: str,
    mapping: Dict[str, int],
    oov_value: int = -1
) -> pd.DataFrame:

    # NaN koruyarak strip
    temiz = _strip_preserve_na(df[kolon])

    # OOV tespiti — NaN olmayan ama mapping'de de olmayan değerler
    notna_mask = temiz.notna()
    oov_mask   = notna_mask & ~temiz.isin(mapping.keys())
    oov_sayisi = int(oov_mask.sum())

    if oov_sayisi > 0:
        oov_degerler = temiz[oov_mask].unique().tolist()
        logger.warning(
            "Ordinal OOV — %s: %d değer mapping'de yok "
            "(%s...), oov_value=%d atandı",
            kolon, oov_sayisi,
            oov_degerler[:5], oov_value
        )

    # Mapping uygula — .where() ile SettingWithCopyWarning önlenir
    mapped = temiz.map(mapping)
    mapped = mapped.where(~oov_mask, other=oov_value)

    df[kolon] = pd.to_numeric(mapped, errors='coerce')\
                  .astype('float32')

    logger.info(
        "Ordinal encoding: %s | mapping=%d kategori | OOV=%d",
        kolon, len(mapping), oov_sayisi
    )
    return df


# 2. ONE-HOT ENCODİNG 

def _apply_onehot(
    df: pd.DataFrame,
    kolon: str,
    max_categories: int = _MAX_CATEGORIES,
    dummy_na: bool = True,
    drop_first: bool = False
) -> pd.DataFrame:
    """
    Tek bir kolona one-hot encoding uygular.
    """
    n_unique = df[kolon].nunique()

    if n_unique > max_categories:
        raise ValueError(
            f"One-hot kardinalite hatası — '{kolon}': "
            f"{n_unique} benzersiz değer var, "
            f"max_categories={max_categories}. "
            f"Bu kolonu grupla_sonra_one_hot veya "
            f"target encoding ile işleyin."
        )

    dummies = pd.get_dummies(
        df[kolon],
        prefix=kolon,
        drop_first=drop_first,
        dummy_na=dummy_na,
        dtype='int8'
    )

    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[kolon])

    logger.info(
        "One-hot encoding: %s → %s (drop_first=%s)",
        kolon, dummies.columns.tolist(), drop_first
    )
    return df


# 3. GRUPLA + ONE-HOT 

def _apply_group_onehot(
    df: pd.DataFrame,
    kolon: str,
    gruplar: Dict[str, List[str]],
    max_categories: int = _MAX_CATEGORIES,
    dummy_na: bool = True,
    drop_first: bool = False
) -> pd.DataFrame:
    """
    Kolonu önce gruplara ayırır, sonra one-hot uygular.

    Mapping'de olmayan değerler 'diger' grubuna atanır.
    SettingWithCopyWarning önlemek için .where() kullanılır.
    """
    # Ters mapping: değer → grup adı
    ters_map: Dict[str, str] = {}
    for grup_adi, degerler in gruplar.items():
        for d in degerler:
            ters_map[d] = grup_adi

    # NaN koruyarak gruplama
    grup_serisi = df[kolon].map(ters_map)

    
    oov_mask   = df[kolon].notna() & grup_serisi.isna()
    oov_sayisi = int(oov_mask.sum())

    if oov_sayisi > 0:
        oov_degerler = df[kolon][oov_mask].unique().tolist()
        logger.warning(
            "Grup OOV — %s: %d değer gruplarda yok "
            "(%s...), 'diger' atandı",
            kolon, oov_sayisi, oov_degerler[:5]
        )

    grup_serisi = grup_serisi.where(~oov_mask, other='diger')

    grup_kolon = f"{kolon}_GRUP"
    df[grup_kolon] = grup_serisi.astype('category')

    # One-hot uygula
    dummies = pd.get_dummies(
        df[grup_kolon],
        prefix=kolon,
        drop_first=drop_first,
        dummy_na=dummy_na,
        dtype='int8'
    )

    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[kolon, grup_kolon])

    logger.info(
        "Grupla+One-hot: %s → %s | OOV=%d (drop_first=%s)",
        kolon, dummies.columns.tolist(), oov_sayisi, drop_first
    )
    return df


# 4. ANA FONKSİYON 

def encode_categoricals(
    df: pd.DataFrame,
    encoding_json: str,
    oov_value: int = -1,
    max_categories: int = _MAX_CATEGORIES,
    dummy_na: bool = True,
    drop_first: bool = False,
    inplace: bool = True
) -> pd.DataFrame:
   
    _ensure_dataframe(df)
    out = df if inplace else df.copy()

    enc = _load_encoding_json(encoding_json)

    # 1. ORDİNAL 
    ordinal_sayac = 0
    for kolon, mapping in enc.get('ordinal', {}).items():
        if kolon not in out.columns:
            logger.warning("Ordinal kolon yok: %s", kolon)
            continue
        out = _apply_ordinal(out, kolon, mapping, oov_value)
        ordinal_sayac += 1

    # 2. ONE-HOT 
    onehot_sayac = 0
    for kolon in enc.get('one_hot', []):
        if kolon not in out.columns:
            logger.warning("One-hot kolon yok: %s", kolon)
            continue
        out = _apply_onehot(
            out, kolon, max_categories, dummy_na, drop_first
        )
        onehot_sayac += 1

    # 3. GRUPLA + ONE-HOT 
    grupla_sayac = 0
    for kolon, gruplar in enc.get('grupla_sonra_one_hot', {}).items():
        if kolon not in out.columns:
            logger.warning("Grupla kolon yok: %s", kolon)
            continue
        out = _apply_group_onehot(
            out, kolon, gruplar, max_categories, dummy_na, drop_first
        )
        grupla_sayac += 1

    logger.info(
        "Encoding tamamlandı — Ordinal: %d | One-hot: %d | "
        "Grupla+One-hot: %d | drop_first=%s | Final shape: %s",
        ordinal_sayac, onehot_sayac, grupla_sayac,
        drop_first, out.shape
    )

    return out


# 5. KOLON HİZALAMA (TRAIN / VAL / TEST) 

def align_columns(
    df: pd.DataFrame,
    train_columns: List[str],
    fill_value: int = 0
) -> pd.DataFrame:

    _ensure_dataframe(df)

   
    out = df.copy()

    eksik = [c for c in train_columns if c not in out.columns]
    fazla = [c for c in out.columns   if c not in train_columns]

    # Eksik kolonları ekle
    for c in eksik:
        out[c] = fill_value

    # Fazla kolonları düşür
    if fazla:
        out = out.drop(columns=fazla)

    # Sıralamayı train ile eşitle
    out = out[train_columns]

    if eksik or fazla:
        logger.info(
            "align_columns — eklenen: %d | düşürülen: %d | "
            "final: %d kolon",
            len(eksik), len(fazla), len(train_columns)
        )
        if eksik:
            logger.info("  Eklenen kolonlar: %s", eksik)
        if fazla:
            logger.info("  Düşürülen kolonlar: %s", fazla)
    else:
        logger.info(
            "align_columns — kolon seti zaten uyumlu "
            "(%d kolon)", len(train_columns)
        )

    return out