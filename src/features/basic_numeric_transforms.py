"""
basic_numeric_transforms.py
───────────────────────────
Görev:
    Sayısal kolonlara log1p transform ve binning uygular.
    Yüksek çarpıklığı azaltır, aralık bazlı gruplar oluşturur.

Üretilen özellikler:
    - {kolon}_LOG : log1p transform uygulanmış kolon
    - {bin_adi}   : pd.cut ile aralıklara bölünmüş kategorik kolon

Notlar:
    - Log transform orijinal kolonu korur, yeni _LOG kolonu ekler.
    - Negatif shift: x - min_val + eps ile numerik stabilite sağlanır.
    - Shift değerleri train setinden hesaplanır ve artifact olarak
      kaydedilir; val/test'e aynı shift uygulanır (leakage önleme).
    - Binning çıktısı kategorik (category dtype) döner.
      Encoding aşamasında (encode_categoricals.py) one-hot veya
      ordinal olarak ele alınmalıdır.
    - Log transform tree modeller için sınırlı katkı sağlar;
      LR pipeline'ında daha faydalıdır.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numerik stabilite için sabit epsilon
_EPS: float = 1e-9


# YARDIMCI FONKSİYONLAR 

def _ensure_dataframe(df: pd.DataFrame) -> None:
   
    if df is None:
        raise ValueError("df is None. DataFrame bekleniyor.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got: {type(df)}"
        )


def _load_json_list(path: str) -> List[str]:
 
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: '{path}'. "
            f"Dosya yolunu kontrol edin."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Geçersiz JSON formatı: {e}")

    if isinstance(data, dict):
        raise ValueError(
            f"'{path}' bir dict döndürdü, list bekleniyor. "
            f"Keys: {list(data.keys())}"
        )
    if not isinstance(data, list):
        raise ValueError(
            f"'{path}' geçersiz format: {type(data)}. "
            f"List bekleniyor."
        )
    return data


# 1. LOG TRANSFORM — FIT 

def fit_log_transforms(
    df_train: pd.DataFrame,
    kolonlar: Optional[List[str]] = None,
    log_json: Optional[str] = None,
    artifact_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Train seti üzerinden shift değerlerini hesaplar.
    Negatif değer varsa shift = -min_val + eps, yoksa 0.

    Bu fonksiyon sadece TRAIN setinde çağrılır.
    Dönen shift_map hem train hem val/test'e uygulanır.

    """
    _ensure_dataframe(df_train)

    # Kolon listesini belirle
    if kolonlar is None and log_json is not None:
        kolonlar = _load_json_list(log_json)

    if not kolonlar:
        logger.warning(
            "fit_log_transforms: kolon listesi boş, atlanıyor"
        )
        return {}

    # Veri setinde mevcut ve sayısal olanları filtrele
    mevcut = [
        c for c in kolonlar
        if c in df_train.columns
        and pd.api.types.is_numeric_dtype(df_train[c])
    ]

    eksik = [c for c in kolonlar if c not in df_train.columns]
    if eksik:
        logger.warning(
            "fit_log_transforms — veri setinde yok: %s", eksik
        )

    shift_map: Dict[str, float] = {}

    for kolon in mevcut:
        # skipna=True ile NaN güvenli min hesabı — sadece train'den
        min_val = float(np.nanmin(df_train[kolon].values))

        if min_val < 0:
            shift = -min_val + _EPS
            logger.info(
                "fit — negatif shift: %s | min=%.4f | shift=%.6f",
                kolon, min_val, shift
            )
        else:
            shift = 0.0

        shift_map[kolon] = shift

    # Artifact olarak kaydet
    if artifact_path is not None:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(shift_map, f, indent=2)
        logger.info(
            "Shift map kaydedildi: %s (%d kolon)",
            artifact_path, len(shift_map)
        )

    logger.info(
        "fit_log_transforms tamamlandı: %d kolon",
        len(shift_map)
    )
    return shift_map


# 2. LOG TRANSFORM — APPLY 

def apply_log_transforms(
    df: pd.DataFrame,
    shift_map: Dict[str, float],
    inplace: bool = True
) -> pd.DataFrame:
   
    _ensure_dataframe(df)

    if not shift_map:
        logger.warning(
            "apply_log_transforms: shift_map boş, atlanıyor"
        )
        return df

    out     = df if inplace else df.copy()
    eklenen: List[str] = []

    for kolon, shift in shift_map.items():
        if kolon not in out.columns:
            logger.warning(
                "apply — kolon yok, atlandı: %s", kolon
            )
            continue

        if not pd.api.types.is_numeric_dtype(out[kolon]):
            logger.warning(
                "apply — sayısal değil, atlandı: %s "
                "(dtype=%s)", kolon, out[kolon].dtype
            )
            continue

        yeni_kolon = f"{kolon}_LOG"

        if shift > 0:
            out[yeni_kolon] = np.log1p(
                out[kolon].astype('float64') + shift
            ).astype('float32')
            logger.info(
                "Log transform (shift=%.6f): %s → %s",
                shift, kolon, yeni_kolon
            )
        else:
            out[yeni_kolon] = np.log1p(
                out[kolon].astype('float64')
            ).astype('float32')
            logger.info(
                "Log transform: %s → %s", kolon, yeni_kolon
            )

        eklenen.append(yeni_kolon)

    logger.info(
        "apply_log_transforms tamamlandı: %d özellik -> %s",
        len(eklenen), eklenen
    )
    return out


# 3. SHIFT MAP — ARTIFACT'TAN YÜKLE 

def load_shift_map(artifact_path: str) -> Dict[str, float]:
    
    try:
        with open(artifact_path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Shift map bulunamadı: '{artifact_path}'. "
            f"Önce fit_log_transforms() çalıştırın."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Geçersiz JSON formatı: {e}")

    if not isinstance(data, dict):
        raise ValueError(
            f"Shift map dict bekleniyor, got: {type(data)}"
        )

    logger.info(
        "Shift map yüklendi: %s (%d kolon)",
        artifact_path, len(data)
    )
    return {k: float(v) for k, v in data.items()}


# 4. BİNNİNG 

def apply_binning(
    df: pd.DataFrame,
    kolon: str,
    bin_adi: str,
    bins: List[float],
    labels: List[str],
    inplace: bool = True
) -> pd.DataFrame:
    """
    Sayısal kolonu kategorik gruplara böler (pd.cut).
    """
    _ensure_dataframe(df)

    # Kolon varlık kontrolü
    if kolon not in df.columns:
        raise ValueError(
            f"'{kolon}' kolonu DataFrame'de bulunamadı. "
            f"Mevcut kolonlar: {df.columns.tolist()}"
        )

    # Kolon sayısal mı?
    if not pd.api.types.is_numeric_dtype(df[kolon]):
        raise TypeError(
            f"'{kolon}' sayısal değil: {df[kolon].dtype}. "
            f"Binning sadece sayısal kolonlara uygulanır."
        )

    # bins / labels uyumu
    beklenen_label_sayisi = len(bins) - 1
    if len(labels) != beklenen_label_sayisi:
        raise ValueError(
            f"bins/labels uyumsuz: "
            f"len(bins)-1={beklenen_label_sayisi}, "
            f"len(labels)={len(labels)}. "
            f"labels tam olarak "
            f"{beklenen_label_sayisi} eleman içermeli."
        )

    out = df if inplace else df.copy()

    out[bin_adi] = pd.cut(
        out[kolon],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype('category')

    dagilim = out[bin_adi].value_counts().sort_index()

    logger.info(
        "Binning: %s → %s | Dağılım: %s",
        kolon, bin_adi, dagilim.to_dict()
    )

    return out