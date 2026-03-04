"""
interaction_features.py
───────────────────────
Görev:
    Finansal oranlar ve etkileşim (interaction) özellikleri üretir.

Üretilen özellikler:
    Finansal oranlar:
        - income_to_loan_ratio        : annual_inc / loan_amnt
        - installment_to_income       : installment / (annual_inc/12)
        - annual_interest_amount_approx: loan_amnt * int_rate/100
        - revol_bal_to_income         : revol_bal / annual_inc
        - fico_avg                    : (fico_low + fico_high) / 2
        - fico_spread                 : fico_high - fico_low
        - dti_ratio                   : dti / 100 (opsiyonel)
        - total_debt_to_income        : total_bal_ex_mort / annual_inc
        - revol_usage_ratio           : revol_bal / tot_hi_cred_lim
        - open_acc_ratio              : open_acc / total_acc

    Etkileşim özellikleri:
        - fico_x_intrate              : fico_avg * int_rate
        - dti_x_intrate               : dti * int_rate
        - revol_util_sq               : revol_util ^ 2
        - delinq_x_inq                : delinq_2yrs * inq_last_6mths
        - credit_age_x_fico           : earliest_cr_line_AY_FARK * fico_avg

Notlar:
    - int_rate: yüzde puanı ölçeğinde beklenir (örn: 13.99)
    - dti: yüzde puanı gibi değerler (örn: 16.06)
    - _safe_divide ile sıfıra/NaN'a bölme koruması vardır.
    - build_features() bağımlılık sırasını garanti eder.
    - Tree modeller ölçekten çok etkilenmez;
      amaç anlamlı feature üretmektir.

Kullanım:
    from src.features.interaction_features import build_features
    from src.features.interaction_features import (
        add_financial_ratios,
        add_interaction_features
    )
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = 1e-9
) -> pd.Series:
    """
    Güvenli bölme: denom 0/NaN olduğunda NaN döndürür.
    eps sadece sayısal stabilite içindir — bias eklemez.
    """
    denom = denominator.astype('float64')
    num   = numerator.astype('float64')

    out = np.where(
        np.isfinite(denom) & (np.abs(denom) > eps),
        num / denom,
        np.nan
    )
    return pd.Series(out, index=numerator.index)


# 1. FİNANSAL ORANLAR 

def add_financial_ratios(
    df: pd.DataFrame,
    *,
    create_dti_ratio: bool = True,
    inplace: bool = True
) -> pd.DataFrame:

    _ensure_dataframe(df)
    out = df if inplace else df.copy()

    added: List[str] = []

    # Mevcut oranlar 

    # Gelir / Kredi oranı — ödeme kapasitesi
    if {'annual_inc', 'loan_amnt'}.issubset(out.columns):
        out['income_to_loan_ratio'] = _safe_divide(
            out['annual_inc'], out['loan_amnt']
        ).astype('float32')
        added.append('income_to_loan_ratio')

    # Taksit / Aylık gelir — aylık yük oranı
    if {'installment', 'annual_inc'}.issubset(out.columns):
        monthly_inc = out['annual_inc'] / 12.0
        out['installment_to_income'] = _safe_divide(
            out['installment'], monthly_inc
        ).astype('float32')
        added.append('installment_to_income')

    # Yaklaşık yıllık faiz tutarı
    if {'loan_amnt', 'int_rate'}.issubset(out.columns):
        out['annual_interest_amount_approx'] = (
            out['loan_amnt'].astype('float64')
            * (out['int_rate'].astype('float64') / 100.0)
        ).astype('float32')
        added.append('annual_interest_amount_approx')

    # Revolver bakiye / Yıllık gelir
    if {'revol_bal', 'annual_inc'}.issubset(out.columns):
        out['revol_bal_to_income'] = _safe_divide(
            out['revol_bal'], out['annual_inc']
        ).astype('float32')
        added.append('revol_bal_to_income')

    # FICO ortalama ve aralık genişliği
    if {'fico_range_low', 'fico_range_high'}.issubset(out.columns):
        low  = out['fico_range_low'].astype('float64')
        high = out['fico_range_high'].astype('float64')
        out['fico_avg']    = ((low + high) / 2.0).astype('float32')
        out['fico_spread'] = (high - low).astype('float32')
        added.extend(['fico_avg', 'fico_spread'])

    # DTI oranı — opsiyonel
    if create_dti_ratio and 'dti' in out.columns:
        out['dti_ratio'] = (
            out['dti'].astype('float64') / 100.0
        ).astype('float32')
        added.append('dti_ratio')

    # Yeni oranlar 

    # Toplam borç / Gelir — mortgage hariç toplam borç yükü
    # total_bal_ex_mort: mortgage dışı toplam bakiye
    if {'total_bal_ex_mort', 'annual_inc'}.issubset(out.columns):
        out['total_debt_to_income'] = _safe_divide(
            out['total_bal_ex_mort'], out['annual_inc']
        ).astype('float32')
        added.append('total_debt_to_income')

    # Gerçek kredi kullanım oranı — revol_bal / toplam kredi limiti
    # tot_hi_cred_lim: tüm kredi hesaplarının toplam limiti
    # revol_util'den farkı: sadece revolving değil tüm limitler
    if {'revol_bal', 'tot_hi_cred_lim'}.issubset(out.columns):
        out['revol_usage_ratio'] = _safe_divide(
            out['revol_bal'], out['tot_hi_cred_lim']
        ).astype('float32')
        added.append('revol_usage_ratio')

    # Hesap aktiflik oranı — açık hesap / toplam hesap
    # Yüksek oran → aktif kredi kullanımı → risk göstergesi
    if {'open_acc', 'total_acc'}.issubset(out.columns):
        out['open_acc_ratio'] = _safe_divide(
            out['open_acc'], out['total_acc']
        ).astype('float32')
        added.append('open_acc_ratio')

    logger.info(
        "Financial ratios added: %d -> %s", len(added), added
    )
    return out


# 2. ETKİLEŞİM ÖZELLİKLERİ 

def add_interaction_features(
    df: pd.DataFrame,
    *,
    require_fico_avg: bool = True,
    create_credit_age_x_fico: bool = True,
    inplace: bool = True
) -> pd.DataFrame:

    _ensure_dataframe(df)
    out = df if inplace else df.copy()

    added: List[str] = []

    # fico_avg bağımlılığını garanti et
    if (require_fico_avg
            and 'fico_avg' not in out.columns
            and {'fico_range_low', 'fico_range_high'}.issubset(
                out.columns)):
        low  = out['fico_range_low'].astype('float64')
        high = out['fico_range_high'].astype('float64')
        out['fico_avg'] = ((low + high) / 2.0).astype('float32')
        logger.info("fico_avg interaction içinde hesaplandı")
        added.append('fico_avg')

    # FICO_AVG x Faiz — avg daha az gürültülü
    if {'fico_avg', 'int_rate'}.issubset(out.columns):
        out['fico_x_intrate'] = (
            out['fico_avg'].astype('float64')
            * out['int_rate'].astype('float64')
        ).astype('float32')
        added.append('fico_x_intrate')

    # DTI x Faiz — yüksek borç + yüksek faiz riski
    if {'dti', 'int_rate'}.issubset(out.columns):
        out['dti_x_intrate'] = (
            out['dti'].astype('float64')
            * out['int_rate'].astype('float64')
        ).astype('float32')
        added.append('dti_x_intrate')

    # Revol util kare — yüksek kullanımın nonlineer etkisi
    if 'revol_util' in out.columns:
        out['revol_util_sq'] = (
            out['revol_util'].astype('float64') ** 2
        ).astype('float32')
        added.append('revol_util_sq')

    # Geçmiş gecikme x Yeni sorgu
    if {'delinq_2yrs', 'inq_last_6mths'}.issubset(out.columns):
        out['delinq_x_inq'] = (
            out['delinq_2yrs'].astype('float64')
            * out['inq_last_6mths'].astype('float64')
        ).astype('float32')
        added.append('delinq_x_inq')

    # Kredi yaşı x FICO — ölçekleme sabiti yok, tree için gereksiz
    if (create_credit_age_x_fico
            and {'earliest_cr_line_AY_FARK', 'fico_avg'}.issubset(
                out.columns)):
        out['credit_age_x_fico'] = (
            out['earliest_cr_line_AY_FARK'].astype('float64')
            * out['fico_avg'].astype('float64')
        ).astype('float32')
        added.append('credit_age_x_fico')

    logger.info(
        "Interaction features added: %d -> %s", len(added), added
    )
    return out


# 3. TEK GİRİŞ NOKTASI 

def build_features(
    df: pd.DataFrame,
    *,
    create_dti_ratio: bool = True,
    create_credit_age_x_fico: bool = True,
    inplace: bool = True
) -> pd.DataFrame:

    _ensure_dataframe(df)

    df = add_financial_ratios(
        df,
        create_dti_ratio=create_dti_ratio,
        inplace=inplace
    )
    df = add_interaction_features(
        df,
        create_credit_age_x_fico=create_credit_age_x_fico,
        inplace=inplace
    )
    return df