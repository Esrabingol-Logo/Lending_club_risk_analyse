"""
feature_selection.py
────────────────────
Görev:
    6 aşamalı feature selection pipeline uygular.
    Tüm kararlar sadece train seti üzerinde alınır (leakage yok).

Aşamalar:
    1. Missing Ratio      : missing > 0.95 → drop
    2. Variance Threshold : var < 0.01 → drop (NaN guard dahil)
    3. Yüksek Korelasyon  : |r| > 0.95 → drop (vektörel, O(n²) değil)
    4. Information Value  : IV < 0.02 → drop
                            numeric   → qcut binning → WoE → IV
                            categorical → direct IV (kardinalite korumalı)
                            WoE standardı: log(dist_good / dist_bad)
                            Laplace smoothing: sıfır dist sorunu yok
    5. LightGBM Importance: importance == 0 → kesin drop
                            alt %25 (sıfır olmayanlardan) → drop
    6. Permutation Importance: importance <= 0.0005 → drop
                               stratified sample ile stabil karar

ÖNEMLİ:
    Bu sınıf encoding SONRASI kullanılmak üzere tasarlanmıştır.
    LightGBM ve Permutation adımları sadece numeric feature'lara
    uygulanır. Pipeline'da encode_categoricals() bu sınıftan önce
    çağrılmalıdır — aksi halde kategorik feature'lar Adım 5-6'da
    değerlendirilemez.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# YARDIMCI 

def _ensure_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("df is None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"DataFrame bekleniyor, got: {type(df)}")


def _save_json(data: object, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ANA SINIF 

class FeatureSelector:

    def __init__(
        self,
        target_col       : str            = 'target',
        exclude_cols     : Optional[List] = None,
        missing_thresh   : float          = 0.95,
        variance_thresh  : float          = 0.01,
        corr_thresh      : float          = 0.95,
        iv_thresh        : float          = 0.02,
        iv_bins          : int            = 10,
        iv_max_categories: int            = 50,
        lgbm_percentile  : float          = 0.25,
        lgbm_n_est       : int            = 200,
        perm_thresh      : float          = 0.0005,
        perm_n_repeats   : int            = 5,
        perm_sample_size : int            = 50_000,
    ):
        self.target_col        = target_col
        self.exclude_cols      = set(exclude_cols or [])
        self.missing_thresh    = missing_thresh
        self.variance_thresh   = variance_thresh
        self.corr_thresh       = corr_thresh
        self.iv_thresh         = iv_thresh
        self.iv_bins           = iv_bins
        self.iv_max_categories = iv_max_categories
        self.lgbm_percentile   = lgbm_percentile
        self.lgbm_n_est        = lgbm_n_est
        self.perm_thresh       = perm_thresh
        self.perm_n_repeats    = perm_n_repeats
        self.perm_sample_size  = perm_sample_size

        # fit() sonrası dolan state
        self.selected_features: List[str]        = []
        self.dropped_features : Dict[str, List]  = {
            'missing'    : [],
            'variance'   : [],
            'correlation': [],
            'iv'         : [],
            'lgbm'       : [],
            'permutation': [],
        }
        self.reports = {
            'iv'         : pd.DataFrame(),
            'lgbm'       : pd.DataFrame(),
            'permutation': pd.DataFrame(),
        }
        self._fitted = False

    # FIT 

    def fit(self, df_train: pd.DataFrame) -> 'FeatureSelector':
       
        _ensure_dataframe(df_train)

        koruma = self.exclude_cols | {self.target_col}
        adaylar = [c for c in df_train.columns if c not in koruma]

        logger.info(
            "FeatureSelector.fit() başladı — %d aday kolon",
            len(adaylar)
        )

        # ADIM 1: MISSING RATIO 
        adaylar, dropped = self._step_missing(df_train, adaylar)
        self.dropped_features['missing'] = dropped
        logger.info(
            "Adım 1 Missing     : %3d elendi → %d kaldı",
            len(dropped), len(adaylar)
        )

        # ADIM 2: VARIANCE THRESHOLD
        adaylar, dropped = self._step_variance(df_train, adaylar)
        self.dropped_features['variance'] = dropped
        logger.info(
            "Adım 2 Variance    : %3d elendi → %d kaldı",
            len(dropped), len(adaylar)
        )

        # ADIM 3: YÜKSEK KORELASYON
        adaylar, dropped = self._step_correlation(df_train, adaylar)
        self.dropped_features['correlation'] = dropped
        logger.info(
            "Adım 3 Korelasyon  : %3d elendi → %d kaldı",
            len(dropped), len(adaylar)
        )

        # ADIM 4: INFORMATION VALUE
        adaylar, dropped, iv_df = self._step_iv(df_train, adaylar)
        self.dropped_features['iv'] = dropped
        self.reports['iv']          = iv_df
        logger.info(
            "Adım 4 IV          : %3d elendi → %d kaldı",
            len(dropped), len(adaylar)
        )

        # ADIM 5: LGBM IMPORTANCE
        adaylar, dropped, lgbm_df = self._step_lgbm(
            df_train, adaylar
        )
        self.dropped_features['lgbm'] = dropped
        self.reports['lgbm']          = lgbm_df
        logger.info(
            "Adım 5 LightGBM    : %3d elendi → %d kaldı",
            len(dropped), len(adaylar)
        )

        # ADIM 6: PERMUTATION IMPORTANCE
        adaylar, dropped, perm_df = self._step_permutation(
            df_train, adaylar
        )
        self.dropped_features['permutation'] = dropped
        self.reports['permutation']          = perm_df
        logger.info(
            "Adım 6 Permutation : %3d elendi → %d kaldı",
            len(dropped), len(adaylar)
        )

        self.selected_features = adaylar
        self._fitted = True

        logger.info(
            "FeatureSelector.fit() tamamlandı — "
            "final: %d feature",
            len(self.selected_features)
        )
        return self

    # TRANSFORM 

    def transform(
        self,
        df            : pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
      
        if not self._fitted:
            raise RuntimeError(
                "fit() önce çağrılmalı."
            )
        _ensure_dataframe(df)

        kolonlar = list(self.selected_features)

        if include_target and self.target_col in df.columns:
            kolonlar = kolonlar + [self.target_col]

        # exclude_cols'u başa ekle (row_id vb.)
        for c in sorted(self.exclude_cols):
            if c in df.columns and c not in kolonlar:
                kolonlar = [c] + kolonlar

        # Eksik kolon uyarısı
        eksik = [c for c in kolonlar if c not in df.columns]
        if eksik:
            logger.warning(
                "transform — df'de eksik kolon: %s", eksik
            )
            kolonlar = [c for c in kolonlar if c in df.columns]

        return df[kolonlar].copy()

    # ARTIFACTS

    def save_artifacts(self, artifact_dir: str) -> None:
       
        if not self._fitted:
            raise RuntimeError("fit() önce çağrılmalı.")

        p = Path(artifact_dir)
        p.mkdir(parents=True, exist_ok=True)

        _save_json(
            self.selected_features,
            str(p / 'selected_features.json')
        )

        toplam_elenen = sum(
            len(v) for v in self.dropped_features.values()
        )
        _save_json(
            {
                'ozet': {
                    adim: len(liste)
                    for adim, liste
                    in self.dropped_features.items()
                },
                'toplam_elenen': toplam_elenen,
                'toplam_kalan' : len(self.selected_features),
                'detay'        : self.dropped_features
            },
            str(p / 'dropped_features.json')
        )

        self._build_report().to_csv(
            str(p / 'feature_selection_report.csv'),
            index=False
        )

        if not self.reports['iv'].empty:
            self.reports['iv'].to_csv(
                str(p / 'iv_scores.csv'), index=False
            )
        if not self.reports['lgbm'].empty:
            self.reports['lgbm'].to_csv(
                str(p / 'lgbm_importance.csv'), index=False
            )
        if not self.reports['permutation'].empty:
            self.reports['permutation'].to_csv(
                str(p / 'permutation_importance.csv'), index=False
            )

        logger.info(
            "Artifacts kaydedildi: %s (%d seçildi, %d elendi)",
            artifact_dir,
            len(self.selected_features),
            toplam_elenen
        )

    def summary(self, use_logger: bool = False) -> str:
       
        if not self._fitted:
            msg = "❌ fit() henüz çağrılmadı."
            print(msg)
            return msg

        baslangic = sum(
            len(v) for v in self.dropped_features.values()
        ) + len(self.selected_features)

        satirlar = [
            "── Feature Selection Özeti ──────────────────",
            f"   Başlangıç            : {baslangic:>4} feature",
            f"   Adım 1 Missing       : "
            f"{len(self.dropped_features['missing']):>4} elendi",
            f"   Adım 2 Variance      : "
            f"{len(self.dropped_features['variance']):>4} elendi",
            f"   Adım 3 Korelasyon    : "
            f"{len(self.dropped_features['correlation']):>4} elendi",
            f"   Adım 4 IV            : "
            f"{len(self.dropped_features['iv']):>4} elendi",
            f"   Adım 5 LightGBM      : "
            f"{len(self.dropped_features['lgbm']):>4} elendi",
            f"   Adım 6 Permutation   : "
            f"{len(self.dropped_features['permutation']):>4} elendi",
            f"   ─────────────────────────────────────────",
            f"   Final feature sayısı : "
            f"{len(self.selected_features):>4}",
        ]
        metin = "\n".join(satirlar)

        if use_logger:
            logger.info("\n%s", metin)
        else:
            print(metin)

        return metin

    # PRIVATE: ADIMLAR

    def _step_missing(
        self,
        df     : pd.DataFrame,
        adaylar: List[str]
    ) -> Tuple[List[str], List[str]]:
       
        n = len(df)
        elenecek = [
            c for c in adaylar
            if df[c].isnull().sum() / n > self.missing_thresh
        ]
        kalan = [c for c in adaylar if c not in elenecek]
        return kalan, elenecek

    def _step_variance(
        self,
        df     : pd.DataFrame,
        adaylar: List[str]
    ) -> Tuple[List[str], List[str]]:
        
        sayisal = [
            c for c in adaylar
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        elenecek = []
        for c in sayisal:
            var = df[c].var(skipna=True)
            if np.isnan(var) or var < self.variance_thresh:
                elenecek.append(c)

        kalan = [c for c in adaylar if c not in elenecek]
        return kalan, elenecek

    def _step_correlation(
        self,
        df     : pd.DataFrame,
        adaylar: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Adım 3: |r| > corr_thresh olan çiftlerden
        target korelasyonu düşük olanı ele.
        """
        sayisal = [
            c for c in adaylar
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        corr = df[sayisal].corr().abs()

        # Target korelasyonu — düşük olanı tercih et
        target_corr: Dict[str, float] = {}
        if self.target_col in df.columns:
            target_corr = (
                df[sayisal]
                .corrwith(df[self.target_col])
                .abs()
                .to_dict()
            )

        # Upper triangle mask — her çift bir kez değerlendirilir
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        # Eşiği aşan çiftleri bul
        yuksek = (
            upper.stack()[upper.stack() > self.corr_thresh]
            .index.tolist()
        )  # [(col_i, col_j), ...]

        elenecek: set = set()
        for col_i, col_j in yuksek:
            # Zaten elenmiş olanları atla
            if col_i in elenecek or col_j in elenecek:
                continue
            # Target korelasyonu düşük olanı ele
            if target_corr.get(col_i, 0) >= \
               target_corr.get(col_j, 0):
                elenecek.add(col_j)
            else:
                elenecek.add(col_i)

        elenecek_liste = list(elenecek)
        kalan = [c for c in adaylar if c not in elenecek]
        return kalan, elenecek_liste

    def _step_iv(
        self,
        df     : pd.DataFrame,
        adaylar: List[str]
    ) -> Tuple[List[str], List[str], pd.DataFrame]:
        y          = df[self.target_col]
        tot_bad    = float(y.sum())
        tot_good   = float((1 - y).sum())

        if tot_bad == 0 or tot_good == 0:
            logger.warning(
                "IV adımı: target tek sınıflı — atlandı"
            )
            return adaylar, [], pd.DataFrame()

        sonuclar = []
        for kolon in adaylar:
            try:
                iv = self._compute_iv_single(
                    df[kolon], y, tot_bad, tot_good
                )
                sonuclar.append({
                    'kolon': kolon,
                    'iv'   : round(float(iv), 4),
                    'yorum': self._iv_label(iv),
                    'tip'  : 'numeric'
                    if pd.api.types.is_numeric_dtype(df[kolon])
                    else 'categorical'
                })
            except Exception as e:
                logger.warning(
                    "IV hesap hatası — %s: %s", kolon, e
                )
                sonuclar.append({
                    'kolon': kolon,
                    'iv'   : 0.0,
                    'yorum': 'hata',
                    'tip'  : 'bilinmiyor'
                })

        iv_df = (
            pd.DataFrame(sonuclar)
            .sort_values('iv', ascending=False)
            .reset_index(drop=True)
        )

        elenecek = (
            iv_df[iv_df['iv'] < self.iv_thresh]['kolon']
            .tolist()
        )
        kalan = [c for c in adaylar if c not in elenecek]
        return kalan, elenecek, iv_df

    def _compute_iv_single(
        self,
        seri    : pd.Series,
        y       : pd.Series,
        tot_bad : float,
        tot_good: float
    ) -> float:
        gecici = pd.DataFrame(
            {'x': seri, 'y': y}
        ).dropna()

        if len(gecici) == 0:
            return 0.0

        # Binning 
        if pd.api.types.is_numeric_dtype(seri):
            try:
                gecici['bin'] = pd.qcut(
                    gecici['x'],
                    q=self.iv_bins,
                    duplicates='drop'
                )
            except Exception:
                gecici['bin'] = pd.cut(
                    gecici['x'],
                    bins=5,
                    duplicates='drop'
                )
        else:
            # Kategorik — kardinalite koruması
            gecici['x'] = gecici['x'].astype(str)
            frekans = gecici['x'].value_counts()
            nadir   = frekans[
                frekans.rank(ascending=False)
                > self.iv_max_categories
            ].index
            gecici['x'] = gecici['x'].where(
                ~gecici['x'].isin(nadir), other='OTHER'
            )
            gecici['bin'] = gecici['x']

        # WoE / IV
        ozet = (
            gecici.groupby('bin', observed=True)['y']
            .agg(['sum', 'count'])
        )
        ozet.columns = ['kotu', 'toplam']
        ozet['iyi']  = ozet['toplam'] - ozet['kotu']

        # Laplace smoothing — sıfır dist yok, perfect sep korunur
        ozet['kotu'] = ozet['kotu'] + 0.5
        ozet['iyi']  = ozet['iyi']  + 0.5

        ozet['dist_bad']  = ozet['kotu'] / tot_bad
        ozet['dist_good'] = ozet['iyi']  / tot_good

        # Finans standardı: woe = log(dist_good / dist_bad)
        ozet['woe'] = np.log(
            ozet['dist_good'] / ozet['dist_bad']
        )
        ozet['iv'] = (
            ozet['dist_good'] - ozet['dist_bad']
        ) * ozet['woe']

        return float(ozet['iv'].sum())

    @staticmethod
    def _iv_label(iv: float) -> str:
        """Finans sektörü IV yorumu."""
        if iv < 0.02: return 'zayif'
        if iv < 0.1 : return 'dusuk'
        if iv < 0.3 : return 'orta'
        if iv < 0.5 : return 'guclu'
        return 'cok_guclu'

    def _step_lgbm(
        self,
        df     : pd.DataFrame,
        adaylar: List[str]
    ) -> Tuple[List[str], List[str], pd.DataFrame]:
       
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning(
                "LightGBM kurulu değil — Adım 5 atlandı. "
                "pip install lightgbm"
            )
            return adaylar, [], pd.DataFrame()

        sayisal_adaylar = [
            c for c in adaylar
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        if not sayisal_adaylar:
            logger.warning("LGBM: sayısal aday yok — atlandı")
            return adaylar, [], pd.DataFrame()

        X = df[sayisal_adaylar]
        y = df[self.target_col]

        model = lgb.LGBMClassifier(
            n_estimators  = self.lgbm_n_est,
            learning_rate = 0.05,
            num_leaves    = 31,
            is_unbalance  = True,
            random_state  = 42,
            n_jobs        = -1,
            verbose       = -1
        )
        model.fit(X, y)

        imp_df = (
            pd.DataFrame({
                'kolon'     : sayisal_adaylar,
                'importance': model.feature_importances_
            })
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )

        # importance == 0 → kesin drop
        sifir = (
            imp_df[imp_df['importance'] == 0]['kolon']
            .tolist()
        )

        # Alt percentile → drop (sadece sıfır olmayanlardan)
        sifir_olmayan = imp_df[imp_df['importance'] > 0]
        if len(sifir_olmayan) > 0:
            esik = sifir_olmayan['importance'].quantile(
                self.lgbm_percentile
            )
            alt_perc = (
                sifir_olmayan[
                    sifir_olmayan['importance'] <= esik
                ]['kolon']
                .tolist()
            )
        else:
            alt_perc = []

        elenecek = list(set(sifir + alt_perc))
        kalan    = [c for c in adaylar if c not in elenecek]

        logger.info(
            "LGBM — sıfır: %d | alt %%%.0f: %d | "
            "n_est: %d",
            len(sifir),
            self.lgbm_percentile * 100,
            len(alt_perc),
            self.lgbm_n_est
        )
        return kalan, elenecek, imp_df

    def _step_permutation(
        self,
        df     : pd.DataFrame,
        adaylar: List[str]
    ) -> Tuple[List[str], List[str], pd.DataFrame]:
     
        from sklearn.inspection import permutation_importance

        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators  = 100,
                is_unbalance  = True,
                random_state  = 42,
                n_jobs        = -1,
                verbose       = -1
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators = 50,
                class_weight = 'balanced',
                random_state = 42,
                n_jobs       = -1
            )
            logger.warning(
                "LightGBM yok — RandomForest kullanılıyor"
            )

        sayisal_adaylar = [
            c for c in adaylar
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        if not sayisal_adaylar:
            logger.warning("Permutation: sayısal aday yok — atlandı")
            return adaylar, [], pd.DataFrame()

        # Stratified sample — class oranını koru
        sample_frac = min(
            self.perm_sample_size / len(df), 1.0
        )
        _, sample_idx = train_test_split(
            range(len(df)),
            test_size  = sample_frac,
            stratify   = df[self.target_col].values,
            random_state = 42
        )
        sample = df.iloc[sample_idx]

        X_s = sample[sayisal_adaylar]
        y_s = sample[self.target_col]

        model.fit(X_s, y_s)

        perm = permutation_importance(
            model, X_s, y_s,
            n_repeats    = self.perm_n_repeats,
            random_state = 42,
            n_jobs       = -1,
            scoring      = 'roc_auc'
        )

        perm_df = (
            pd.DataFrame({
                'kolon'          : sayisal_adaylar,
                'importance_mean': perm.importances_mean,
                'importance_std' : perm.importances_std
            })
            .sort_values('importance_mean', ascending=False)
            .reset_index(drop=True)
        )

        # perm_thresh = 0.0005 noise bound
        elenecek = (
            perm_df[
                perm_df['importance_mean'] <= self.perm_thresh
            ]['kolon']
            .tolist()
        )
        kalan = [c for c in adaylar if c not in elenecek]

        logger.info(
            "Permutation — eşik: %.4f | elenen: %d | "
            "sample: %d (stratified)",
            self.perm_thresh, len(elenecek), len(sample)
        )
        return kalan, elenecek, perm_df

    # RAPOR 

    def _build_report(self) -> pd.DataFrame:
        """
        Her feature için hangi adımda ne olduğunu gösteren
        özet rapor üretir.
        feature_selection_report.csv olarak kaydedilir.
        """
        tum_elenecek: Dict[str, str] = {}
        for adim, liste in self.dropped_features.items():
            for kolon in liste:
                tum_elenecek[kolon] = adim

        def _iv_val(kolon: str) -> Optional[float]:
            if self.reports['iv'].empty:
                return None
            sat = self.reports['iv'][
                self.reports['iv']['kolon'] == kolon
            ]
            return float(sat['iv'].values[0]) \
                if len(sat) > 0 else None

        def _lgbm_val(kolon: str) -> Optional[float]:
            if self.reports['lgbm'].empty:
                return None
            sat = self.reports['lgbm'][
                self.reports['lgbm']['kolon'] == kolon
            ]
            return float(sat['importance'].values[0]) \
                if len(sat) > 0 else None

        def _perm_val(kolon: str) -> Optional[float]:
            if self.reports['permutation'].empty:
                return None
            sat = self.reports['permutation'][
                self.reports['permutation']['kolon'] == kolon
            ]
            return float(sat['importance_mean'].values[0]) \
                if len(sat) > 0 else None

        satirlar = []

        for kolon, adim in tum_elenecek.items():
            satirlar.append({
                'kolon'               : kolon,
                'durum'               : 'elendi',
                'eleme_adimi'         : adim,
                'iv'                  : _iv_val(kolon),
                'lgbm_importance'     : _lgbm_val(kolon),
                'perm_importance_mean': _perm_val(kolon),
            })

        for kolon in self.selected_features:
            satirlar.append({
                'kolon'               : kolon,
                'durum'               : 'secildi',
                'eleme_adimi'         : None,
                'iv'                  : _iv_val(kolon),
                'lgbm_importance'     : _lgbm_val(kolon),
                'perm_importance_mean': _perm_val(kolon),
            })

        return (
            pd.DataFrame(satirlar)
            .sort_values(
                ['durum', 'eleme_adimi'],
                na_position='last'
            )
            .reset_index(drop=True)
        )