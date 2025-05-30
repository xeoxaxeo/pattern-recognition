# utils/preprocessing.py
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   FunctionTransformer)
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------- VIF / correlation helpers ---------------------------------------
def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    df_num = df_num.fillna(df_num.mean()).loc[:, df_num.std(ddof=0) > 0]
    if df_num.empty:
        return pd.DataFrame(columns=["feature","VIF"])
    vif = pd.DataFrame()
    vif["feature"] = df_num.columns
    vif["VIF"] = [variance_inflation_factor(df_num.values, i)
                  for i in range(df_num.shape[1])]
    return vif

def remove_corr_vif(df: pd.DataFrame, cols: List[str],
                    corr_thr: float = 0.95, vif_thr: float = 10) -> List[str]:
    removed = []
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns
            for r in upper.index if upper.loc[r, c] > corr_thr}
    df_f = df.drop(columns=list(drop)); removed.extend(drop)
    X = df_f[[c for c in cols if c in df_f.columns]].copy()
    while True:
        vif_df = calculate_vif(X)
        if vif_df.empty: break
        worst = vif_df.sort_values("VIF", ascending=False).iloc[0]
        if worst["VIF"] <= vif_thr: break
        X.drop(columns=[worst["feature"]], inplace=True)
        removed.append(worst["feature"])
    return removed

# ---------- preprocessor builder --------------------------------------------
@dataclass
class FeatureGroups:
    log1p:   List[str]
    signed:  List[str]
    standard:List[str]
    minmax:  List[str]
    cat:     List[str]

def build_preprocessor(groups: FeatureGroups) -> Tuple[ColumnTransformer, List[int]]:
    """Return ColumnTransformer and cat_feature_indices."""
    log1p_pipe = SKPipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None))))
    ])
    signed_pipe = SKPipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("slog", FunctionTransformer(lambda x: np.sign(x)*np.log1p(np.abs(x))))
    ])
    std_pipe = SKPipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("std", StandardScaler())
    ])
    mm_pipe = SKPipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("mm", MinMaxScaler())
    ])
    cat_pipe = SKPipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("str", FunctionTransformer(lambda x: x.astype(str)))
    ])

    ct = ColumnTransformer([
        ("log1p",  log1p_pipe,   groups.log1p),
        ("signed", signed_pipe,  groups.signed),
        ("std",    std_pipe,     groups.standard),
        ("mm",     mm_pipe,      groups.minmax),
        ("cat",    cat_pipe,     groups.cat)
    ], remainder="passthrough")

    pre_len = len(groups.log1p)+len(groups.signed)+len(groups.standard)+len(groups.minmax)
    cat_idx = list(range(pre_len, pre_len+len(groups.cat)))
    return ct, cat_idx
