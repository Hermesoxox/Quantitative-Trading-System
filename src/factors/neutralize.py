"""
因子预处理：去极值(winsorize) -> 标准化(z-score) -> 行业与市值中性化。

为什么要中性化
--------------
原始因子常与"行业"和"市值"高度相关。例如低估值因子会系统性偏向银行/地产，
小市值因子会让组合一直押注小盘股。若不剥离这些暴露，因子的 alpha 会被
beta（行业轮动、市值风格）污染，回测好看但实盘风格漂移、回撤失控。

做法：在每个横截面（同一交易日）上，把因子值对 [行业哑变量 + ln(市值)] 做
OLS 回归，取残差作为"纯净"因子。残差与行业、市值正交。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """
    基于中位数绝对偏差(MAD)的去极值：比 ±3σ 更稳健，不受极端值拉动。
    上下界 = median ± n_mad * 1.4826 * MAD。
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    upper = med + n_mad * 1.4826 * mad
    lower = med - n_mad * 1.4826 * mad
    return s.clip(lower, upper)


def standardize(s: pd.Series) -> pd.Series:
    """横截面 z-score 标准化：(x - mean) / std。"""
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def neutralize_cross_section(
    factor: pd.Series,
    industry: pd.Series,
    log_cap: pd.Series,
) -> pd.Series:
    """
    单个横截面的行业+市值中性化。

    factor   : index=code 的因子值（已去极值、标准化）
    industry : index=code 的行业标签
    log_cap  : index=code 的对数市值
    返回残差（index=code），与行业哑变量、市值正交。
    """
    df = pd.concat([factor.rename("y"), industry.rename("ind"),
                    log_cap.rename("cap")], axis=1).dropna()
    if len(df) < 10:
        return factor  # 样本太少不做回归

    # 构造设计矩阵：行业哑变量（去掉一列防共线）+ 市值 + 截距
    dummies = pd.get_dummies(df["ind"], drop_first=True).astype(float)
    X = pd.concat([dummies, df["cap"]], axis=1)
    X.insert(0, "const", 1.0)
    X_mat = X.values
    y = df["y"].values

    # 最小二乘解：beta = (X'X)^-1 X'y，用 lstsq 更数值稳定
    beta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
    resid = y - X_mat @ beta
    out = pd.Series(resid, index=df.index)
    return out.reindex(factor.index)


def neutralize_factor(
    factor_wide: pd.DataFrame,
    industry_map: pd.Series,
    log_cap_map: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """
    对整张因子宽表逐日做：去极值 -> 标准化 -> 行业市值中性化 -> 再标准化。

    factor_wide : index=date, columns=code
    industry_map: index=code -> industry（静态近似；生产中应按日更新）
    log_cap_map : index=code -> log_cap（静态）或 index=date,col=code 的宽表
    返回中性化后的宽表。
    """
    out = pd.DataFrame(index=factor_wide.index, columns=factor_wide.columns,
                       dtype=float)
    cap_is_panel = isinstance(log_cap_map, pd.DataFrame)

    for date, row in factor_wide.iterrows():
        s = row.dropna()
        if len(s) < 10:
            continue
        s = standardize(winsorize(s))
        ind = industry_map.reindex(s.index)
        if cap_is_panel:
            cap = log_cap_map.loc[date].reindex(s.index) \
                if date in log_cap_map.index else pd.Series(index=s.index)
        else:
            cap = log_cap_map.reindex(s.index)
        resid = neutralize_cross_section(s, ind, cap)
        out.loc[date, resid.index] = standardize(resid).values
    return out
