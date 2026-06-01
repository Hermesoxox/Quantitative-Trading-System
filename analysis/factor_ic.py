"""
因子有效性评估：IC / RankIC / ICIR 与分组（quantile）回测。

IC (Information Coefficient)
----------------------------
定义为 t 期因子值与 t->t+h 期收益的横截面相关系数：
    IC_t = corr( factor_{·,t},  ret_{·, t->t+h} )
使用 Spearman 秩相关称为 RankIC，对异常值更稳健，是 A股最常用口径。

衍生指标：
    IC_mean : IC 序列均值，衡量因子预测方向与强度
    ICIR    : IC_mean / IC_std，衡量因子稳定性（类似信息比率）
    IC>0占比: IC 为正的期数占比，衡量方向一致性

分组回测：每期按因子值排序分成 Q 组，看各组未来收益是否单调，
最高组减最低组（多空）的累计收益是因子选股能力的直观体现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """未来 horizon 日收益（用于与当期因子对齐计算 IC）。"""
    return close.shift(-horizon) / close - 1


def compute_ic(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """逐日计算横截面 IC，返回 IC 时间序列。"""
    common_cols = factor.columns.intersection(fwd_ret.columns)
    f = factor[common_cols]
    r = fwd_ret[common_cols]
    ic = {}
    for date in f.index:
        if date not in r.index:
            continue
        x = f.loc[date]
        y = r.loc[date]
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 10:
            continue
        ic[date] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return pd.Series(ic).sort_index()


def ic_summary(ic: pd.Series) -> dict:
    """IC 序列的汇总统计。"""
    ic = ic.dropna()
    if ic.empty:
        return {}
    mean, std = ic.mean(), ic.std()
    return {
        "IC_mean": round(mean, 4),
        "IC_std": round(std, 4),
        "ICIR": round(mean / std, 4) if std > 0 else np.nan,
        "IC>0_pct": round((ic > 0).mean(), 4),
        "t_stat": round(mean / std * np.sqrt(len(ic)), 2) if std > 0 else np.nan,
        "n_periods": len(ic),
    }


def quantile_backtest(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    分组回测：每期按因子分 n_groups 组，返回各组平均未来收益（按期）。

    返回 DataFrame：index=date, columns=[Q1..Qn, LongShort]
    其中 Q_n 为因子值最高组，LongShort = Q_n - Q1。
    """
    rows = []
    for date in factor.index:
        if date not in fwd_ret.index:
            continue
        x = factor.loc[date].dropna()
        y = fwd_ret.loc[date].reindex(x.index)
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < n_groups * 3:
            continue
        labels = pd.qcut(pair.iloc[:, 0], n_groups, labels=False,
                         duplicates="drop")
        grp = pair.iloc[:, 1].groupby(labels).mean()
        rows.append(pd.Series(grp.values,
                              index=[f"Q{int(i)+1}" for i in grp.index],
                              name=date))
    out = pd.DataFrame(rows)
    if {"Q1", f"Q{n_groups}"}.issubset(out.columns):
        out["LongShort"] = out[f"Q{n_groups}"] - out["Q1"]
    return out


def evaluate_factor_library(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizon: int = 5,
    n_groups: int = 5,
) -> pd.DataFrame:
    """对全部因子批量计算 IC 汇总，返回排序后的评估表。"""
    fwd = forward_return(close, horizon)
    rows = {}
    for name, fac in factors.items():
        ic = compute_ic(fac, fwd)
        summ = ic_summary(ic)
        if summ:
            rows[name] = summ
    table = pd.DataFrame(rows).T
    if "ICIR" in table.columns:
        table = table.sort_values("ICIR", ascending=False)
    return table
