"""
组合构建：从综合得分到目标权重。

流程
----
1. 候选：综合得分排名前 top_n_buy，且通过趋势过滤（价格 > 200日均线）。
2. 选 n_holdings 只（8~15）。
3. 配权：等权 或 风险平价（低波动多配、高波动少配）。
4. 施加约束：单票 <= 10%，单行业 <= 30%；超限则截断并重新归一。

风险平价（逆波动率）权重
------------------------
    w_i ∝ 1 / σ_i ,   再归一化使 Σ w_i = 1
直觉：让每只股票对组合风险的贡献尽量均衡，避免少数高波动股主导回撤。
这是在不引入复杂协方差估计（易过拟合）前提下的稳健近似。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import PORTFOLIO


def _cap_weights(weights: pd.Series, industry: pd.Series,
                 max_stock: float, max_ind: float) -> pd.Series:
    """迭代施加单票与行业上限约束，每次截断后重新归一。"""
    w = weights.copy()
    for _ in range(50):
        changed = False
        # 单票上限
        over = w[w > max_stock]
        if not over.empty:
            excess = (over - max_stock).sum()
            w[w > max_stock] = max_stock
            free = w[w < max_stock]
            if free.sum() > 0:
                w[free.index] += excess * free / free.sum()
            changed = True
        # 行业上限
        ind_sum = w.groupby(industry.reindex(w.index)).sum()
        over_ind = ind_sum[ind_sum > max_ind]
        if not over_ind.empty:
            for ind in over_ind.index:
                members = industry.reindex(w.index)
                idx = members[members == ind].index
                scale = max_ind / w[idx].sum()
                w[idx] *= scale
            w = w / w.sum()
            changed = True
        if not changed:
            break
    return w / w.sum()


def build_target_weights(
    score_row: pd.Series,
    trend_pass: pd.Series,
    vol_row: pd.Series,
    industry: pd.Series,
    weighting: str | None = None,
) -> pd.Series:
    """
    给定某一调仓日的横截面，计算目标权重（index=code, sum=1）。

    score_row  : 综合得分（index=code）
    trend_pass : 趋势过滤布尔（index=code）
    vol_row    : 20日波动率（风险平价用，index=code）
    industry   : 行业标签（index=code）
    """
    weighting = weighting or PORTFOLIO.weighting
    cand = score_row.dropna()
    # 趋势过滤
    cand = cand[trend_pass.reindex(cand.index).fillna(False)]
    if cand.empty:
        return pd.Series(dtype=float)

    # 取得分前 top_n_buy
    cand = cand.sort_values(ascending=False).head(PORTFOLIO.top_n_buy)
    # 实际持仓数量：在 [min, max] 内，取候选数与上限的较小者
    n = min(len(cand), PORTFOLIO.n_holdings_max)
    n = max(n, min(PORTFOLIO.n_holdings_min, len(cand)))
    selected = cand.head(n)

    if weighting == "equal":
        w = pd.Series(1.0 / len(selected), index=selected.index)
    else:  # risk_parity 逆波动率
        vol = vol_row.reindex(selected.index).replace(0, np.nan)
        vol = vol.fillna(vol.median() if vol.notna().any() else 1.0)
        inv = 1.0 / vol
        w = inv / inv.sum()

    w = _cap_weights(w, industry,
                     PORTFOLIO.max_weight_per_stock,
                     PORTFOLIO.max_weight_per_industry)
    return w
