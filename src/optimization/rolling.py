"""
滚动优化：用样本内 IC 标定因子权重，逐期向前滚动。

设计哲学（防过拟合优先）
------------------------
* 不做无约束的权重搜索（那是过拟合温床）。改用"IC 加权"这一有经济含义、
  参数极少的稳健方法：在训练窗内表现稳定（ICIR 高）的因子获得更高权重。
* 权重始终非负（只做多、且我们已对每个因子调正方向），并在五大类内/间归一。
* 每 12 个月用前 36 个月数据重新标定一次，后 12 个月作为该套权重的验证期，
  逐期滚动。训练与验证严格不重叠，杜绝前视。

IC 加权公式
-----------
    w_f ∝ max(ICIR_f, 0)          # 只保留正向稳定的因子
    w_f = w_f / Σ w_f             # 归一
其中 ICIR_f = mean(IC_f) / std(IC_f)，在训练窗内计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.factor_ic import compute_ic, forward_return
from config import FACTOR_GROUPS


def ic_weighted_weights(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    horizon: int = 5,
    group_cap: float = 0.30,
) -> dict[str, float]:
    """
    在 [train_start, train_end] 训练窗内，用 ICIR 标定因子权重。

    加入"单类因子权重上限"(group_cap)，避免某一类（如动量）独大，
    保持五大类的分散——这是稳健性的来源，也限制了自由度。
    """
    fwd = forward_return(close, horizon)
    icir = {}
    for name, fac in factors.items():
        sub = fac.loc[(fac.index >= train_start) & (fac.index <= train_end)]
        ic = compute_ic(sub, fwd)
        ic = ic.dropna()
        if len(ic) < 10 or ic.std() == 0:
            icir[name] = 0.0
        else:
            icir[name] = ic.mean() / ic.std()

    # 只保留正 ICIR
    raw = {k: max(v, 0.0) for k, v in icir.items()}
    if sum(raw.values()) == 0:
        # 退化：等权
        n = len(factors)
        return {k: 1.0 / n for k in factors}

    # 先在每类内归一，再对类做上限约束，最后整体归一
    weights = {}
    for group, members in FACTOR_GROUPS.items():
        members = [m for m in members if m in raw]
        gsum = sum(raw[m] for m in members)
        for m in members:
            weights[m] = raw[m]
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # 施加单类上限
    for group, members in FACTOR_GROUPS.items():
        members = [m for m in members if m in weights]
        gw = sum(weights[m] for m in members)
        if gw > group_cap and gw > 0:
            scale = group_cap / gw
            for m in members:
                weights[m] *= scale
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def rolling_factor_weights(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    all_dates: list[pd.Timestamp],
    train_months: int = 36,
    valid_months: int = 12,
    step_months: int = 12,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    逐期滚动生成因子权重时间表。

    返回 DataFrame：index=验证期起始日, columns=因子名, 值=该期使用的权重。
    回测时按"当前日期落在哪个验证窗"取用对应权重。
    """
    dates = pd.DatetimeIndex(sorted(all_dates))
    start = dates.min()
    end = dates.max()
    rows = {}

    cursor = start + pd.DateOffset(months=train_months)
    while cursor + pd.DateOffset(months=valid_months) <= end + pd.DateOffset(days=1):
        train_start = cursor - pd.DateOffset(months=train_months)
        train_end = cursor - pd.DateOffset(days=1)
        w = ic_weighted_weights(factors, close, train_start, train_end, horizon)
        valid_start = cursor
        rows[valid_start] = w
        cursor = cursor + pd.DateOffset(months=step_months)

    return pd.DataFrame(rows).T.sort_index()


def weights_for_date(weight_schedule: pd.DataFrame,
                     date: pd.Timestamp) -> dict[str, float]:
    """取 date 所处验证窗对应的权重（取 <= date 的最近一行）。"""
    valid = weight_schedule[weight_schedule.index <= date]
    if valid.empty:
        return weight_schedule.iloc[0].to_dict()
    return valid.iloc[-1].to_dict()
