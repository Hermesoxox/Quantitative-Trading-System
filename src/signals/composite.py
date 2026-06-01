"""
信号合成：把多个中性化后的因子，按方向与权重线性合成为综合得分。

合成公式
--------
对每个因子 f：
    adj_f = direction_f * neutralized_f          # 调正方向，使"越大越好"
综合得分：
    Score_{i,t} = sum_f ( w_f * adj_f_{i,t} )     # 加权求和
            （sum_f w_f = 1）

为什么用线性加权而不是机器学习
------------------------------
* 线性合成参数少、可解释、不易过拟合，每个权重都有经济含义。
* 因子已各自标准化，量纲一致，可直接加权。
* 滚动优化阶段会用样本内 IC 重新标定 w_f（见 optimization/rolling.py），
  但始终约束在五大类的先验框架内，避免数据挖掘。
"""

from __future__ import annotations

import pandas as pd

from config import FACTOR_DIRECTION, PORTFOLIO


def apply_direction(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """按 config.FACTOR_DIRECTION 调正每个因子方向（统一为越大越好）。"""
    return {name: FACTOR_DIRECTION.get(name, 1) * df
            for name, df in factors.items()}


def composite_score(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    线性加权合成综合得分。factors 应为"已中性化"的宽表字典。

    缺失因子按 0 贡献处理；权重在可用因子上重新归一，避免某日某因子
    大面积缺失时整体得分被压低。
    """
    adj = apply_direction(factors)
    # 对齐到同一 index/columns
    sample = next(iter(adj.values()))
    score = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)
    weight_sum = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)

    for name, w in weights.items():
        if name not in adj:
            continue
        f = adj[name].reindex_like(score)
        mask = f.notna()
        score = score.add((f.fillna(0) * w), fill_value=0)
        weight_sum = weight_sum.add(mask.astype(float) * w, fill_value=0)

    return score.where(weight_sum > 0).div(weight_sum.replace(0, pd.NA))


def smooth_score(score: pd.DataFrame, span: int | None = None) -> pd.DataFrame:
    """
    对综合得分做时间维 EMA 平滑，降低信号抖动与换手率。

    公式:  Smoothed_{i,t} = EMA(Score_{i,·}, span)_t
    直觉:  原始横截面得分逐日跳动，会触发频繁的进出与权重微调，吃掉成本。
           EMA 让得分更"黏"，只有持续性的相对强弱变化才改变持仓，
           契合 5-20 日持仓周期，且不引入未来信息（仅用历史）。span=0 关闭。
    """
    s = PORTFOLIO.score_smooth_span if span is None else span
    if not s or s <= 1:
        return score
    return score.ewm(span=s, min_periods=1).mean()


def trend_filter(close: pd.DataFrame, ma_window: int | None = None) -> pd.DataFrame:
    """
    趋势过滤：价格在长期均线之上才允许买入，返回布尔宽表。

    公式:  Pass_{i,t} = Close_{i,t} > SMA(Close, ma_window)_{i,t}
    直觉:  只在中长期趋势向上的标的中选股，过滤掉处于下降通道的"价值陷阱"，
           显著降低回撤。这是把"选股 alpha"与"择时 beta"解耦的关键一步。
    """
    w = ma_window or PORTFOLIO.trend_ma_long
    ma = close.rolling(w).mean()
    return close > ma
