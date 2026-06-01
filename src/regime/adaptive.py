"""
自适应风险叠加层：让"目标波动"与"回撤守卫带宽"随市场状态自我调节，
而不是写死一个数字。

为什么要自适应(而非固定参数)
----------------------------
固定的 target_vol=10% / dd_hard=15% 在不同市场环境下并不等价：牛市低波时
10% 目标会过度降仓踏空，熊市高波时同样的目标又限不住回撤。机构做法是让
风险预算"跟着市场的波动状态走"，且**用规则而非拟合**来调节——规则可解释、
无自由参数、不会过拟合。

两个自适应件：
1) adaptive_target_vol：目标波动 = 基准近一段实现波动的滚动分位数(中位数)，
   即"以市场自身常态波动为锚"。市场整体进入高波动期，目标自然抬升(不至于
   一直空仓)；进入低波期，目标下降(不被噪声放大仓位)。仅 1 个 lookback 参数。
2) vol_scaled_dd_bands：回撤守卫的软/硬带宽随当前波动缩放——高波时适当放宽
   (避免被日常抖动频繁打飞)，低波时收紧。规则式，无拟合。

注意：这些是"规则化自适应"，不是"参数寻优"。真正需要在样本内标定的少量选择
(如 target 相对锚的倍数)，应走 examples/run_walkforward.py 的净化滚动流程。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detector import build_market_proxy, market_regime_exposure


def adaptive_target_vol(
    market: pd.Series,
    lookback: int = 252,
    realized_window: int = 20,
    floor: float = 0.06,
    cap: float = 0.20,
) -> pd.Series:
    """
    自适应目标波动：以基准近 lookback 日实现波动的滚动中位数为锚，clip 到 [floor,cap]。
    返回逐日 target_vol 序列。
    """
    realized = market.pct_change().rolling(realized_window).std() * np.sqrt(252)
    anchor = realized.rolling(lookback, min_periods=realized_window).median()
    return anchor.clip(floor, cap).bfill()


def adaptive_combined_exposure(
    close: pd.DataFrame,
    benchmark: pd.Series | None = None,
    ma_window: int = 200,
    vol_lookback: int = 252,
    vol_window: int = 20,
    smooth: int = 5,
) -> pd.Series:
    """
    自适应版总仓位：趋势过滤 × (自适应目标波动 / 实现波动)，再平滑。
    与 detector.combined_exposure 接口一致，但目标波动随市场自我调节。
    """
    market = build_market_proxy(close, benchmark)
    trend = market_regime_exposure(market, ma_window)
    realized = market.pct_change().rolling(vol_window).std() * np.sqrt(252)
    tgt = adaptive_target_vol(market, vol_lookback, vol_window)
    scalar = (tgt / realized.replace(0, np.nan)).clip(0, 1)
    expo = (trend * scalar).clip(0, 1).fillna(0.0)
    if smooth and smooth > 1:
        expo = expo.ewm(span=smooth, min_periods=1).mean()
    return expo.reindex(close.index).fillna(0.0)


def vol_scaled_dd_bands(
    market: pd.Series,
    base_soft: float = 0.10,
    base_hard: float = 0.15,
    vol_window: int = 20,
    ref_vol: float = 0.20,
) -> pd.DataFrame:
    """
    波动缩放的回撤守卫带宽：band = base × clip(realized_vol / ref_vol, 0.7, 1.5)。
    返回逐日 [soft, hard] 两列。高波放宽、低波收紧；规则式，无拟合参数。
    """
    realized = market.pct_change().rolling(vol_window).std() * np.sqrt(252)
    scale = (realized / ref_vol).clip(0.7, 1.5).fillna(1.0)
    return pd.DataFrame({"soft": (base_soft * scale).clip(0.05, 0.15),
                         "hard": (base_hard * scale).clip(0.08, 0.22)},
                        index=market.index)
