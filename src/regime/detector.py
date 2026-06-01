"""
市场状态识别与动态仓位(对标 SOTA 的 regime / vol-targeting 叠加)。

对集中持仓(≤5只)而言，这一层比选股更重要
------------------------------------------
持 5 只股票时单票风险极大，纯靠选股无法守住 20% 回撤目标。机构做法是在选股
之上叠加两层"总仓位"控制，把"何时该满仓/减仓/空仓"与"买哪几只"解耦：

1) 市场状态过滤 (Trend Regime)
   用市场基准(沪深300或等权全市场)是否在 200 日均线上方判定牛熊：
       Exposure_trend = 1.0  若 Index > MA200 (上行)
                      = 0.0  若 Index < MA200 (下行，空仓避险)
   直觉：A股熊市是系统性下跌，集中持仓在熊市会被团灭；趋势在均线下方时直接
   退出权益、持币观望，是控制最大回撤最有效、最稳健的单一手段。

2) 波动率目标 (Volatility Targeting)
   把组合年化波动缩放到目标水平(如 12%)：
       Scalar_vol = clip( target_vol / realized_vol , 0, 1 )
   实现波动率用基准近 20 日收益年化估计。市场越动荡，自动降杠杆(降仓位)，
   反之亦然——使风险预算在时间上更均衡，显著平滑净值、压低尾部回撤。

最终总仓位 = Exposure_trend × Scalar_vol，喂给回测引擎缩放可投资金额。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_proxy(close: pd.DataFrame,
                       benchmark: pd.Series | None = None) -> pd.Series:
    """
    市场基准序列：优先用传入基准(如沪深300)，否则用全样本等权指数代理。
    等权指数 = 各股日收益的横截面均值累乘。
    """
    if benchmark is not None and len(benchmark) > 1:
        return benchmark.reindex(close.index).ffill()
    eq_ret = close.pct_change().mean(axis=1)
    return (1 + eq_ret.fillna(0)).cumprod()


def market_regime_exposure(market: pd.Series, ma_window: int = 200) -> pd.Series:
    """趋势过滤：基准在 MA200 上方=1(可持仓)，下方=0(空仓)。"""
    ma = market.rolling(ma_window).mean()
    return (market > ma).astype(float)


def vol_target_scalar(market: pd.Series, target_vol: float = 0.12,
                      window: int = 20, cap: float = 1.0) -> pd.Series:
    """波动率目标缩放系数 ∈ [0, cap]。"""
    realized = market.pct_change().rolling(window).std() * np.sqrt(252)
    scalar = (target_vol / realized.replace(0, np.nan)).clip(0, cap)
    return scalar.fillna(0.0)


def combined_exposure(
    close: pd.DataFrame,
    benchmark: pd.Series | None = None,
    ma_window: int = 200,
    target_vol: float = 0.12,
    vol_window: int = 20,
    smooth: int = 5,
) -> pd.Series:
    """
    合成总仓位时间表(date -> [0,1])：趋势过滤 × 波动率目标，再做小幅平滑
    以避免在临界点频繁满仓/空仓切换造成的换手。
    """
    market = build_market_proxy(close, benchmark)
    trend = market_regime_exposure(market, ma_window)
    volsc = vol_target_scalar(market, target_vol, vol_window)
    expo = (trend * volsc).clip(0, 1)
    if smooth and smooth > 1:
        expo = expo.ewm(span=smooth, min_periods=1).mean()
    return expo.reindex(close.index).fillna(0.0)
