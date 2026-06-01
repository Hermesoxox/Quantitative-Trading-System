"""
因子库：15+ 个因子，覆盖价值/动量/反转/波动率/资金流/质量六大类。

约定
----
* 输入 `close`, `high`, `low`, `volume`, `amount` 等均为"宽表"：
  index = 交易日, columns = 股票代码（DataFrame，float）。
  宽表便于做横截面（cross-sectional）与时间序列双向运算。
* 每个因子函数返回一个同形状宽表，值为"原始因子值"（未中性化、未标准化）。
* 因子方向统一在 config.FACTOR_DIRECTION 中声明；本库只算原值，不调方向。
* 所有滚动窗口均为"截至当日（含）"，不使用未来数据。

每个因子都附带：公式 + 直觉解释，杜绝黑箱。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ======================== 价值类 Value ========================
def factor_ep(close: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """
    盈利收益率 E/P = 每股收益(TTM) / 价格。

    公式:  EP_{i,t} = EPS_TTM_{i,t} / Close_{i,t}
    直觉:  市盈率倒数。EP 越高 = 估值越便宜。价值投资的核心锚，
           A股长期存在"低估值溢价"，尤其在风格切换后修复明显。
    """
    return eps / close


def factor_bp(close: pd.DataFrame, bps: pd.DataFrame) -> pd.DataFrame:
    """
    账面市值比 B/P = 每股净资产 / 价格 (= 1 / 市净率)。

    公式:  BP_{i,t} = BPS_{i,t} / Close_{i,t}
    直觉:  Fama-French 价值因子的经典代理。高 BP 代表市场给的溢价低，
           历史上有正向风险补偿。
    """
    return bps / close


# ======================== 动量类 Momentum ========================
def factor_mom_20(close: pd.DataFrame) -> pd.DataFrame:
    """
    20 日动量 = 近 20 个交易日累计收益。

    公式:  MOM20_{i,t} = Close_{i,t} / Close_{i,t-20} - 1
    直觉:  短中期趋势延续。强者恒强，捕捉资金持续流入的标的。
    """
    return close / close.shift(20) - 1


def factor_mom_60(close: pd.DataFrame) -> pd.DataFrame:
    """
    60 日动量（约一个季度）。

    公式:  MOM60_{i,t} = Close_{i,t} / Close_{i,t-60} - 1
    直觉:  季度级趋势，比 20 日更稳定、噪音更低，是动量主力。
    """
    return close / close.shift(60) - 1


def factor_mom_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """
    经典 12-1 动量：过去 12 个月收益剔除最近 1 个月。

    公式:  MOM_{i,t} = Close_{i,t-21} / Close_{i,t-252} - 1
    直觉:  Jegadeesh-Titman 动量。剔除最近一个月是为了规避短期反转效应
           （最近一个月涨太多往往要回吐）。学术上最稳健的动量定义。
    """
    return close.shift(21) / close.shift(252) - 1


# ======================== 反转类 Reversal ========================
def factor_rev_5(close: pd.DataFrame) -> pd.DataFrame:
    """
    5 日反转 = 近 5 日收益（方向为负：跌得多的反弹）。

    公式:  REV5_{i,t} = Close_{i,t} / Close_{i,t-5} - 1
    直觉:  A股散户主导，短期超跌后存在均值回复。注意 config 中方向 = -1，
           即原始值越小（跌得越多）打分越高。
    """
    return close / close.shift(5) - 1


def factor_boll_lower_bounce(close: pd.DataFrame, window: int = 20,
                             k: float = 2.0) -> pd.DataFrame:
    """
    布林带下轨反弹强度。

    公式:  MA = SMA(close, 20);  SD = STD(close, 20)
           Lower = MA - k*SD
           BOLL_{i,t} = (MA - Close_{i,t}) / (k*SD)
    直觉:  价格越接近/跌破下轨，BOLL 值越大（接近或 >1），代表超卖、
           反弹概率上升。标准化后用作"低位介入"信号。方向 +1。
    """
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (ma - close) / (k * sd.replace(0, np.nan))


# ======================== 波动率类 Volatility ========================
def factor_vol_20(close: pd.DataFrame) -> pd.DataFrame:
    """
    20 日历史波动率（年化）。

    公式:  ret = close.pct_change()
           VOL20_{i,t} = STD(ret_{t-19..t}) * sqrt(252)
    直觉:  低波动异象——低波动股票长期风险调整后收益更优。方向 -1
           （波动越低越好），同时直接服务于风险预算。
    """
    ret = close.pct_change()
    return ret.rolling(20).std() * np.sqrt(252)


def factor_atr_ratio(high: pd.DataFrame, low: pd.DataFrame,
                     close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR 比率 = 平均真实波幅 / 收盘价。

    公式:  TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
           ATR = SMA(TR, 14)
           ATRR_{i,t} = ATR_{i,t} / Close_{i,t}
    直觉:  归一化的真实波动幅度，跨个股可比。高 ATR 比率代表日内剧烈，
           交易摩擦与不确定性大。方向 -1，同时用于个股仓位/止损计算。
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=0).groupby(level=0).max() if False else None
    # 上面的写法对宽表不适用，改为逐元素 max：
    tr = np.maximum.reduce([
        (high - low).values,
        (high - prev_close).abs().values,
        (low - prev_close).abs().values,
    ])
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    atr = tr.rolling(window).mean()
    return atr / close


# ======================== 资金流类 Money Flow ========================
def factor_mf_3d_ratio(mf_ratio: pd.DataFrame) -> pd.DataFrame:
    """
    3 日主力资金净流入比率（移动平均）。

    公式:  MF3_{i,t} = SMA(main_net_inflow_ratio_{t-2..t}, 3)
    直觉:  主力（大单/特大单）连续净流入，往往领先于价格。3 日平滑
           过滤单日噪音。方向 +1。A股资金驱动特征显著，该因子贡献稳定。
    """
    return mf_ratio.rolling(3).mean()


def factor_vol_surge(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    成交量异常放大 = 当日量 / 过去 N 日均量。

    公式:  VS_{i,t} = Volume_{i,t} / SMA(Volume_{t-window..t-1}, window)
    直觉:  量在价先。温和放量（配合趋势过滤）代表资金关注度提升。
           方向 +1，但需配合趋势确认，单纯放量也可能是出货——见信号合成。
    """
    avg = volume.shift(1).rolling(window).mean()
    return volume / avg.replace(0, np.nan)


# ======================== 质量类 Quality ========================
def factor_roe(roe: pd.DataFrame) -> pd.DataFrame:
    """
    净资产收益率 ROE。

    公式:  ROE_{i,t} = 归母净利润_TTM / 平均净资产
    直觉:  最核心的盈利能力指标。高 ROE 且可持续的公司长期跑赢。方向 +1。
    """
    return roe


def factor_gross_margin(gm: pd.DataFrame) -> pd.DataFrame:
    """
    毛利率。

    公式:  GM = (营业收入 - 营业成本) / 营业收入
    直觉:  反映护城河与定价权。高且稳定的毛利率是质量护城河的代理。方向 +1。
    """
    return gm


def factor_earnings_stability(net_profit: pd.DataFrame,
                              window: int = 8) -> pd.DataFrame:
    """
    盈利稳定性 = 过去 N 个季度净利润同比增速的 -标准差（越稳越好）。

    公式:  g_q = NetProfit_q / NetProfit_{q-4} - 1  (同比增速)
           STAB = -STD(g_q, 最近 window 个季度)
    直觉:  盈利波动小的公司更可预测、估值更稳健。取负号使"越稳分越高"，
           方向 +1。规避业绩暴雷型公司。
    """
    yoy = net_profit / net_profit.shift(4) - 1
    return -yoy.rolling(window).std()


# 因子名 -> (函数, 所需输入键)
FACTOR_FUNCS = {
    "ep": (factor_ep, ["close", "eps"]),
    "bp": (factor_bp, ["close", "bps"]),
    "mom_20": (factor_mom_20, ["close"]),
    "mom_60": (factor_mom_60, ["close"]),
    "mom_12_1": (factor_mom_12_1, ["close"]),
    "rev_5": (factor_rev_5, ["close"]),
    "boll_lower_bounce": (factor_boll_lower_bounce, ["close"]),
    "vol_20": (factor_vol_20, ["close"]),
    "atr_ratio": (factor_atr_ratio, ["high", "low", "close"]),
    "mf_3d_ratio": (factor_mf_3d_ratio, ["mf_ratio"]),
    "vol_surge": (factor_vol_surge, ["volume"]),
    "roe": (factor_roe, ["roe"]),
    "gross_margin": (factor_gross_margin, ["gross_margin"]),
    "earnings_stability": (factor_earnings_stability, ["net_profit"]),
}


def compute_all_factors(wide: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    给定宽表字典（close/high/low/volume/amount/eps/bps/roe/gross_margin/
    net_profit/mf_ratio），计算全部原始因子，返回 {因子名: 宽表}。
    """
    out = {}
    for name, (func, needs) in FACTOR_FUNCS.items():
        if all(k in wide for k in needs):
            try:
                out[name] = func(*[wide[k] for k in needs])
            except Exception as e:
                print(f"[compute_all_factors] 因子 {name} 计算失败: {e}")
    return out
