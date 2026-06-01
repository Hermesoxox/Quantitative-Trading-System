"""
股票池过滤：剔除 ST/*ST、上市不满 60 天、流动性不足的标的。

过滤是"点位敏感"的——必须用 t 时刻可知的信息构造 t 时刻可交易的池子，
否则会引入幸存者偏差与未来函数。本模块在每个调仓日动态重算可交易池。
"""

from __future__ import annotations

import pandas as pd

from config import UNIVERSE


def is_star_or_gem(code: str) -> bool:
    """创业板(300/301) 或 科创板(688) -> 涨跌停 20%。"""
    return code.startswith(("300", "301", "688"))


def build_tradable_universe(
    price: pd.DataFrame,
    as_of: pd.Timestamp,
    st_codes: set | None = None,
    listing_date: pd.Series | None = None,
) -> list[str]:
    """
    返回在 as_of 日可纳入选股的股票代码列表。

    过滤规则（全部使用 as_of 当日及之前的数据）：
      1. 非 ST / *ST
      2. 上市满 min_listing_days
      3. 最近 amount_window 日的日均成交额 >= min_avg_amount
    """
    st_codes = st_codes or set()
    # 截取 as_of 之前的窗口
    hist = price[price.index.get_level_values("date") <= as_of]
    window_start = as_of - pd.Timedelta(days=UNIVERSE.amount_window * 2)
    recent = hist[hist.index.get_level_values("date") >= window_start]

    avg_amount = recent.groupby("code")["amount"].mean()

    out = []
    for code in avg_amount.index:
        if UNIVERSE.exclude_st and code in st_codes:
            continue
        if avg_amount[code] < UNIVERSE.min_avg_amount:
            continue
        if listing_date is not None and code in listing_date.index:
            days = (as_of - listing_date[code]).days
            if days < UNIVERSE.min_listing_days:
                continue
        # 必须有足够历史用于计算长周期因子（200日均线等）
        n_obs = (hist.index.get_level_values("code") == code).sum()
        if n_obs < 60:
            continue
        out.append(code)
    return out
