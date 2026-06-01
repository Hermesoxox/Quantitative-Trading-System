"""
数据加载层。

设计原则：
1. 业务代码只依赖标准化后的 DataFrame，不直接依赖任何数据源 API。
   这样把 AkShare 换成 Tushare / Wind / 聚宽 时，只需改本文件。
2. 所有行情统一为"前复权"价格用于计算因子与信号；回测撮合时另用
   不复权价 + 复权因子还原真实成交价，避免复权造成的未来函数。
3. 提供本地缓存（parquet），避免重复拉取、便于离线复现。

标准化后的数据结构（长表，MultiIndex = [date, code]）：
    price : open, high, low, close, volume, amount, adj_factor, pre_close
    fund  : report_date, roe, gross_margin, net_profit, total_equity, bps, eps ...
    flow  : main_net_inflow (主力净流入额), main_net_inflow_ratio ...
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# AkShare 适配器（演示用）。生产环境建议换成机构级数据源。
# ----------------------------------------------------------------------------
def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def load_daily_price(
    codes: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    拉取日线行情（前复权）。返回 MultiIndex(date, code) 的长表。

    列：open, high, low, close, volume(手), amount(元), pct_chg, pre_close
    """
    cache = _cache_path(f"price_{start}_{end}")
    if use_cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df[df.index.get_level_values("code").isin(codes)]

    import akshare as ak  # 延迟导入，无网络环境下不影响其余模块

    frames = []
    for code in codes:
        try:
            # adjust="qfq" 前复权；接口返回中文列名，需重命名
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            if raw is None or raw.empty:
                continue
            raw = raw.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "pct_chg",
            })
            raw["date"] = pd.to_datetime(raw["date"])
            raw["code"] = code
            raw["pre_close"] = raw["close"].shift(1)
            frames.append(raw[["date", "code", "open", "high", "low",
                               "close", "volume", "amount", "pct_chg",
                               "pre_close"]])
        except Exception as e:  # 单只股票失败不应中断全局
            print(f"[load_daily_price] {code} 拉取失败: {e}")

    if not frames:
        raise RuntimeError("未能获取任何行情数据，请检查网络或数据源。")

    df = pd.concat(frames).set_index(["date", "code"]).sort_index()
    if use_cache:
        df.to_parquet(cache)
    return df


def load_fundamentals(codes: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    拉取财务季度数据，返回 MultiIndex(report_date, code)。

    关键列：roe(净资产收益率), gross_margin(毛利率),
            net_profit(归母净利润), total_equity(净资产),
            bps(每股净资产), eps(每股收益)
    注意：财务数据必须按"公告日"而非"报告期"对齐，否则引入未来函数。
    本演示用 report_date + 一个保守的披露滞后（90 天）近似公告日。
    """
    cache = _cache_path("fundamentals")
    if use_cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df[df.index.get_level_values("code").isin(codes)]

    import akshare as ak
    frames = []
    for code in codes:
        try:
            raw = ak.stock_financial_abstract(symbol=code)
            # 接口字段随版本变化，此处仅示意标准化流程
            raw["code"] = code
            frames.append(raw)
        except Exception as e:
            print(f"[load_fundamentals] {code} 拉取失败: {e}")
    if not frames:
        raise RuntimeError("未能获取财务数据。")
    df = pd.concat(frames)
    # 真实落地需在此处做字段映射与公告日对齐
    if use_cache:
        df.to_parquet(cache)
    return df


def load_money_flow(codes: List[str], start: str, end: str,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    拉取个股资金流向（主力净流入），返回 MultiIndex(date, code)。

    关键列：main_net_inflow(主力净流入额，元),
            main_net_inflow_ratio(主力净流入占成交额比)
    """
    cache = _cache_path(f"flow_{start}_{end}")
    if use_cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df[df.index.get_level_values("code").isin(codes)]

    import akshare as ak
    frames = []
    for code in codes:
        try:
            market = "sh" if code.startswith(("6", "9")) else "sz"
            raw = ak.stock_individual_fund_flow(stock=code, market=market)
            raw = raw.rename(columns={
                "日期": "date",
                "主力净流入-净额": "main_net_inflow",
                "主力净流入-净占比": "main_net_inflow_ratio",
            })
            raw["date"] = pd.to_datetime(raw["date"])
            raw["code"] = code
            frames.append(raw[["date", "code", "main_net_inflow",
                               "main_net_inflow_ratio"]])
        except Exception as e:
            print(f"[load_money_flow] {code} 拉取失败: {e}")
    if not frames:
        raise RuntimeError("未能获取资金流数据。")
    df = pd.concat(frames).set_index(["date", "code"]).sort_index()
    if use_cache:
        df.to_parquet(cache)
    return df


def load_industry_map(use_cache: bool = True) -> pd.DataFrame:
    """
    返回 code -> industry(申万一级或中信一级) 的映射表。
    用于因子的行业中性化与组合的行业暴露约束。
    """
    cache = _cache_path("industry")
    if use_cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    import akshare as ak
    df = ak.stock_board_industry_name_em()  # 示意
    if use_cache:
        df.to_parquet(cache)
    return df


# ----------------------------------------------------------------------------
# 合成模拟数据：在无网络 / CI 环境下让全流程可跑通、可单元测试。
# ----------------------------------------------------------------------------
def make_synthetic_dataset(
    n_stocks: int = 80,
    start: str = "2016-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> dict:
    """
    生成结构上与真实数据一致的合成数据集，用于演示与回测引擎自测。
    价格用带漂移和横截面共同因子的几何布朗运动，使因子具备弱预测力。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    codes = [f"{600000 + i:06d}" if i % 2 == 0 else f"{300000 + i:06d}"
             for i in range(n_stocks)]

    industries = rng.integers(0, 10, n_stocks)
    # 每只股票一个隐藏"质量"得分，决定其长期漂移
    quality = rng.normal(0, 1, n_stocks)
    log_caps = rng.normal(23, 1.2, n_stocks)  # 市值对数

    T = len(dates)
    market = rng.normal(0.0003, 0.012, T)     # 市场共同因子
    price_panel, flow_panel = [], []

    for j, code in enumerate(codes):
        drift = 0.0002 + 0.0004 * quality[j]
        idio = rng.normal(0, 0.018, T)
        ret = drift + 0.9 * market + idio
        close = 10 * np.exp(np.cumsum(ret))
        openp = close * (1 + rng.normal(0, 0.004, T))
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.006, T)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.006, T)))
        vol = rng.lognormal(15, 0.5, T)
        amount = vol * close
        pre_close = np.r_[close[0], close[:-1]]

        df = pd.DataFrame({
            "date": dates, "code": code, "open": openp, "high": high,
            "low": low, "close": close, "volume": vol, "amount": amount,
            "pct_chg": np.r_[0, np.diff(close) / close[:-1]] * 100,
            "pre_close": pre_close,
        })
        price_panel.append(df)

        # 资金流与未来收益弱相关，制造一点可被因子捕捉的信号
        fwd = np.r_[ret[1:], 0]
        mf_ratio = 0.3 * fwd / 0.018 + rng.normal(0, 1, T)
        flow_panel.append(pd.DataFrame({
            "date": dates, "code": code,
            "main_net_inflow": mf_ratio * amount * 0.05,
            "main_net_inflow_ratio": mf_ratio,
        }))

    price = pd.concat(price_panel).set_index(["date", "code"]).sort_index()
    flow = pd.concat(flow_panel).set_index(["date", "code"]).sort_index()

    industry = pd.DataFrame({
        "code": codes,
        "industry": [f"IND_{i}" for i in industries],
        "log_cap": log_caps,
    }).set_index("code")

    # 财务数据：每季度一条，roe/毛利率与隐藏 quality 相关
    rep_dates = pd.date_range(start, end, freq="QE")
    fund_rows = []
    for j, code in enumerate(codes):
        for rd in rep_dates:
            fund_rows.append({
                "report_date": rd, "code": code,
                "roe": 0.10 + 0.06 * quality[j] + rng.normal(0, 0.02),
                "gross_margin": 0.30 + 0.10 * quality[j] + rng.normal(0, 0.03),
                "net_profit": np.exp(log_caps[j]) * 0.05 *
                              (1 + 0.1 * quality[j] + rng.normal(0, 0.05)),
                "bps": 5 + quality[j] + rng.normal(0, 0.5),
                "eps": 0.5 + 0.3 * quality[j] + rng.normal(0, 0.1),
            })
    fund = pd.DataFrame(fund_rows)
    # 公告日 = 报告期 + 保守滞后，防止未来函数
    fund["announce_date"] = fund["report_date"] + pd.Timedelta(days=45)
    fund = fund.set_index(["announce_date", "code"]).sort_index()

    return {"price": price, "flow": flow, "industry": industry, "fund": fund,
            "dates": dates, "codes": codes}
