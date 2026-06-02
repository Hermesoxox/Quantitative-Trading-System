"""
Yahoo Finance 数据源（国际可达，A股用 .SS/.SZ 后缀）。

为什么需要它
------------
东方财富/腾讯接口对**境外/云端 IP**（GitHub Actions 美区 runner、部分沙箱）
会整体封锁，导致在这些环境根本拉不到 A 股数据。Yahoo Finance 是全球 CDN，
在这些环境通常可达，因此作为"国际兜底"数据源，使真实回测能在云端/CI 跑起来。

口径
----
* 复权：用 Yahoo 的 adjclose 反推复权比例 ratio=adjclose/close，对 OHLC 同比
  缩放，得到一致的后复权序列（与因子计算自洽即可）。
* 成交量：Yahoo 为"股"，成交额用 volume*close 近似（供流动性/规模因子）。
* 指数：CSI300=000300.SS 等通过 INDEX_SYMBOLS 映射，供 regime 层作基准。
"""

from __future__ import annotations

import time
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

# 常用指数的 Yahoo 代码（消除 000xxx 股票/指数的二义性）
INDEX_SYMBOLS = {
    "000300": "000300.SS",   # 沪深300
    "000905": "000905.SS",   # 中证500
    "000016": "000016.SS",   # 上证50
    "399006": "399006.SZ",   # 创业板指
}


def _yahoo_symbol(code: str, is_index: bool = False) -> str:
    if is_index and code in INDEX_SYMBOLS:
        return INDEX_SYMBOLS[code]
    # 6/9/5 开头沪市 .SS，其余深市 .SZ
    return code + (".SS" if code.startswith(("6", "9", "5")) else ".SZ")


def _ts(date_str: str) -> int:
    y, m, d = map(int, date_str.split("-"))
    return int(time.mktime(dt.date(y, m, d).timetuple()))


def fetch_kline_yahoo(
    code: str,
    start: str,
    end: str,
    adjust: str = "qfq",
    is_index: bool = False,
    tries: int = 4,
    pause: float = 0.8,
) -> Optional[pd.DataFrame]:
    """抓取单只(或指数)日线，返回标准化长表或 None。带 429 限流重试。"""
    if requests is None:
        return None
    sym = _yahoo_symbol(code, is_index)
    params = {"period1": _ts(start), "period2": _ts(end),
              "interval": "1d", "events": "div,splits"}
    for attempt in range(tries):
        try:
            r = requests.get(_CHART.format(sym=sym), params=params,
                             headers=_HEADERS, timeout=20)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                res = r.json().get("chart", {}).get("result")
                if res:
                    df = _parse_yahoo(code, res[0], adjust)
                    if df is not None and not df.empty:
                        return df
            # 429/5xx -> 退避重试
        except Exception:
            pass
        time.sleep(pause * (2 ** attempt))
    return None


def _parse_yahoo(code, result, adjust) -> Optional[pd.DataFrame]:
    ts = result.get("timestamp")
    ind = result.get("indicators", {})
    q = (ind.get("quote") or [{}])[0]
    if not ts or not q:
        return None
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s").normalize(),
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"),
        "volume": q.get("volume"),
    }).dropna(subset=["close"])
    # 复权：用 adjclose 反推比例缩放 OHLC（后复权口径，自洽即可）
    adj = (ind.get("adjclose") or [{}])[0].get("adjclose")
    if adjust in ("qfq", "hfq") and adj is not None:
        adj = pd.Series(adj, index=df.index[:len(adj)]).reindex(df.index)
        ratio = (adj / df["close"]).fillna(1.0)
        for c in ["open", "high", "low"]:
            df[c] = df[c] * ratio
        df["close"] = adj.fillna(df["close"])
    df["code"] = code
    df["volume"] = df["volume"].fillna(0)
    df["amount"] = df["volume"] * df["close"]      # 成交额近似
    df["pct_chg"] = df["close"].pct_change() * 100
    df["pre_close"] = df["close"].shift(1)
    return df[["date", "code", "open", "high", "low", "close",
               "volume", "amount", "pct_chg", "pre_close"]].reset_index(drop=True)
