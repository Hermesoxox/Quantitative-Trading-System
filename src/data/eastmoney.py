"""
东方财富(Eastmoney)直连数据抓取器 —— 真实A股数据落地实现。

为什么自建抓取器而不直接依赖 akshare
------------------------------------
* akshare 把数据源主机名硬编码到个别被限流/被墙的子域，环境受限时整体不可用，
  且其顶层 import 会拖入一堆易冲突的依赖(jsonpath/py_mini_racer/curl_cffi)。
* 本抓取器只用 requests，直连东财公开行情接口，字段/复权口径已对齐真实返回，
  带重试退避与 parquet 缓存，便于离线复现与增量更新。

接口与字段(已对真实返回校验)
----------------------------
GET https://push2.eastmoney.com/api/qt/stock/kline/get
  secid  = f"{market}.{code}"   market: 1=沪市, 0=深市
  klt=101 日线;  fqt=1 前复权 / 2 后复权 / 0 不复权
  fields2 = f51..f61: 日期,开,收,高,低,成交量(手),成交额(元),振幅,涨跌幅,涨跌额,换手率

注意: 若运行环境网络策略限制了数据API主机(返回 502/503/空)，请在
本机或放宽网络策略的环境运行。代码本身与真实接口一致，无需改动。
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

_KLINE_URL = "https://push2.eastmoney.com/api/qt/stock/kline/get"
# 备选主机，按可用性轮询(不同环境放行的子域不同)
_HOSTS = ["push2.eastmoney.com", "push2his.eastmoney.com",
          "13.push2his.eastmoney.com", "82.push2his.eastmoney.com"]
_HEADERS = {"User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/"}

# klines 字段顺序(fields2)
_KCOLS = ["date", "open", "close", "high", "low", "volume", "amount",
          "amplitude", "pct_chg", "change", "turnover"]


def _secid(code: str) -> str:
    """根据代码前缀生成东财 secid。6/9开头=沪市(1)，其余=深市(0)。"""
    market = 1 if code.startswith(("6", "9", "5")) else 0
    return f"{market}.{code}"


def fetch_kline(
    code: str,
    start: str,
    end: str,
    adjust: str = "qfq",
    tries: int = 5,
    pause: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    抓取单只股票日线(默认前复权)。返回标准化 DataFrame 或 None。

    带主机轮询 + 指数退避重试，应对限流/瞬时故障。
    """
    if requests is None:
        raise RuntimeError("requests 未安装")
    fqt = {"qfq": 1, "hfq": 2, "none": 0}.get(adjust, 1)
    params = {
        "secid": _secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": str(fqt),
        "beg": start.replace("-", ""), "end": end.replace("-", ""),
    }
    last_err = None
    for attempt in range(tries):
        host = _HOSTS[attempt % len(_HOSTS)]
        url = _KLINE_URL.replace("push2.eastmoney.com", host)
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                j = r.json()
                data = j.get("data")
                if data and data.get("klines"):
                    return _parse_klines(code, data["klines"])
            last_err = f"status={r.status_code}"
        except Exception as e:  # 网络瞬断/超时
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(pause * (2 ** attempt))  # 指数退避: 0.5,1,2,4,8s
    print(f"[fetch_kline] {code} 失败: {last_err}")
    return None


def _parse_klines(code: str, klines: list[str]) -> pd.DataFrame:
    """把 ['日期,开,收,...', ...] 解析为标准化长表行。"""
    rows = [s.split(",") for s in klines]
    df = pd.DataFrame(rows, columns=_KCOLS)
    df["date"] = pd.to_datetime(df["date"])
    num = ["open", "close", "high", "low", "volume", "amount",
           "amplitude", "pct_chg", "change", "turnover"]
    df[num] = df[num].astype(float)
    df["code"] = code
    df["pre_close"] = df["close"].shift(1)
    return df[["date", "code", "open", "high", "low", "close",
               "volume", "amount", "pct_chg", "pre_close"]]


def fetch_universe_prices(
    codes: list[str],
    start: str,
    end: str,
    adjust: str = "qfq",
    cache_path: str | None = None,
    polite_pause: float = 0.3,
) -> pd.DataFrame:
    """
    批量抓取一篮子股票，拼成 MultiIndex(date, code) 长表并可缓存。
    与 src/data/loader.load_daily_price 的输出结构完全一致，可直接替换。
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    frames = []
    from tqdm import tqdm
    for code in tqdm(codes, desc="fetch klines"):
        df = fetch_kline(code, start, end, adjust)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(polite_pause)  # 控速，避免触发限流
    if not frames:
        raise RuntimeError(
            "未获取到任何行情。多半是运行环境网络策略限制了数据API主机，"
            "请在本机或放宽网络策略的环境重试。")
    panel = pd.concat(frames).set_index(["date", "code"]).sort_index()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        panel.to_parquet(cache_path)
    return panel


def fetch_csi300_codes(tries: int = 5) -> list[str]:
    """
    取沪深300成分股代码(作为容量充足、流动性好的默认股票池)。
    用东财指数成分接口；失败时回退到内置的部分蓝筹清单。
    """
    fallback = ["600519", "601318", "600036", "000858", "600900",
                "000333", "600276", "601166", "002594", "600030",
                "000001", "601888", "600887", "000651", "600309"]
    if requests is None:
        return fallback
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {"pn": "1", "pz": "400", "po": "1", "np": "1",
              "fs": "b:BK0500", "fields": "f12"}  # 沪深300板块
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                diff = r.json().get("data", {}).get("diff", [])
                codes = [d["f12"] for d in diff] if diff else []
                if codes:
                    return codes
        except Exception:
            pass
        time.sleep(0.5 * (2 ** attempt))
    print("[fetch_csi300_codes] 成分接口不可用，回退到内置蓝筹清单。")
    return fallback
