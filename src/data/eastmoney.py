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

# 指数代码（走 Yahoo 指数路径，作 regime 基准）
_INDEX_CODES = {"000300", "000905", "000016", "399006"}


def _secid(code: str) -> str:
    """根据代码前缀生成东财 secid。6/9开头=沪市(1)，其余=深市(0)。"""
    market = 1 if code.startswith(("6", "9", "5")) else 0
    return f"{market}.{code}"


def fetch_kline(
    code: str,
    start: str,
    end: str,
    adjust: str = "qfq",
    tries: int = 4,
    pause: float = 0.6,
) -> Optional[pd.DataFrame]:
    """
    抓取单只股票日线(默认前复权)。返回标准化 DataFrame 或 None。

    多源容错：先试东方财富(主机轮询+指数退避)，失败再回退腾讯接口。
    实测东财对云端/海外 IP 会大面积 502 限流，腾讯接口更稳，故双源互备。
    """
    if requests is None:
        raise RuntimeError("requests 未安装")
    # 数据源优先级：Yahoo(国际可达, 云端/CI首选) -> 东财 -> 腾讯(境内更快)。
    # 这样同一份代码在沙箱/CI(走Yahoo)与用户境内Mac(走东财/腾讯)都能拿到数据。
    is_index = code in _INDEX_CODES
    from .yahoo import fetch_kline_yahoo
    df = fetch_kline_yahoo(code, start, end, adjust, is_index=is_index)
    if df is not None and not df.empty:
        return df
    if is_index:
        return None
    df = _fetch_eastmoney(code, start, end, adjust, tries, pause)
    if df is not None and not df.empty:
        return df
    return _fetch_tencent(code, start, end, adjust)


def _fetch_eastmoney(code, start, end, adjust, tries, pause):
    fqt = {"qfq": 1, "hfq": 2, "none": 0}.get(adjust, 1)
    params = {
        "secid": _secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": str(fqt),
        "beg": start.replace("-", ""), "end": end.replace("-", ""),
    }
    for attempt in range(tries):
        host = _HOSTS[attempt % len(_HOSTS)]
        url = _KLINE_URL.replace("push2.eastmoney.com", host)
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                data = r.json().get("data")
                if data and data.get("klines"):
                    return _parse_klines(code, data["klines"])
        except Exception:
            pass
        time.sleep(pause * (2 ** attempt))
    return None


def _tencent_symbol(code: str) -> str:
    """腾讯接口前缀：6/9/5 开头沪市 sh，其余深市 sz。"""
    return ("sh" if code.startswith(("6", "9", "5")) else "sz") + code


def _fetch_tencent(code, start, end, adjust, tries=3, pause=0.6):
    """
    腾讯财经日线(前复权)。接口:
      https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
      param = <symbol>,day,<start>,<end>,<count>,<qfq|hfq|''>
    返回 data[symbol]["qfqday"|"day"] = [[date,open,close,high,low,volume,...], ...]
    """
    sym = _tencent_symbol(code)
    fq = {"qfq": "qfq", "hfq": "hfq", "none": ""}.get(adjust, "qfq")
    key = {"qfq": "qfqday", "hfq": "hfqday", "none": "day"}.get(adjust, "qfqday")
    param = f"{sym},day,{start},{end},2600,{fq}"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    for attempt in range(tries):
        try:
            r = requests.get(url, params={"param": param},
                             headers=_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                d = r.json().get("data", {}).get(sym, {})
                rows = d.get(key) or d.get("day")
                if rows:
                    return _parse_tencent(code, rows)
        except Exception:
            pass
        time.sleep(pause * (2 ** attempt))
    print(f"[fetch_kline] {code} 东财+腾讯均失败")
    return None


def _parse_tencent(code: str, rows: list) -> pd.DataFrame:
    """腾讯 [date,open,close,high,low,volume(手),...] -> 标准化长表。"""
    recs = []
    for r in rows:
        try:
            recs.append((r[0], float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5])))
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(recs, columns=["date", "open", "close", "high",
                                     "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    # 腾讯不直接给成交额，用 量(手)×收盘×100股 近似，供流动性/规模因子使用
    df["amount"] = df["volume"] * df["close"] * 100
    df["pct_chg"] = df["close"].pct_change() * 100
    df["pre_close"] = df["close"].shift(1)
    return df[["date", "code", "open", "high", "low", "close",
               "volume", "amount", "pct_chg", "pre_close"]]


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
    polite_pause: float = 0.5,
) -> pd.DataFrame:
    """
    批量抓取一篮子股票，拼成 MultiIndex(date, code) 长表并可缓存。
    与 src/data/loader.load_daily_price 的输出结构完全一致，可直接替换。
    缓存读写对 parquet 引擎缺失做容错(回退 pickle)，不因缓存问题中断回测。
    """
    if cache_path:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    frames, ok, fail = [], 0, 0
    from tqdm import tqdm
    for code in tqdm(codes, desc="fetch klines"):
        df = fetch_kline(code, start, end, adjust)
        if df is not None and not df.empty:
            frames.append(df); ok += 1
        else:
            fail += 1
        time.sleep(polite_pause)  # 控速，避免触发限流
    print(f"[fetch_universe_prices] 成功 {ok} / 失败 {fail}")
    if not frames:
        raise RuntimeError(
            "未获取到任何行情。东财与腾讯接口均不可达，"
            "请在本机或放宽网络策略的环境重试。")
    panel = pd.concat(frames).set_index(["date", "code"]).sort_index()
    if cache_path:
        _write_cache(panel, cache_path)
    return panel


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    """优先读 parquet，缺引擎或失败时回退同名 .pkl。"""
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    pkl = path + ".pkl"
    if os.path.exists(pkl):
        try:
            return pd.read_pickle(pkl)
        except Exception:
            pass
    return None


def _write_cache(panel: pd.DataFrame, path: str) -> None:
    """写缓存：parquet 不可用则回退 pickle，绝不因此中断主流程。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        panel.to_parquet(path)
    except Exception as e:
        try:
            panel.to_pickle(path + ".pkl")
            print(f"[cache] parquet 不可用({type(e).__name__})，已回退 pickle。")
        except Exception:
            print("[cache] 缓存写入失败，跳过(不影响本次回测)。")


def fetch_csi300_codes(tries: int = 5) -> list[str]:
    """
    取沪深300成分股代码(作为容量充足、流动性好的默认股票池)。
    用东财指数成分接口；失败时回退到内置的部分蓝筹清单。
    """
    # 内置约 60 只沪深300大盘蓝筹清单（境外成分接口不可达时的稳健兜底）
    fallback = [
        "600519", "601318", "600036", "000858", "600900", "000333", "600276",
        "601166", "002594", "600030", "000001", "601888", "600887", "000651",
        "600309", "601012", "600028", "601398", "601628", "600585", "000002",
        "600031", "603259", "601668", "600048", "000725", "002415", "300750",
        "601288", "601988", "600000", "601857", "600104", "601601", "601211",
        "600690", "000568", "002475", "600406", "603288", "601336", "600196",
        "000776", "002714", "601066", "600438", "601899", "603501", "600009",
        "000538", "002304", "600436", "601225", "000063", "600745", "603986",
        "002241", "600547", "601088", "300059",
    ]
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
