"""
A股财务数据(东方财富数据中心) —— 激活价值/质量类因子。

背景
----
行情源(Yahoo/东财/腾讯)只有量价，缺财务，导致 15 个因子里价值(ep/bp)与
质量(roe/gross_margin/earnings_stability)5 个无法计算。本模块补上这块。

数据源
------
东财数据中心 datacenter-web.eastmoney.com 的 F10 主要财务指标接口
(RPT_F10_FINANCE_MAINFINADATA)，实测对沙箱/云端**可达**(与被封的 push2 不同)，
按 SECUCODE 逐只返回历年季度指标，含**公告日 NOTICE_DATE**——可做点位对齐、杜绝未来函数。

字段映射(东财 -> 本系统)
    ROEJQ           -> roe            (净资产收益率，%→小数)
    XSMLL           -> gross_margin   (销售毛利率，%→小数)
    EPSJB           -> eps            (每股收益，季度累计)
    BPS             -> bps            (每股净资产)
    PARENTNETPROFIT -> net_profit     (归母净利润)
返回 MultiIndex(announce_date, code)，与合成数据的 fund 结构一致，可直接接入
run_advanced.load_real 的公告日 ffill 对齐流程。

注意：财务为季度累计口径；因子在横截面上排序中性化，量纲不敏感。境内/沙箱均可用。
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

_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_HEADERS = {"User-Agent": "Mozilla/5.0",
            "Referer": "https://data.eastmoney.com/"}
_COLS = "SECUCODE,REPORT_DATE,NOTICE_DATE,EPSJB,BPS,ROEJQ,XSMLL,PARENTNETPROFIT"


def _secucode(code: str) -> str:
    return code + (".SH" if code.startswith(("6", "9", "5")) else ".SZ")


def _fetch_one(code: str, tries: int = 3, pause: float = 0.4) -> list[dict]:
    for attempt in range(tries):
        try:
            r = requests.get(_URL, params={
                "reportName": "RPT_F10_FINANCE_MAINFINADATA", "columns": _COLS,
                "filter": f'(SECUCODE="{_secucode(code)}")',
                "pageSize": 80, "pageNumber": 1,
                "sortColumns": "REPORT_DATE", "sortTypes": -1,
            }, headers=_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                res = r.json().get("result")
                if res and res.get("data"):
                    return res["data"]
                return []          # 有效响应但无数据
        except Exception:
            pass
        time.sleep(pause * (2 ** attempt))
    return []


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_fundamentals(
    codes: list[str],
    cache_path: str | None = None,
    polite_pause: float = 0.35,
) -> Optional[pd.DataFrame]:
    """
    批量抓取财务季度数据，返回 MultiIndex(announce_date, code)：
        列 = roe, gross_margin, eps, bps, net_profit
    任一环节整体失败(接口不可达)则返回 None，调用方据此跳过价值/质量因子。
    """
    if requests is None:
        return None
    if cache_path:
        pkl = cache_path if cache_path.endswith(".pkl") else cache_path + ".pkl"
        if os.path.exists(pkl):
            try:
                return pd.read_pickle(pkl)
            except Exception:
                pass

    rows, ok = [], 0
    for code in codes:
        recs = _fetch_one(code)
        if recs:
            ok += 1
        for x in recs:
            nd = x.get("NOTICE_DATE")
            if not nd:
                continue
            roe = _to_num(x.get("ROEJQ"))
            gm = _to_num(x.get("XSMLL"))
            rows.append({
                "announce_date": pd.to_datetime(nd).normalize(),
                "code": code,
                "roe": roe / 100 if roe is not None else None,
                "gross_margin": gm / 100 if gm is not None else None,
                "eps": _to_num(x.get("EPSJB")),
                "bps": _to_num(x.get("BPS")),
                "net_profit": _to_num(x.get("PARENTNETPROFIT")),
            })
        time.sleep(polite_pause)

    if ok == 0 or not rows:
        print("[cn_fundamental] 财务接口不可达或无数据，跳过价值/质量因子。")
        return None
    df = (pd.DataFrame(rows)
          .dropna(subset=["announce_date", "code"])
          .drop_duplicates(["announce_date", "code"], keep="last")
          .set_index(["announce_date", "code"]).sort_index())
    print(f"[cn_fundamental] 财务数据 {ok}/{len(codes)} 只，共 {len(df)} 条季度记录。")
    if cache_path:
        try:
            df.to_pickle(cache_path if cache_path.endswith(".pkl")
                         else cache_path + ".pkl")
        except Exception:
            pass
    return df
