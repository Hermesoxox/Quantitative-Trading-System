"""
A股资金流数据(东财 fflow 接口) + 全市场股票名单(东财数据中心)。

激活最后两个因子: mf_3d_ratio(3日主力净流入比率)、并让 vol_surge 之外的
资金维度进入模型; 同时提供"全A股名单"以支持真正的全市场选股。

接口(字段已对真实返回校验)
--------------------------
1) 个股资金流日线:
   GET https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get
     secid=1.600519  klt=101  lmt=0
     fields2=f51..f61: 日期,主力净流入额,小单,中单,大单,超大单,
                       主力净占比%,小单占比,中单占比,大单占比,超大单占比
   注意: 该接口对云端/境外IP限流不稳定(与行情同源),在境内Mac上稳定;
   且历史深度有限(通常约近1-2年),更早区间因子自动为空(模型按中性0处理)。

2) 全A股名单(数据中心,实测云端也可达):
   RPT_LICO_FN_CPD 按报告期分页返回全部上市公司 SECURITY_CODE/NAME,
   用于构建全市场股票池(剔除ST/退市标记由名称过滤,流动性过滤在拉到行情后做)。
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

_HEADERS = {"User-Agent": "Mozilla/5.0",
            "Referer": "https://data.eastmoney.com/"}
_FFLOW_HOSTS = ["push2his.eastmoney.com", "push2.eastmoney.com"]
# 注: push2his 返回约近半年(~120交易日)历史——接口天然深度上限;
#     push2 仅当日快照(兜底,足够盘后信号使用)。更早区间因子按中性0处理。


def _secid(code: str) -> str:
    return ("1." if code.startswith(("6", "9", "5")) else "0.") + code


def fetch_moneyflow(code: str, tries: int = 4, pause: float = 0.5
                    ) -> Optional[pd.DataFrame]:
    """
    单只股票的主力资金流日线。返回长表(date, code, main_net_inflow,
    main_net_inflow_ratio) 或 None。主机轮询+指数退避。
    """
    if requests is None:
        return None
    params = {"secid": _secid(code), "fields1": "f1,f2,f3,f7",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
              "lmt": "0", "klt": "101"}
    for attempt in range(tries):
        host = _FFLOW_HOSTS[attempt % len(_FFLOW_HOSTS)]
        url = f"https://{host}/api/qt/stock/fflow/daykline/get"
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                data = r.json().get("data")
                if data and data.get("klines"):
                    return _parse_fflow(code, data["klines"])
        except Exception:
            pass
        time.sleep(pause * (2 ** attempt))
    return None


def _parse_fflow(code: str, klines: list[str]) -> pd.DataFrame:
    rows = []
    for s in klines:
        p = s.split(",")
        try:
            rows.append((pd.to_datetime(p[0]), float(p[1]), float(p[6])))
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows, columns=["date", "main_net_inflow",
                                     "main_net_inflow_ratio"])
    df["code"] = code
    return df[["date", "code", "main_net_inflow", "main_net_inflow_ratio"]]


def fetch_moneyflow_panel(codes: list[str], cache_path: str | None = None,
                          polite_pause: float = 0.4) -> Optional[pd.DataFrame]:
    """批量抓取资金流 -> MultiIndex(date, code)。整体失败返回 None(因子自动跳过)。"""
    if cache_path:
        pkl = cache_path if cache_path.endswith(".pkl") else cache_path + ".pkl"
        if os.path.exists(pkl):
            try:
                return pd.read_pickle(pkl)
            except Exception:
                pass
    frames, ok = [], 0
    for code in codes:
        df = fetch_moneyflow(code)
        if df is not None and not df.empty:
            frames.append(df); ok += 1
        time.sleep(polite_pause)
    if ok == 0:
        print("[cn_moneyflow] 资金流接口不可达(云端常见)，跳过资金流因子；"
              "境内Mac上运行可自动激活。")
        return None
    panel = (pd.concat(frames).drop_duplicates(["date", "code"])
             .set_index(["date", "code"]).sort_index())
    print(f"[cn_moneyflow] 资金流 {ok}/{len(codes)} 只，共 {len(panel)} 行。")
    if cache_path:
        try:
            panel.to_pickle(cache_path if cache_path.endswith(".pkl")
                            else cache_path + ".pkl")
        except Exception:
            pass
    return panel


# ---------------------------------------------------------------- 全市场名单
def fetch_all_a_codes(report_date: str = "2024-12-31", tries: int = 3,
                      max_pages: int = 60) -> list[str]:
    """
    全A股代码清单(东财数据中心业绩报表,分页遍历)。剔除名称含 ST/退 的标的。
    接口对云端可达; 失败返回空列表(调用方回退蓝筹清单)。
    """
    if requests is None:
        return []
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    seen: dict[str, str] = {}
    page = 1
    while page <= max_pages:
        got = None
        for attempt in range(tries):
            try:
                r = requests.get(url, params={
                    "reportName": "RPT_LICO_FN_CPD",
                    "columns": "SECURITY_CODE,SECURITY_NAME_ABBR",
                    "filter": f"(REPORTDATE='{report_date}')",
                    "pageSize": 100, "pageNumber": page,
                    "sortColumns": "SECURITY_CODE", "sortTypes": 1,
                }, headers=_HEADERS, timeout=15)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    res = r.json().get("result")
                    got = (res or {}).get("data") or []
                    break
            except Exception:
                pass
            time.sleep(0.5 * (2 ** attempt))
        if got is None:          # 网络失败
            break
        if not got:              # 翻完
            break
        for x in got:
            c, n = x.get("SECURITY_CODE"), x.get("SECURITY_NAME_ABBR") or ""
            if not c or "ST" in n.upper() or "退" in n:
                continue
            # 只保留 A股主板/创业板/科创板代码段
            if c.startswith(("60", "00", "30", "68")):
                seen[c] = n
        page += 1
    codes = sorted(seen)
    if codes:
        print(f"[universe] 全市场名单 {len(codes)} 只(已剔除ST/退,截至报告期 {report_date})。")
    return codes
