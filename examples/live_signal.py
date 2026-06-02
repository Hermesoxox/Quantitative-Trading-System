"""
盘后实盘信号生成器(日频) —— 把回测策略变成"明天开盘怎么操作"的可执行清单。

定位
----
本脚本是策略的"实盘适配层":每个交易日**收盘后**运行一次，它会
  1. 拉取最新行情(到今天)，复算因子/综合得分/市场状态/个股风控；
  2. 读取你当前的持仓(positions.json，可选)；
  3. 输出三张清单 + 风险预警：
       【卖出】触发止损/跌破60日线/得分跌出/熊市清仓的持仓
       【买入】进入目标前5且趋势通过的标的(按目标权重×当前总仓位给出股数)
       【持有】已在目标内、继续持有的
       【风控预警】每只持仓的止损价、是否接近止损/移动止盈

为什么是"盘后"而不是"实时盯盘"
------------------------------
策略是 T+1、持仓 5–20 日的中频策略：收盘算信号、次日开盘成交即可，
日内每秒盯盘既无必要(会被成本和T+1吃掉)，也不在本策略的能力圈内。

重要声明
--------
* 本脚本**只产出建议清单，不连券商、不自动下单**。最终下单与否由你决定。
* 仅供研究/辅助决策，不构成投资建议；实盘盈亏自负，务必先小资金验证。

用法
----
    # 空仓起步，给出建仓买入清单(默认蓝筹池)
    python -m examples.live_signal --capital 1000000

    # 已有持仓：准备 positions.json 后传入
    python -m examples.live_signal --positions positions.json --capital 1000000

positions.json 格式：
    {"600519": {"shares": 100, "cost": 1500.0},
     "000858": {"shares": 500, "cost": 180.0}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import PORTFOLIO, RISK, OVERLAY
from src.data.eastmoney import fetch_universe_prices, fetch_csi300_codes, fetch_kline
from src.factors import compute_all_factors, neutralize_factor
from src.signals import composite_score, trend_filter, smooth_score
from src.portfolio import build_target_weights
from src.regime import combined_exposure
from src.ml.combiner import MLCombiner, _stack_features, _forward_return_label


def long_to_wide(panel, field):
    return panel[field].unstack("code").sort_index()


def latest_ml_score(neut, close, horizon=5, embargo=10):
    """用截至(今天-禁运)的历史训练一次 GBDT，预测最新截面得分(因果，无前视)。"""
    feats = _stack_features(neut).dropna(how="all")
    label = _forward_return_label(close, horizon)
    data = feats.join(label.rename("_y"), how="inner").dropna()
    if data.empty:
        return pd.Series(dtype=float)
    today = close.index.max()
    cutoff = today - pd.Timedelta(days=embargo + horizon)
    train = data[data.index.get_level_values("date") <= cutoff]
    cols = list(neut.keys())
    if len(train) < 500:
        # 历史太短，退化为线性等权合成
        from config import DEFAULT_FACTOR_WEIGHTS
        w = {k: v for k, v in DEFAULT_FACTOR_WEIGHTS.items() if k in neut}
        row = {n: f.loc[[today]] for n, f in neut.items() if today in f.index}
        return composite_score(row, w).loc[today] if row else pd.Series(dtype=float)
    model = MLCombiner().fit(train[cols], train["_y"])
    latest = pd.DataFrame({c: neut[c].loc[today] for c in cols}).dropna()
    if latest.empty:
        return pd.Series(dtype=float)
    return pd.Series(model.predict(latest), index=latest.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="", help="持仓 JSON 文件(可选)")
    ap.add_argument("--capital", type=float, default=1_000_000, help="总资金")
    ap.add_argument("--codes", default="", help="逗号分隔股票池(默认蓝筹池)")
    ap.add_argument("--n", type=int, default=40, help="股票池数量")
    args = ap.parse_args()

    # 1) 拉取最近约2.2年行情(足够算200日均线与因子)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=820)
    codes = ([c.strip() for c in args.codes.split(",") if c.strip()]
             or fetch_csi300_codes())[:args.n]
    print(f"股票池 {len(codes)} 只，行情区间 {start.date()} ~ {end.date()}")
    price = fetch_universe_prices(codes, str(start.date()), str(end.date()),
                                  polite_pause=0.6)
    wide = {f: long_to_wide(price, f) for f in
            ["open", "high", "low", "close", "volume", "amount", "pre_close"]}
    close = wide["close"]
    today = close.index.max()
    industry = pd.Series("ALL", index=close.columns, name="industry")
    log_cap = np.log(wide["amount"].mean()).rename("log_cap")

    # 2) 因子 -> 中性化 -> 最新截面综合得分
    raw = compute_all_factors(wide)
    neut = {k: neutralize_factor(v, industry, log_cap) for k, v in raw.items()}
    score_row = latest_ml_score(neut, close)
    if score_row.empty:
        print("无法生成得分(数据不足)。"); return

    # 3) 市场状态总仓位(今天)
    bench = fetch_kline("000300", str(start.date()), str(end.date()))
    benchmark = None
    if bench is not None:
        b = bench.set_index("date")["close"].reindex(close.index)
        if b.notna().mean() >= 0.9:
            benchmark = b
    exposure = combined_exposure(close, benchmark=benchmark,
                                 ma_window=OVERLAY.regime_ma,
                                 target_vol=OVERLAY.target_vol).iloc[-1]

    # 4) 目标组合
    trend_row = trend_filter(close, 200).iloc[-1]
    vol_row = (close.pct_change().rolling(20).std() * np.sqrt(252)).iloc[-1]
    target_w = build_target_weights(score_row, trend_row, vol_row, industry)
    investable = args.capital * float(exposure)

    # 5) 当前持仓
    positions = {}
    if args.positions and os.path.exists(args.positions):
        positions = json.load(open(args.positions, encoding="utf-8"))

    ranked = score_row.rank(ascending=False, pct=True)
    ma60 = close.rolling(60).mean().iloc[-1]
    px = close.iloc[-1]

    sells, holds, alerts = [], [], []
    bear = float(exposure) < 0.10
    for code, pos in positions.items():
        p = float(px.get(code, np.nan))
        cost = pos.get("cost")
        reason = None
        if bear:
            reason = "熊市清仓(市场状态转空)"
        elif cost and p <= cost * (1 - RISK.stop_loss):
            reason = f"止损(现价{p:.2f}≤成本{cost:.2f}×{1-RISK.stop_loss:.2f})"
        elif code in ma60.index and p < ma60[code]:
            reason = f"跌破60日线({p:.2f}<{ma60[code]:.2f})"
        elif ranked.get(code, 1.0) > PORTFOLIO.hold_buffer_rank:
            reason = f"得分跌出前{int(PORTFOLIO.hold_buffer_rank*100)}%"
        if reason:
            sells.append((code, pos.get("shares"), reason))
        else:
            holds.append((code, pos.get("shares")))
            if cost:
                stop = cost * (1 - RISK.stop_loss)
                gain = p / cost - 1
                alerts.append(f"  {code}: 现价{p:.2f} 成本{cost:.2f} "
                              f"浮盈{gain:+.1%} 止损价{stop:.2f}"
                              + ("  ⚠️接近止损" if p <= stop * 1.03 else ""))

    held = set(positions)
    buys = []
    for code, w in target_w.items():
        if code in held:
            continue
        p = float(px.get(code, np.nan))
        if not np.isfinite(p) or p <= 0:
            continue
        shares = int(np.floor(investable * w / p / 100) * 100)
        if shares > 0:
            buys.append((code, shares, w, p))

    # 6) 报告
    print("\n" + "=" * 60)
    print(f"  盘后实盘信号  日期: {today.date()}")
    print(f"  市场状态总仓位: {float(exposure):.0%}"
          + ("  ❄️ 熊市/高波，建议空仓或极轻仓" if bear else "  ☀️ 可持仓"))
    print("=" * 60)
    print("\n【次日开盘 — 卖出】")
    print("  无" if not sells else "")
    for c, s, r in sells:
        print(f"  ✗ 卖出 {c}  {s}股  ← {r}")
    print("\n【次日开盘 — 买入】(目标前5，按当前总仓位分配)")
    print("  无" if not buys else "")
    for c, s, w, p in buys:
        print(f"  ✓ 买入 {c}  {s}股  约{s*p:,.0f}元  目标权重{w:.0%}  现价{p:.2f}")
    print("\n【继续持有】")
    print("  无" if not holds else "")
    for c, s in holds:
        print(f"  • 持有 {c}  {s}股")
    if alerts:
        print("\n【持仓风控预警】")
        print("\n".join(alerts))
    print("\n注：以上为辅助建议，非投资建议；请人工复核后再决定是否下单。")


if __name__ == "__main__":
    main()
