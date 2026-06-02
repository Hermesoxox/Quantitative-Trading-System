"""
走步式(walk-forward)自适应回测 —— 把"风险叠加层"做成滚动自标定，而非写死数字。

为什么这是最诚实的回测
----------------------
固定一个 target_vol=10% 然后报告全样本回撤，本质是"看着答案选参数"。真正的
做法是：每段只用它**之前**的数据来选择该段要用的风险设置，再把各段拼成一条
**纯样本外**净值曲线。这样报告的绩效里没有任何"事后诸葛"的成分。

流程(每 12 个月一折)
--------------------
1. 因子权重：滚动 IC 加权(rolling_factor_weights)，只用训练窗 -> 线性合成得分。
2. 风险叠加层选择：候选 = {固定目标波动 0.08/0.10/0.12, 规则化自适应}。
   在每折的"训练窗"上用一个轻量代理策略(top-5 等权篮子×该候选总仓位)的
   Calmar 比率挑出最优候选——只看训练期，不看未来。
3. 把每折"训练期选定"的总仓位序列，拼到其后的"验证期(OOS)"上。
4. 用拼好的总仓位时间表跑一次回测，仅在 OOS 区间统计绩效，并出 DSR/分折表。

运行: python -m examples.run_walkforward
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import PERIOD, OVERLAY
from src.data.loader import make_synthetic_dataset
from src.factors import compute_all_factors, neutralize_factor
from src.signals import composite_score, trend_filter, smooth_score
from src.regime import combined_exposure, adaptive_combined_exposure
from src.optimization.rolling import rolling_factor_weights, weights_for_date
from src.backtest import Backtester, performance_summary, annual_breakdown
from analysis.factor_ic import evaluate_factor_library
from analysis.overfit import deflated_sharpe_ratio
from analysis.plots import generate_report


def long_to_wide(panel, field):
    return panel[field].unstack("code").sort_index()


def proxy_equity(score, close, exposure, top_k=5):
    """轻量代理策略净值：每日按得分取 top-k 等权，下一日收益×当日总仓位。"""
    nxt = close.pct_change().shift(-1)
    rank = score.rank(axis=1, ascending=False)
    mask = (rank <= top_k).astype(float)
    cnt = mask.sum(axis=1).replace(0, np.nan)
    basket = (nxt * mask).sum(axis=1) / cnt
    strat = (basket.fillna(0) * exposure.shift(1).fillna(0))  # 仓位滞后,防前视
    return (1 + strat).cumprod()


def calmar(equity):
    eq = equity.dropna()
    if len(eq) < 20:
        return -np.inf
    ann = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    return ann / abs(dd) if dd < 0 else ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用真实数据(Yahoo/东财)")
    ap.add_argument("--n", type=int, default=0, help="股票数(0=全部/合成80)")
    ap.add_argument("--codes", type=str, default="")
    ap.add_argument("--start", default=PERIOD.in_sample_start)
    ap.add_argument("--end", default=PERIOD.out_sample_end)
    args = ap.parse_args()

    print("=" * 72)
    print("走步式自适应回测 — 风险叠加层滚动自标定 + 纯样本外拼接"
          + ("  [真实数据]" if args.real else "  [合成数据]"))
    print("=" * 72)

    benchmark = None
    if args.real:
        from examples.run_advanced import load_real
        wide, industry, log_cap, benchmark = load_real(
            args.n, args.start, args.end, args.codes)
        close = wide["close"]
    else:
        ds = make_synthetic_dataset(n_stocks=80, start=args.start, end=args.end)
        wide = {f: long_to_wide(ds["price"], f) for f in
                ["open", "high", "low", "close", "volume", "amount", "pre_close"]}
        wide["mf_ratio"] = long_to_wide(ds["flow"], "main_net_inflow_ratio")
        close = wide["close"]
        industry, log_cap = ds["industry"]["industry"], ds["industry"]["log_cap"]
        fund = ds["fund"]
        def align(col):
            f = fund.reset_index().pivot_table(index="announce_date",
                                               columns="code", values=col)
            return f.reindex(close.index, method="ffill").reindex(
                columns=close.columns)
        for c in ["roe", "gross_margin", "net_profit", "bps", "eps"]:
            wide[c] = align(c)

    raw = compute_all_factors(wide)
    neut = {n: neutralize_factor(f, industry, log_cap) for n, f in raw.items()}

    # 因子权重：滚动 IC 加权(只用训练窗) -> 线性合成
    wsched = rolling_factor_weights(neut, close, list(close.index),
                                    train_months=PERIOD.roll_train_months,
                                    valid_months=PERIOD.roll_valid_months,
                                    step_months=PERIOD.roll_step_months, horizon=5)
    from config import DEFAULT_FACTOR_WEIGHTS
    avail = {k: v for k, v in DEFAULT_FACTOR_WEIGHTS.items() if k in neut}
    score = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for date in close.index:
        w = avail if (wsched.empty or date < wsched.index.min()) \
            else weights_for_date(wsched, date)
        row = {n: f.loc[[date]] for n, f in neut.items() if date in f.index}
        if row:
            score.loc[date] = composite_score(row, w).loc[date]
    score = smooth_score(score)

    # 候选风险叠加层(全样本各算一次，校准时只切训练窗)；真实数据带基准
    candidates = {
        "vol0.08": combined_exposure(close, benchmark=benchmark, target_vol=0.08),
        "vol0.10": combined_exposure(close, benchmark=benchmark, target_vol=0.10),
        "vol0.12": combined_exposure(close, benchmark=benchmark, target_vol=0.12),
        "adaptive": adaptive_combined_exposure(close, benchmark=benchmark),
    }

    # 走步式：每折在训练窗挑最优候选，应用到其后验证窗
    dates = close.index
    start = dates.min()
    chosen_expo = pd.Series(0.0, index=dates)
    fold_rows = []
    cursor = start + pd.DateOffset(months=PERIOD.roll_train_months)
    end = dates.max()
    while cursor <= end:
        is_start = cursor - pd.DateOffset(months=PERIOD.roll_train_months)
        is_mask = (dates >= is_start) & (dates < cursor)
        oos_end = cursor + pd.DateOffset(months=PERIOD.roll_valid_months)
        oos_mask = (dates >= cursor) & (dates < oos_end)
        if oos_mask.sum() == 0:
            break
        # 训练窗挑 Calmar 最优候选
        best, best_c = None, -np.inf
        for name, expo in candidates.items():
            pe = proxy_equity(score[is_mask], close[is_mask], expo[is_mask])
            c = calmar(pe)
            if c > best_c:
                best_c, best = c, name
        chosen_expo[oos_mask] = candidates[best][oos_mask].values
        fold_rows.append({"oos_start": cursor.date(), "chosen": best,
                          "is_calmar": round(best_c, 3)})
        cursor = cursor + pd.DateOffset(months=PERIOD.roll_step_months)

    print("\n--- 各折训练期选定的风险叠加层 ---")
    print(pd.DataFrame(fold_rows).to_string(index=False))

    # 用拼好的(纯样本外)总仓位时间表跑一次回测
    trend = trend_filter(close, ma_window=200)
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    ma60 = close.rolling(60).mean()
    rebal = list(close.index[::PERIOD.rebalance_freq])
    bt = Backtester(
        prices={k: wide[k] for k in
                ["open", "high", "low", "close", "volume", "pre_close"]},
        score=score, trend_pass=trend, vol=vol20, ma_exit=ma60,
        industry=industry, rebalance_dates=rebal, init_capital=1.0e7,
        exposure_schedule=chosen_expo)
    res = bt.run()
    equity = res["equity"]

    # 仅在 OOS 区间统计(第一折验证开始之后)
    oos_start = start + pd.DateOffset(months=PERIOD.roll_train_months)
    oos_eq = equity[equity.index >= oos_start]

    print(f"\n--- 纯样本外绩效 (始于 {oos_start.date()}) ---")
    for k, v in performance_summary(oos_eq, res["trades"], res["turnover"]).items():
        print(f"  {k:20s}: {v}")
    print("\n--- 分年度(OOS) ---")
    print(annual_breakdown(oos_eq).to_string())
    print("\n--- 过拟合诊断(OOS) ---")
    print("  DSR:", deflated_sharpe_ratio(oos_eq.pct_change().dropna(), n_trials=50))

    ic_table = evaluate_factor_library(neut, close, horizon=5)
    outdir = os.path.join(os.path.dirname(__file__), "..", "reports_walkforward")
    paths = generate_report(oos_eq, ic_table, outdir)
    print("\n图表已生成:")
    for p in paths:
        print("  ", os.path.abspath(p))


if __name__ == "__main__":
    main()
