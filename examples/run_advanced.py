"""
进阶策略端到端演示(对标 SOTA) —— 集中持仓 ≤5 只。

相比基础版(examples/run_backtest.py)的升级：
  1. 非线性因子合成：LightGBM 梯度提升 + 净化禁运滚动训练(防泄露)。
  2. 市场状态识别 + 波动率目标叠加层：动态总仓位，集中持仓的回撤防线。
  3. 集中组合：最多持有 5 只，单票上限 30%。
  4. 过拟合诊断：紧缩夏普 DSR + 过拟合概率 PBO，给出可信度判定。
  5. 全套绩效可视化(reports/)。

运行:
    python -m examples.run_advanced                 # 合成数据
    python -m examples.run_advanced --real --n 60    # 真实数据(东财)+沪深300基准
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# 屏蔽 LightGBM/sklearn/常量截面相关的无害告警，保持输出整洁
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DEFAULT_FACTOR_WEIGHTS, PERIOD, OVERLAY, PORTFOLIO
from src.data.loader import make_synthetic_dataset
from src.factors import compute_all_factors, neutralize_factor
from src.signals import composite_score, trend_filter, smooth_score
from src.ml import rolling_ml_scores
from src.regime import combined_exposure
from src.backtest import Backtester, performance_summary, annual_breakdown
from analysis.factor_ic import evaluate_factor_library
from analysis.overfit import deflated_sharpe_ratio, pbo_cscv
from analysis.plots import generate_report


def long_to_wide(panel, field):
    return panel[field].unstack("code").sort_index()


def candidate_strategy_returns(neut_factors, close, top_k=5):
    """为 PBO 构造候选策略收益矩阵：每个因子 -> 次日 top-k 等权篮子收益。"""
    nxt = close.pct_change().shift(-1)
    out = {}
    for name, fac in neut_factors.items():
        rank = fac.rank(axis=1, ascending=False)
        mask = (rank <= top_k).astype(float)
        cnt = mask.sum(axis=1).replace(0, np.nan)
        out[name] = (nxt * mask).sum(axis=1) / cnt
    return pd.DataFrame(out).dropna(how="all")


def load_synthetic():
    ds = make_synthetic_dataset(n_stocks=80, start=PERIOD.in_sample_start,
                                end=PERIOD.out_sample_end)
    wide = {f: long_to_wide(ds["price"], f) for f in
            ["open", "high", "low", "close", "volume", "amount", "pre_close"]}
    wide["mf_ratio"] = long_to_wide(ds["flow"], "main_net_inflow_ratio")
    close = wide["close"]
    industry = ds["industry"]["industry"]
    log_cap = ds["industry"]["log_cap"]
    fund = ds["fund"]
    def align(col):
        f = fund.reset_index().pivot_table(index="announce_date",
                                           columns="code", values=col)
        return f.reindex(close.index, method="ffill").reindex(columns=close.columns)
    for c in ["roe", "gross_margin", "net_profit", "bps", "eps"]:
        wide[c] = align(c)
    return wide, industry, log_cap, None


def load_real(n, start, end, codes_arg):
    """真实数据(东财直连)：行情全量；资金流/财务缺失则相关因子自动跳过。"""
    from src.data.eastmoney import (fetch_universe_prices, fetch_csi300_codes,
                                    fetch_kline)
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data_cache")
    if codes_arg:
        codes = [c.strip() for c in codes_arg.split(",") if c.strip()]
    else:
        codes = fetch_csi300_codes()
    if n > 0:
        codes = codes[:n]
    print(f"真实股票池: {len(codes)} 只  {start}~{end}")
    cache = os.path.join(cache_dir, f"adv_real_{start}_{end}_{len(codes)}.parquet")
    price = fetch_universe_prices(codes, start, end, adjust="qfq",
                                  cache_path=cache)
    wide = {f: long_to_wide(price, f) for f in
            ["open", "high", "low", "close", "volume", "amount", "pre_close"]}
    close = wide["close"]
    # 规模用对数成交额近似，行业占位(真实落地接行业接口)
    log_cap = np.log(wide["amount"].mean()).rename("log_cap")
    industry = pd.Series("ALL", index=close.columns, name="industry")

    # 财务数据(东财数据中心，沙箱/云端亦可达) -> 激活价值/质量因子。
    # 按公告日 ffill 对齐到交易日，杜绝未来函数；接口不可达则自动跳过(仅用量价因子)。
    from src.data.cn_fundamental import fetch_fundamentals
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data_cache")
    fund = fetch_fundamentals(list(close.columns),
                              cache_path=os.path.join(cache_dir, f"fund_{len(close.columns)}.pkl"))
    if fund is not None and not fund.empty:
        def align(col):
            f = fund.reset_index().pivot_table(index="announce_date",
                                               columns="code", values=col)
            return f.reindex(close.index, method="ffill").reindex(columns=close.columns)
        for c in ["roe", "gross_margin", "net_profit", "bps", "eps"]:
            wide[c] = align(c)
    # 沪深300基准用于 regime 层；若基准历史覆盖不足(<90%)则置空，
    # 改用等权全样本代理(覆盖完整、更稳健)，避免基准缺口扭曲择时。
    bench_df = fetch_kline("000300", start, end, adjust="qfq")
    benchmark = None
    if bench_df is not None:
        b = bench_df.set_index("date")["close"].reindex(close.index)
        if b.notna().mean() >= 0.90:
            benchmark = b
        else:
            print(f"[基准] 沪深300覆盖率仅 {b.notna().mean():.0%}，"
                  f"改用等权市场代理。")
    return wide, industry, log_cap, benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用真实东财数据")
    ap.add_argument("--n", type=int, default=0, help="股票数(0=全部/合成80)")
    ap.add_argument("--codes", type=str, default="")
    ap.add_argument("--start", default=PERIOD.in_sample_start)
    ap.add_argument("--end", default=PERIOD.out_sample_end)
    args = ap.parse_args()

    print("=" * 72)
    print("进阶多因子集中持仓策略(≤5只) — ML合成 + 状态识别 + 波动率目标 + 回撤守卫")
    print("=" * 72)

    if args.real:
        wide, industry, log_cap, benchmark = load_real(
            args.n, args.start, args.end, args.codes)
    else:
        wide, industry, log_cap, benchmark = load_synthetic()
    close = wide["close"]

    # 因子 + 中性化
    raw = compute_all_factors(wide)
    neut = {n: neutralize_factor(f, industry, log_cap) for n, f in raw.items()}
    print(f"因子数: {len(neut)}  后端: ", end="")

    # ---- 1) 因子合成：ML 或 线性 ----
    if OVERLAY.use_ml_combiner:
        from src.ml.combiner import _BACKEND
        print(f"ML({_BACKEND})")
        score, imp = rolling_ml_scores(
            neut, close, horizon=OVERLAY.ml_horizon,
            retrain_freq=OVERLAY.ml_retrain_freq,
            embargo_days=OVERLAY.ml_embargo_days)
        if imp is not None:
            print("\n最近一次模型特征重要度(Top8):")
            print(imp.head(8).to_string())
    else:
        print("线性加权")
        score = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        avail = {k: v for k, v in DEFAULT_FACTOR_WEIGHTS.items() if k in neut}
        for date in close.index:
            row = {n: f.loc[[date]] for n, f in neut.items() if date in f.index}
            if row:
                score.loc[date] = composite_score(row, avail).loc[date]

    score = smooth_score(score)

    # ---- 2) 市场状态 + 波动率目标 -> 总仓位时间表 ----
    exposure = None
    if OVERLAY.use_regime or OVERLAY.use_vol_target:
        exposure = combined_exposure(
            close, benchmark=benchmark, ma_window=OVERLAY.regime_ma,
            target_vol=OVERLAY.target_vol, vol_window=OVERLAY.vol_window,
            smooth=OVERLAY.exposure_smooth)
        print(f"\n总仓位叠加层: 均值={exposure.mean():.2f}, "
              f"空仓天数占比={(exposure < 0.1).mean():.1%}")

    # ---- 3) 回测(集中持仓 ≤5) ----
    trend = trend_filter(close, ma_window=200)
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    ma60 = close.rolling(60).mean()
    rebal = list(close.index[::PERIOD.rebalance_freq])

    bt = Backtester(
        prices={k: wide[k] for k in
                ["open", "high", "low", "close", "volume", "pre_close"]},
        score=score, trend_pass=trend, vol=vol20, ma_exit=ma60,
        industry=industry, rebalance_dates=rebal, init_capital=1.0e7,
        exposure_schedule=exposure)
    res = bt.run()
    equity = res["equity"]

    print(f"\n--- 全样本绩效 (最多持有 {PORTFOLIO.n_holdings_max} 只) ---")
    for k, v in performance_summary(equity, res["trades"], res["turnover"]).items():
        print(f"  {k:20s}: {v}")
    print("\n--- 分年度绩效 ---")
    print(annual_breakdown(equity).to_string())

    # ---- 4) 过拟合诊断 ----
    print("\n--- 过拟合诊断 ---")
    rets = equity.pct_change().dropna()
    dsr = deflated_sharpe_ratio(rets, n_trials=50)
    print(f"  紧缩夏普 DSR: {dsr}")
    cand = candidate_strategy_returns(neut, close, top_k=5)
    pbo = pbo_cscv(cand, n_splits=10)
    print(f"  过拟合概率 PBO: {pbo}")
    verdict = []
    if dsr.get("DSR") and dsr["DSR"] >= 0.95:
        verdict.append("DSR≥0.95(可信)")
    else:
        verdict.append("DSR偏低(谨慎)")
    if pbo.get("PBO") is not None and pbo["PBO"] <= 0.5:
        verdict.append("PBO≤0.5(过拟合风险可控)")
    else:
        verdict.append("PBO偏高(过拟合风险)")
    print("  判定:", "; ".join(verdict))

    # ---- 5) 可视化 ----
    ic_table = evaluate_factor_library(neut, close, horizon=5)
    outdir = os.path.join(os.path.dirname(__file__), "..", "reports_advanced")
    paths = generate_report(equity, ic_table, outdir)
    print("\n图表已生成:")
    for p in paths:
        print("  ", os.path.abspath(p))


if __name__ == "__main__":
    main()
