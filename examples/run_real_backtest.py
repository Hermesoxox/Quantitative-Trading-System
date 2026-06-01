"""
真实数据端到端回测(东方财富直连) + 绩效可视化。

用法
----
    python -m examples.run_real_backtest                # 默认沪深300, 2016-2025
    python -m examples.run_real_backtest --n 80         # 限制股票数(加速)
    python -m examples.run_real_backtest --codes 600519,000858,601318

说明
----
* 行情(前复权)经东财接口实时抓取并缓存到 data_cache/。资金流与财务在真实
  落地时同样可接东财/Tushare；本脚本若拿不到资金流/财务，则相应因子自动跳过
  (compute_all_factors 已做容错)，不影响价格/动量/波动率/反转类因子与回测。
* 若运行环境网络策略限制了数据API主机(返回 502/503/空)，抓取会失败并给出
  明确提示，请在本机或放宽网络策略的环境运行——代码与真实接口一致，无需改动。
* 基准用沪深300指数(sh000300)，无法获取时退化为等权全样本。

输出
----
  控制台: 因子IC表、全样本/分年度/样本内外绩效
  reports/: equity_drawdown.png, monthly_heatmap.png, rolling_sharpe.png,
            factor_icir.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DEFAULT_FACTOR_WEIGHTS, PERIOD
from src.data.eastmoney import (fetch_universe_prices, fetch_csi300_codes,
                                fetch_kline)
from src.factors import compute_all_factors, neutralize_factor
from src.signals import composite_score, trend_filter, smooth_score
from src.backtest import Backtester, performance_summary, annual_breakdown
from src.optimization.rolling import rolling_factor_weights, weights_for_date
from analysis.factor_ic import evaluate_factor_library
from analysis.plots import generate_report

CACHE = os.path.join(os.path.dirname(__file__), "..", "data_cache")


def long_to_wide(panel, field):
    return panel[field].unstack("code").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="限制股票数(0=全部)")
    ap.add_argument("--codes", type=str, default="", help="逗号分隔的代码")
    ap.add_argument("--start", default=PERIOD.in_sample_start)
    ap.add_argument("--end", default=PERIOD.out_sample_end)
    args = ap.parse_args()

    # 1) 股票池
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        codes = fetch_csi300_codes()
    if args.n > 0:
        codes = codes[:args.n]
    print(f"股票池: {len(codes)} 只  区间: {args.start} ~ {args.end}")

    # 2) 抓取行情(前复权) + 缓存
    cache_path = os.path.join(CACHE, f"real_price_{args.start}_{args.end}.parquet")
    price = fetch_universe_prices(codes, args.start, args.end,
                                  adjust="qfq", cache_path=cache_path)
    print(f"行情抓取完成: {price.shape[0]} 行")

    wide = {f: long_to_wide(price, f) for f in
            ["open", "high", "low", "close", "volume", "amount", "pre_close"]}
    close = wide["close"]
    # 行业/市值: 真实落地应接行业接口；此处用对数成交额近似规模，行业占位
    log_cap = np.log(wide["amount"].mean()).rename("log_cap")
    industry = pd.Series("ALL", index=close.columns, name="industry")

    # 3) 因子(价格类全量；资金流/财务缺失则自动跳过)
    raw_factors = compute_all_factors(wide)
    print(f"已计算因子: {list(raw_factors.keys())}")
    neut = {n: neutralize_factor(f, industry, log_cap)
            for n, f in raw_factors.items()}

    # 4) IC 评估
    ic_table = evaluate_factor_library(neut, close, horizon=5)
    print("\n--- 因子 IC 评估 ---")
    print(ic_table.to_string())

    # 5) 滚动权重
    wsched = rolling_factor_weights(
        neut, close, list(close.index),
        train_months=PERIOD.roll_train_months,
        valid_months=PERIOD.roll_valid_months,
        step_months=PERIOD.roll_step_months, horizon=5)

    # 6) 综合得分
    score = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    avail = {k: v for k, v in DEFAULT_FACTOR_WEIGHTS.items() if k in neut}
    for date in close.index:
        if wsched.empty or date < wsched.index.min():
            w = avail
        else:
            w = weights_for_date(wsched, date)
        row = {n: f.loc[[date]] for n, f in neut.items() if date in f.index}
        if row:
            score.loc[date] = composite_score(row, w).loc[date]

    # 6b) 得分平滑降换手
    score = smooth_score(score)

    # 7) 辅助宽表 + 回测
    trend = trend_filter(close, ma_window=200)
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    ma60 = close.rolling(60).mean()
    rebal = list(close.index[::PERIOD.rebalance_freq])

    bt = Backtester(
        prices={k: wide[k] for k in
                ["open", "high", "low", "close", "volume", "pre_close"]},
        score=score, trend_pass=trend, vol=vol20, ma_exit=ma60,
        industry=industry, rebalance_dates=rebal, init_capital=1.0e7)
    res = bt.run()
    equity = res["equity"]

    print("\n--- 全样本绩效 ---")
    for k, v in performance_summary(equity, res["trades"], res["turnover"]).items():
        print(f"  {k:20s}: {v}")
    print("\n--- 分年度绩效 ---")
    print(annual_breakdown(equity).to_string())

    is_eq = equity[equity.index <= PERIOD.in_sample_end]
    oos_eq = equity[equity.index >= PERIOD.out_sample_start]
    if len(is_eq) > 10:
        print("\n样本内:", performance_summary(is_eq))
    if len(oos_eq) > 10:
        print("样本外:", performance_summary(oos_eq))

    # 8) 基准 + 可视化
    benchmark = None
    bench_df = fetch_kline("000300", args.start, args.end, adjust="qfq")
    if bench_df is not None:
        benchmark = bench_df.set_index("date")["close"]

    outdir = os.path.join(os.path.dirname(__file__), "..", "reports")
    paths = generate_report(equity, ic_table, outdir, benchmark)
    print("\n图表已生成:")
    for p in paths:
        print("  ", os.path.abspath(p))


if __name__ == "__main__":
    main()
