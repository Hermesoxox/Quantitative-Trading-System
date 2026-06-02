"""
真实数据上的"边测边优化"：在不过拟合的前提下，用回撤预算换取更高收益。

方法
----
* 选股得分(ML合成)只算一次——它与总仓位无关。
* 只扫描"总仓位叠加层"的目标波动 target_vol（少量、有经济含义的网格），
  因为基线最大回撤(-16%)明显小于 20% 预算，说明有风险预算可用于提升收益。
* 以 Calmar(年化/最大回撤) 为主、并强制"最大回撤 < 20%"为硬约束来挑选，
  而不是单纯追年化——避免把回撤预算一次性赌光。

注意：这是"在风险预算内调风险敞口"，参数少、逻辑清晰；不是对选股因子做参数寻优
(那才是过拟合高发区)。最终仍应以 run_walkforward 的纯样本外结果为准。
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import OVERLAY, RISK
from src.factors import compute_all_factors, neutralize_factor
from src.signals import trend_filter, smooth_score
from src.ml import rolling_ml_scores
from src.regime import combined_exposure
from src.backtest import Backtester, performance_summary
from examples.run_advanced import load_real, long_to_wide


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    start, end = "2016-01-01", "2025-12-31"
    print(f"真实数据优化扫描  股票数={n}  {start}~{end}")

    wide, industry, log_cap, benchmark = load_real(n, start, end, "")
    close = wide["close"]

    # 因子 + 中性化 + ML 得分（只算一次）
    raw = compute_all_factors(wide)
    neut = {k: neutralize_factor(v, industry, log_cap) for k, v in raw.items()}
    score, _ = rolling_ml_scores(neut, close, horizon=OVERLAY.ml_horizon,
                                 retrain_freq=OVERLAY.ml_retrain_freq,
                                 embargo_days=OVERLAY.ml_embargo_days)
    score = smooth_score(score)

    trend = trend_filter(close, ma_window=200)
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    ma60 = close.rolling(60).mean()
    rebal = list(close.index[::10])
    prices = {k: wide[k] for k in
              ["open", "high", "low", "close", "volume", "pre_close"]}

    rows = []
    for tv in [0.10, 0.13, 0.16, 0.20]:
        expo = combined_exposure(close, benchmark=benchmark,
                                 ma_window=OVERLAY.regime_ma, target_vol=tv,
                                 vol_window=OVERLAY.vol_window,
                                 smooth=OVERLAY.exposure_smooth)
        bt = Backtester(prices=prices, score=score, trend_pass=trend,
                        vol=vol20, ma_exit=ma60, industry=industry,
                        rebalance_dates=rebal, init_capital=1e7,
                        exposure_schedule=expo)
        res = bt.run()
        perf = performance_summary(res["equity"], res["trades"], res["turnover"])
        perf["target_vol"] = tv
        perf["avg_exposure"] = round(float(expo.mean()), 2)
        rows.append(perf)

    df = pd.DataFrame(rows).set_index("target_vol")
    cols = ["annual_return", "max_drawdown", "sharpe", "calmar",
            "annual_turnover", "avg_exposure"]
    print("\n=== target_vol 扫描结果 (真实数据) ===")
    print(df[cols].to_string())

    # 选择：最大回撤<20% 的前提下 Calmar 最高
    ok = df[df["max_drawdown"] > -0.20]
    if not ok.empty:
        best = ok["calmar"].idxmax()
        print(f"\n推荐 target_vol = {best}  "
              f"(满足回撤<20%且Calmar最优; 年化={ok.loc[best,'annual_return']:.1%}, "
              f"回撤={ok.loc[best,'max_drawdown']:.1%})")
    else:
        print("\n所有档位回撤均超 20%，应保持最保守档并检查策略。")


if __name__ == "__main__":
    main()
