"""
端到端演示：数据 -> 因子 -> 中性化 -> 信号 -> 滚动权重 -> 回测 -> 绩效。

默认使用合成数据（无需联网），可一键跑通全流程并打印绩效与分年度表格。
将 `USE_REAL_DATA = True` 并配置 AkShare 股票池即可切换到真实数据。

运行:
    cd Quantitative-Trading-System
    python -m examples.run_backtest
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DEFAULT_FACTOR_WEIGHTS, PERIOD
from src.data.loader import make_synthetic_dataset, load_daily_price, \
    load_money_flow, load_industry_map
from src.factors import compute_all_factors, neutralize_factor
from src.signals import composite_score, trend_filter
from src.backtest import Backtester, performance_summary, annual_breakdown
from src.optimization.rolling import rolling_factor_weights
from src.optimization.rolling import weights_for_date
from analysis.factor_ic import evaluate_factor_library

USE_REAL_DATA = False


def long_to_wide(panel: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """长表(MultiIndex date,code) -> 宽表(date x code)。"""
    return panel[field_name].unstack("code").sort_index()


def build_wide_inputs(ds: dict) -> dict[str, pd.DataFrame]:
    """把各源数据整理成因子库需要的宽表字典。"""
    price, flow, fund = ds["price"], ds["flow"], ds["fund"]
    wide = {
        "open": long_to_wide(price, "open"),
        "high": long_to_wide(price, "high"),
        "low": long_to_wide(price, "low"),
        "close": long_to_wide(price, "close"),
        "volume": long_to_wide(price, "volume"),
        "amount": long_to_wide(price, "amount"),
        "pre_close": long_to_wide(price, "pre_close"),
        "mf_ratio": long_to_wide(flow, "main_net_inflow_ratio"),
    }
    close = wide["close"]

    # 财务数据：从公告日对齐到每个交易日（前向填充，杜绝未来函数）
    def align_fund(col):
        f = fund.reset_index().pivot_table(
            index="announce_date", columns="code", values=col)
        return f.reindex(close.index, method="ffill").reindex(
            columns=close.columns)

    wide["roe"] = align_fund("roe")
    wide["gross_margin"] = align_fund("gross_margin")
    wide["net_profit"] = align_fund("net_profit")
    wide["bps"] = align_fund("bps")
    wide["eps"] = align_fund("eps")
    return wide


def main():
    print("=" * 70)
    print("A股多因子中低频量化系统 — 端到端回测演示")
    print("=" * 70)

    # 1) 数据
    if USE_REAL_DATA:
        raise NotImplementedError("请在此配置 AkShare 真实股票池与拉取逻辑。")
    ds = make_synthetic_dataset(n_stocks=80,
                                start=PERIOD.in_sample_start,
                                end=PERIOD.out_sample_end)
    wide = build_wide_inputs(ds)
    close = wide["close"]
    industry = ds["industry"]["industry"]
    log_cap = ds["industry"]["log_cap"]
    print(f"股票数: {close.shape[1]}, 交易日数: {close.shape[0]}")

    # 2) 因子计算 + 中性化
    raw_factors = compute_all_factors(wide)
    print(f"已计算因子: {list(raw_factors.keys())}")
    neut_factors = {}
    for name, fac in raw_factors.items():
        neut_factors[name] = neutralize_factor(fac, industry, log_cap)

    # 3) 因子有效性（IC）评估
    print("\n--- 因子 IC 评估 (horizon=5日) ---")
    ic_table = evaluate_factor_library(neut_factors, close, horizon=5)
    print(ic_table.to_string())

    # 4) 滚动因子权重（防过拟合：IC加权 + 类上限 + 训练/验证不重叠）
    print("\n--- 滚动因子权重 (3年训练/1年验证) ---")
    weight_schedule = rolling_factor_weights(
        neut_factors, close, list(close.index),
        train_months=PERIOD.roll_train_months,
        valid_months=PERIOD.roll_valid_months,
        step_months=PERIOD.roll_step_months, horizon=5)
    if not weight_schedule.empty:
        print(weight_schedule.round(3).to_string())

    # 5) 合成综合得分（逐日使用对应滚动权重；样本内启动期用默认权重）
    score = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for date in close.index:
        if weight_schedule.empty or date < weight_schedule.index.min():
            w = DEFAULT_FACTOR_WEIGHTS
        else:
            w = weights_for_date(weight_schedule, date)
        row_factors = {n: f.loc[[date]] for n, f in neut_factors.items()
                       if date in f.index}
        if row_factors:
            s = composite_score(row_factors, w)
            score.loc[date] = s.loc[date]

    # 6) 趋势过滤 + 辅助宽表
    trend = trend_filter(close, ma_window=200)
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    ma60 = close.rolling(60).mean()

    # 调仓日：每 rebalance_freq 个交易日
    rebal_dates = list(close.index[::PERIOD.rebalance_freq])

    # 7) 回测
    print("\n--- 运行回测 ---")
    bt = Backtester(
        prices={k: wide[k] for k in
                ["open", "high", "low", "close", "volume", "pre_close"]},
        score=score, trend_pass=trend, vol=vol20, ma_exit=ma60,
        industry=industry, rebalance_dates=rebal_dates,
        init_capital=1.0e7)
    result = bt.run()
    equity = result["equity"]

    # 8) 绩效
    print("\n--- 全样本绩效 ---")
    perf = performance_summary(equity, result["trades"], result["turnover"])
    for k, v in perf.items():
        print(f"  {k:20s}: {v}")

    print("\n--- 分年度绩效 ---")
    print(annual_breakdown(equity).to_string())

    # 样本内 / 样本外切分
    is_eq = equity[equity.index <= PERIOD.in_sample_end]
    oos_eq = equity[equity.index >= PERIOD.out_sample_start]
    if len(is_eq) > 10:
        print("\n样本内绩效:", performance_summary(is_eq))
    if len(oos_eq) > 10:
        print("样本外绩效:", performance_summary(oos_eq))

    # 9) 绩效可视化(与真实数据脚本共用同一套绘图代码)
    try:
        from analysis.plots import generate_report
        outdir = os.path.join(os.path.dirname(__file__), "..", "reports")
        paths = generate_report(equity, ic_table, outdir)
        print("\n图表已生成:")
        for p in paths:
            print("  ", os.path.abspath(p))
    except Exception as e:
        print(f"[绘图跳过] {e}")

    print("\n演示完成。真实数据请运行 examples/run_real_backtest.py。")


if __name__ == "__main__":
    main()
