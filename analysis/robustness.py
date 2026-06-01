"""
稳健性检验：关键参数扰动 ±20%，观察绩效是否仍然有效。

逻辑：一个真正捕捉到市场规律的策略，对参数应不敏感——参数在合理邻域内
变动，年化/夏普/回撤应平滑变化，不应出现"悬崖"。若某参数稍变策略就失效，
基本可判定为过拟合。
"""

from __future__ import annotations

import copy
import itertools

import pandas as pd


def perturb_grid(base_params: dict, pct: float = 0.20,
                 keys: list[str] | None = None) -> list[dict]:
    """
    对指定数值参数生成 {-pct, 0, +pct} 的扰动组合。
    返回参数字典列表（含基准）。为控制组合爆炸，逐一单参数扰动（OAT）。
    """
    keys = keys or [k for k, v in base_params.items()
                    if isinstance(v, (int, float))]
    grid = [dict(base_params)]  # 基准
    for k in keys:
        for f in (1 - pct, 1 + pct):
            p = dict(base_params)
            v = base_params[k]
            p[k] = type(v)(v * f) if not isinstance(v, bool) else v
            p["_perturbed"] = f"{k}*{f:.2f}"
            grid.append(p)
    return grid


def run_robustness(run_backtest_fn, base_params: dict,
                   pct: float = 0.20, keys: list[str] | None = None
                   ) -> pd.DataFrame:
    """
    对每组扰动参数跑一次回测，汇总绩效。

    run_backtest_fn(params) -> dict(含 annual_return/sharpe/max_drawdown)
    """
    results = []
    for params in perturb_grid(base_params, pct, keys):
        tag = params.pop("_perturbed", "baseline")
        perf = run_backtest_fn(params)
        perf["scenario"] = tag
        results.append(perf)
    df = pd.DataFrame(results).set_index("scenario")
    return df


def robustness_verdict(df: pd.DataFrame,
                       sharpe_floor: float = 0.8) -> dict:
    """
    给出稳健性判定：所有扰动场景的夏普是否都在地板之上，
    年化收益的离散度（变异系数）是否可控。
    """
    sharpe = df["sharpe"]
    ann = df["annual_return"]
    return {
        "min_sharpe": round(sharpe.min(), 3),
        "all_above_floor": bool((sharpe >= sharpe_floor).all()),
        "annual_return_cv": round(ann.std() / abs(ann.mean()), 3)
        if ann.mean() != 0 else None,
        "robust": bool((sharpe >= sharpe_floor).all()
                       and sharpe.min() > 0),
    }
