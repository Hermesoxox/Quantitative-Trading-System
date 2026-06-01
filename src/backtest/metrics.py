"""
绩效评估指标。

输入统一为"日度净值序列" equity（pd.Series, index=date）或日收益序列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series) -> float:
    """几何年化收益。"""
    n = len(equity)
    if n < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    return total ** (TRADING_DAYS / n) - 1


def annualized_vol(equity: pd.Series) -> float:
    return to_returns(equity).std() * np.sqrt(TRADING_DAYS)


def sharpe(equity: pd.Series, rf: float = 0.02) -> float:
    """夏普比率 = (年化收益 - 无风险利率) / 年化波动。"""
    vol = annualized_vol(equity)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (annualized_return(equity) - rf) / vol


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负值）。"""
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return dd.min()


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    return annualized_return(equity) / mdd


def win_rate_and_pnl(trades: pd.DataFrame) -> dict:
    """
    基于"已平仓交易"计算胜率与盈亏比。
    trades 需含列 'pnl_pct'（每笔交易的收益率）。
    """
    if trades is None or trades.empty or "pnl_pct" not in trades:
        return {"win_rate": np.nan, "profit_loss_ratio": np.nan}
    wins = trades[trades["pnl_pct"] > 0]["pnl_pct"]
    losses = trades[trades["pnl_pct"] < 0]["pnl_pct"]
    win_rate = len(wins) / len(trades)
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = abs(losses.mean()) if len(losses) else np.nan
    plr = avg_win / avg_loss if avg_loss and not np.isnan(avg_loss) else np.nan
    return {"win_rate": round(win_rate, 4),
            "profit_loss_ratio": round(plr, 4) if not np.isnan(plr) else np.nan}


def monthly_win_rate(equity: pd.Series) -> float:
    """月胜率：盈利月份占比。"""
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return np.nan
    return round((monthly > 0).mean(), 4)


def performance_summary(equity: pd.Series, trades: pd.DataFrame | None = None,
                        turnover: float | None = None) -> dict:
    """汇总主要绩效指标。"""
    out = {
        "annual_return": round(annualized_return(equity), 4),
        "annual_vol": round(annualized_vol(equity), 4),
        "sharpe": round(sharpe(equity), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "calmar": round(calmar(equity), 4),
        "monthly_win_rate": monthly_win_rate(equity),
    }
    out.update(win_rate_and_pnl(trades))
    if turnover is not None:
        out["annual_turnover"] = round(turnover, 4)
    return out


def annual_breakdown(equity: pd.Series) -> pd.DataFrame:
    """分年度绩效表：年收益、年波动、夏普、当年最大回撤。"""
    rows = {}
    for year, grp in equity.groupby(equity.index.year):
        if len(grp) < 2:
            continue
        rows[year] = {
            "return": round(grp.iloc[-1] / grp.iloc[0] - 1, 4),
            "vol": round(annualized_vol(grp), 4),
            "sharpe": round(sharpe(grp), 4),
            "max_drawdown": round(max_drawdown(grp), 4),
        }
    return pd.DataFrame(rows).T
