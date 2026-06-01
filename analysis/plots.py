"""
绩效可视化(matplotlib)。生成四类图：
  1. 净值曲线 + 回撤面积(双子图)
  2. 月度收益热力图
  3. 滚动夏普(126日)
  4. 因子 IC 柱状图(按 ICIR 排序)

全部用 Agg 后端，无需显示环境，直接存 PNG，适合服务器/CI。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["axes.unicode_minus"] = False


def plot_equity_drawdown(equity: pd.Series, path: str,
                         benchmark: pd.Series | None = None) -> str:
    """净值曲线 + 回撤面积。"""
    eq = equity / equity.iloc[0]
    dd = eq / eq.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(eq.index, eq.values, lw=1.6, color="#1f6feb", label="Strategy")
    if benchmark is not None and len(benchmark) > 1:
        bm = benchmark.reindex(eq.index).ffill()
        bm = bm / bm.iloc[0]
        ax1.plot(bm.index, bm.values, lw=1.2, color="#999999",
                 label="Benchmark")
    ax1.set_ylabel("Net Value (norm.)")
    ax1.set_title("Equity Curve & Drawdown")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(dd.index, dd.values, 0, color="#d1242f", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_monthly_heatmap(equity: pd.Series, path: str) -> str:
    """月度收益热力图(年 x 月)。"""
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return path
    tbl = monthly.to_frame("ret")
    tbl["year"] = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot_table(index="year", columns="month", values="ret")

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(pivot) + 2))
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{m}月" if False else f"M{m}" for m in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_title("Monthly Returns (%)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rolling_sharpe(equity: pd.Series, path: str, window: int = 126) -> str:
    """滚动夏普(默认半年窗)。"""
    ret = equity.pct_change().dropna()
    roll = (ret.rolling(window).mean() / ret.rolling(window).std()
            * np.sqrt(252))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll.index, roll.values, color="#8957e5", lw=1.3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="green", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(f"Rolling Sharpe ({window}d)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_factor_ic(ic_table: pd.DataFrame, path: str) -> str:
    """因子 ICIR 柱状图(绿正红负)。ic_table 来自 evaluate_factor_library。"""
    if "ICIR" not in ic_table.columns:
        return path
    s = ic_table["ICIR"].sort_values()
    colors = ["#d1242f" if v < 0 else "#2da44e" for v in s.values]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(s) + 2))
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Factor ICIR (in-sample)")
    ax.set_xlabel("ICIR")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_report(equity: pd.Series, ic_table: pd.DataFrame | None,
                    outdir: str, benchmark: pd.Series | None = None) -> list[str]:
    """一次性生成全部图表，返回文件路径列表。"""
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    paths.append(plot_equity_drawdown(equity, os.path.join(
        outdir, "equity_drawdown.png"), benchmark))
    paths.append(plot_monthly_heatmap(equity, os.path.join(
        outdir, "monthly_heatmap.png")))
    paths.append(plot_rolling_sharpe(equity, os.path.join(
        outdir, "rolling_sharpe.png")))
    if ic_table is not None and not ic_table.empty:
        paths.append(plot_factor_ic(ic_table, os.path.join(
            outdir, "factor_icir.png")))
    return paths
