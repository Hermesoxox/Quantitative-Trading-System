"""
过拟合量化诊断(对标 SOTA：Bailey & López de Prado)。

回测做得越多、调参越狠，"最好"的那条曲线越可能是运气。两个工具量化这种风险：

1) 概率夏普比率 PSR / 紧缩夏普比率 DSR
   - PSR(SR*) = 给定基准 SR*，观测夏普显著大于它的概率，已修正收益的偏度/峰度。
   - DSR = 把 SR* 设为"做了 N 次独立试验后期望出现的最大夏普"，从而扣除
     "多重检验"带来的虚高。DSR 越接近 1 越可信，<0.95 应高度警惕。

2) 过拟合概率 PBO (Probability of Backtest Overfitting, via CSCV)
   - 把多条候选策略的收益矩阵切成 S 块，组合出训练/测试对；在训练集挑最优，
     看它在测试集的相对排名。若经常"训练最优、测试垫底"，说明在挑噪声。
   - PBO = 训练最优者在测试集逻辑收益(logit of rank)为负的频率。>0.5 即过拟合严重。
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm


def _sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    return returns.mean() / sd * np.sqrt(252) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0
                               ) -> float:
    """
    PSR：观测夏普显著超过 sr_benchmark(年化)的概率，修正偏度峰度。
    返回 0~1 概率。
    """
    r = returns.dropna().values
    n = len(r)
    if n < 30:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    sr = r.mean() / sd                      # 非年化(每期)夏普
    sr_bench = sr_benchmark / np.sqrt(252)  # 年化基准转每期
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurt() + 3          # pandas 给的是超额峰度
    num = (sr - sr_bench) * np.sqrt(n - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(num / den)) if den > 0 else np.nan


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          sr_trials_std: float | None = None) -> dict:
    """
    DSR：扣除多重检验后的夏普可信度。

    n_trials      : 你大致尝试过的策略/参数组合数(诚实估计)。
    sr_trials_std : 各试验夏普的标准差(每期口径)。缺省用一个保守近似。
    """
    r = returns.dropna().values
    n = len(r)
    if n < 30:
        return {"DSR": np.nan, "SR_benchmark_ann": np.nan}
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    # 期望最大夏普(多重检验阈值)，Bailey-López de Prado 近似
    if sr_trials_std is None:
        sr_trials_std = abs(sr) * 0.5 + 1e-6   # 保守占位
    gamma = 0.5772156649  # Euler-Mascheroni
    e1 = norm.ppf(1 - 1.0 / n_trials)
    e2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = sr_trials_std * ((1 - gamma) * e1 + gamma * e2)
    dsr = probabilistic_sharpe_ratio(returns, sr0 * np.sqrt(252))
    return {"DSR": round(dsr, 4) if dsr == dsr else np.nan,
            "SR_benchmark_ann": round(sr0 * np.sqrt(252), 4),
            "observed_SR_ann": round(sr * np.sqrt(252), 4),
            "n_trials": n_trials}


def pbo_cscv(perf_matrix: pd.DataFrame, n_splits: int = 10) -> dict:
    """
    用 CSCV 估计过拟合概率 PBO。

    perf_matrix : index=时间, columns=候选策略, 值=每期收益。
    返回 {"PBO": float, "n_combinations": int}。
    """
    # 先剔除覆盖过低的候选列(如财务因子早期无数据)，避免整体行被 dropna 清空
    M = perf_matrix.dropna(axis=1, thresh=int(0.6 * len(perf_matrix)))
    M = M.dropna(how="any")
    T, N = M.shape
    if N < 2 or T < n_splits * 2:
        return {"PBO": np.nan, "n_combinations": 0}
    # 切成 n_splits 个等长时间块
    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits = []
    for train_combo in itertools.combinations(range(n_splits), half):
        test_combo = [b for b in range(n_splits) if b not in train_combo]
        tr_idx = np.concatenate([blocks[b] for b in train_combo])
        te_idx = np.concatenate([blocks[b] for b in test_combo])
        tr_sr = M.iloc[tr_idx].apply(lambda c: _sharpe(c.values))
        te_sr = M.iloc[te_idx].apply(lambda c: _sharpe(c.values))
        best = tr_sr.idxmax()                       # 训练集最优策略
        # 它在测试集的相对排名(0~1)
        rank = te_sr.rank(pct=True)[best]
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))    # logit
    logits = np.array(logits)
    pbo = float((logits < 0).mean())                # 测试集排名落后中位的频率
    return {"PBO": round(pbo, 4), "n_combinations": len(logits)}
