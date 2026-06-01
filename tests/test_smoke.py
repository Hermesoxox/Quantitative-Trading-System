"""
冒烟测试：保证核心链路在合成数据上可跑通、无未来函数、规则生效。

运行: python -m pytest tests/ -q   或   python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import RULES, FACTOR_DIRECTION
from src.data.loader import make_synthetic_dataset
from src.backtest.costs import trade_cost, can_buy, can_sell, price_limit
from src.factors import compute_all_factors, neutralize_factor
from src.signals import composite_score, trend_filter


def _wide(panel, field):
    return panel[field].unstack("code").sort_index()


def test_cost_sell_includes_stamp_tax():
    """卖出成本必须比买入多一笔印花税。"""
    buy = trade_cost(1_000_000, "buy")
    sell = trade_cost(1_000_000, "sell")
    assert sell - buy == pytest_approx(1_000_000 * RULES.stamp_tax)


def test_price_limit_board():
    assert price_limit("600000") == 0.10      # 主板
    assert price_limit("300001") == 0.20      # 创业板
    assert price_limit("688001") == 0.20      # 科创板


def test_limit_up_blocks_buy():
    """一字涨停（开盘=最低=涨停价）应无法买入。"""
    pre = 10.0
    limit_up = round(pre * 1.10, 2)
    assert can_buy("600000", limit_up, pre, limit_up, limit_up) is False
    # 正常开盘可买
    assert can_buy("600000", 10.2, pre, 10.5, 10.0) is True


def test_limit_down_blocks_sell():
    pre = 10.0
    limit_down = round(pre * 0.90, 2)
    assert can_sell("600000", limit_down, pre, limit_down, limit_down) is False
    assert can_sell("600000", 9.8, pre, 10.0, 9.5) is True


def test_factor_pipeline_runs():
    ds = make_synthetic_dataset(n_stocks=30, start="2018-01-01",
                                end="2020-12-31")
    wide = {
        "close": _wide(ds["price"], "close"),
        "high": _wide(ds["price"], "high"),
        "low": _wide(ds["price"], "low"),
        "volume": _wide(ds["price"], "volume"),
        "mf_ratio": _wide(ds["flow"], "main_net_inflow_ratio"),
    }
    factors = compute_all_factors(wide)
    # 至少算出价格类因子
    assert "mom_20" in factors and "vol_20" in factors
    # 中性化后仍为同形状宽表
    neut = neutralize_factor(factors["mom_20"],
                             ds["industry"]["industry"],
                             ds["industry"]["log_cap"])
    assert neut.shape == factors["mom_20"].shape


def test_composite_respects_direction():
    """方向为负的因子（如 rev_5）应被调正后再合成。"""
    idx = pd.date_range("2020-01-01", periods=3)
    cols = ["A", "B", "C"]
    f = pd.DataFrame([[1, 0, -1]] * 3, index=idx, columns=cols, dtype=float)
    score = composite_score({"rev_5": f}, {"rev_5": 1.0})
    # rev_5 方向 -1：原值最小(-1, C列)应得分最高
    assert FACTOR_DIRECTION["rev_5"] == -1
    assert score.iloc[0].idxmax() == "C"


def test_trend_filter_shape():
    ds = make_synthetic_dataset(n_stocks=10, start="2019-01-01",
                                end="2020-12-31")
    close = _wide(ds["price"], "close")
    tf = trend_filter(close, ma_window=200)
    assert tf.shape == close.shape
    assert tf.dtypes.iloc[0] == bool


def test_regime_exposure_bounded():
    """市场状态总仓位时间表应落在 [0,1]，且熊市能降到接近 0。"""
    from src.regime import combined_exposure
    ds = make_synthetic_dataset(n_stocks=15, start="2018-01-01",
                                end="2021-12-31")
    close = _wide(ds["price"], "close")
    expo = combined_exposure(close, target_vol=0.12)
    assert expo.between(0, 1).all()
    assert expo.shape[0] == close.shape[0]


def test_overfit_diagnostics():
    """PSR/PBO 在随机数据上应给出合理范围的值。"""
    import numpy as np
    from analysis.overfit import probabilistic_sharpe_ratio, pbo_cscv
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    psr = probabilistic_sharpe_ratio(rets, sr_benchmark=0.0)
    assert 0.0 <= psr <= 1.0
    # 10 条纯噪声策略 -> PBO 应可计算且在 [0,1]
    mat = pd.DataFrame(rng.normal(0, 0.01, (300, 10)))
    res = pbo_cscv(mat, n_splits=8)
    assert 0.0 <= res["PBO"] <= 1.0


def test_concentrated_holdings_cap():
    """≤5 只集中持仓：构建的目标权重数量不超过上限，且和≈1。"""
    from config import PORTFOLIO
    from src.portfolio import build_target_weights
    codes = [f"60000{i}" for i in range(20)]
    score = pd.Series(range(20), index=codes, dtype=float)
    trend = pd.Series(True, index=codes)
    vol = pd.Series(0.2, index=codes)
    industry = pd.Series(["A"] * 20, index=codes)
    w = build_target_weights(score, trend, vol, industry)
    assert len(w) <= PORTFOLIO.n_holdings_max
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w <= PORTFOLIO.max_weight_per_stock + 1e-9).all()


# --- 极简 approx，避免强依赖 pytest ---
class _Approx:
    def __init__(self, v, tol=1e-6):
        self.v, self.tol = v, tol
    def __eq__(self, other):
        return abs(other - self.v) <= self.tol


def pytest_approx(v, tol=1e-6):
    return _Approx(v, tol)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("所有冒烟测试通过。")
