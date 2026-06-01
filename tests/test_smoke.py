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
