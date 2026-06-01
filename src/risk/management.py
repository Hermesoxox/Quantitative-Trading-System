"""
风险管理：个股层面 + 组合层面 + 流动性。

个股层面
--------
* 止损：建仓后相对成本价下跌 8% 无条件止损（含 T+1 约束：次日才能卖）。
* 移动止盈：浮盈超过 15% 后启动；此后从最高浮盈回撤超过"盈利的 30%"则止盈。
    触发条件:  (peak_gain - cur_gain) > 0.30 * peak_gain  且 peak_gain > 0.15
* 趋势退出：跌破 60 日均线卖出（在引擎中结合信号判定）。

组合层面
--------
* 当日组合亏损 > 5% -> 次日减仓至半仓。
* 滚动周回撤 > 8% -> 清仓并暂停交易 5 个交易日。

流动性
------
* 单只股票当日成交量不超过其当日成交量的 5%，超出部分顺延（分批）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import RISK


@dataclass
class PositionState:
    """单只持仓的状态，用于个股风控判定。"""
    code: str
    entry_price: float          # 成本价
    shares: float
    peak_price: float = 0.0     # 持有期最高价（移动止盈用）

    def update_peak(self, price: float) -> None:
        self.peak_price = max(self.peak_price, price, self.entry_price)

    def gain(self, price: float) -> float:
        return price / self.entry_price - 1

    def peak_gain(self) -> float:
        return self.peak_price / self.entry_price - 1


class RiskManager:
    """组合级风控状态机。"""

    def __init__(self):
        self.halt_until_idx: int | None = None   # 暂停交易截止的 bar 序号
        self.half_position = False               # 是否处于减半仓状态
        self.week_peak_equity = None

    # ---------- 个股层面 ----------
    @staticmethod
    def check_stop_loss(pos: PositionState, price: float) -> bool:
        """相对成本下跌达到止损线。"""
        return pos.gain(price) <= -RISK.stop_loss

    @staticmethod
    def check_trailing_stop(pos: PositionState, price: float) -> bool:
        """移动止盈：已启动（峰值盈利>阈值）且回撤超过峰值盈利的比例。"""
        pg = pos.peak_gain()
        if pg < RISK.trailing_activate:
            return False
        cur = pos.gain(price)
        # 从峰值盈利回吐超过 30%
        return (pg - cur) > RISK.trailing_drawdown_ratio * pg

    # ---------- 组合层面 ----------
    def on_daily_close(self, bar_idx: int, daily_return: float,
                       equity: float, week_window_min: float) -> dict:
        """
        每日收盘后更新组合风控状态，返回动作指令。

        返回 dict:
            {"action": "halt"|"halve"|"none", "target_exposure": float}
        target_exposure: 期望的总仓位（1.0 满仓 / 0.5 半仓 / 0.0 空仓）。
        """
        # 周回撤：当前净值相对近 5 日窗口高点的回撤
        weekly_dd = equity / week_window_min - 1 if week_window_min else 0

        if daily_return <= -RISK.daily_loss_cut:
            self.half_position = True
            return {"action": "halve", "target_exposure": 0.5}

        if weekly_dd <= -RISK.weekly_drawdown_halt:
            self.halt_until_idx = bar_idx + RISK.halt_days
            return {"action": "halt", "target_exposure": 0.0}

        return {"action": "none",
                "target_exposure": 0.5 if self.half_position else 1.0}

    def is_halted(self, bar_idx: int) -> bool:
        return self.halt_until_idx is not None and bar_idx < self.halt_until_idx

    def maybe_resume(self, bar_idx: int) -> None:
        if self.halt_until_idx is not None and bar_idx >= self.halt_until_idx:
            self.halt_until_idx = None
            self.half_position = False

    # ---------- 流动性 ----------
    @staticmethod
    def liquidity_cap_shares(day_volume: float) -> float:
        """当日可成交的最大股数（手数*100），不超过当日成交量的 5%。"""
        return day_volume * RISK.max_pct_of_volume
