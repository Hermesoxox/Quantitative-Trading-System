"""
事件驱动日度回测引擎，严格遵守 A股交易规则。

时序约定（杜绝未来函数 / 满足 T+1）
-----------------------------------
* 第 t 日"收盘后"用截至 t 日的数据生成调仓目标与风控卖出指令。
* 这些指令在第 t+1 日"开盘"以开盘价 * (1 ± 滑点) 撮合。
* 因此当日买入的股票当日不可卖（T+1 自动满足，卖出只针对此前持仓）。
* 涨停（开盘一字板）无法买入，跌停（开盘一字板）无法卖出 -> 顺延。
* 单笔成交量受流动性约束（<= 当日成交量 5%），超出顺延到后续交易日。

净值 = 现金 + Σ(持股数 * 收盘价)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import PORTFOLIO, RULES
from src.risk import RiskManager, PositionState
from .costs import trade_cost, can_buy, can_sell


@dataclass
class Order:
    code: str
    side: str          # "buy" / "sell"
    target_value: float  # 目标成交金额（买）或目标减仓金额（卖，None=清仓）
    reason: str = ""


class Backtester:
    def __init__(
        self,
        prices: dict[str, pd.DataFrame],   # open/high/low/close/volume/pre_close 宽表
        score: pd.DataFrame,               # 综合得分宽表 (date x code)
        trend_pass: pd.DataFrame,          # 趋势过滤布尔宽表
        vol: pd.DataFrame,                 # 20日波动率宽表（风险平价用）
        ma_exit: pd.DataFrame,             # 60日均线宽表（跌破退出）
        industry: pd.Series,               # code -> 行业
        rebalance_dates: list[pd.Timestamp],
        init_capital: float = 1.0e7,
        build_weights_fn=None,             # 注入组合构建函数，便于解耦/测试
    ):
        self.px = prices
        self.score = score
        self.trend = trend_pass
        self.vol = vol
        self.ma_exit = ma_exit
        self.industry = industry
        self.rebal = set(pd.to_datetime(rebalance_dates))
        self.init_capital = init_capital
        self.dates = list(prices["close"].index)

        if build_weights_fn is None:
            from src.portfolio import build_target_weights
            build_weights_fn = build_target_weights
        self.build_weights = build_weights_fn

        # 状态
        self.cash = init_capital
        self.positions: dict[str, PositionState] = {}
        self.risk = RiskManager()
        self.pending: list[Order] = []

        # 记录
        self.equity_curve: dict[pd.Timestamp, float] = {}
        self.trade_log: list[dict] = []
        self.turnover_traded = 0.0   # 累计成交额，用于换手率

    # ------------------------------------------------------------------
    def _price(self, field_name: str, date, code):
        try:
            v = self.px[field_name].at[date, code]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    def _portfolio_value(self, date) -> float:
        val = self.cash
        for code, pos in self.positions.items():
            px = self._price("close", date, code)
            if px:
                val += pos.shares * px
        return val

    # ------------------------------------------------------------------
    def _execute_open(self, date):
        """在开盘以开盘价撮合 pending 订单（先卖后买，释放现金）。"""
        carried = []
        # 先卖
        for o in [x for x in self.pending if x.side == "sell"]:
            self._fill_sell(date, o, carried)
        # 后买
        for o in [x for x in self.pending if x.side == "buy"]:
            self._fill_buy(date, o, carried)
        self.pending = carried  # 未成交（涨跌停/流动性）顺延

    def _fill_sell(self, date, o: Order, carried: list):
        pos = self.positions.get(o.code)
        if pos is None or pos.shares <= 0:
            return
        op = self._price("open", date, o.code)
        pre = self._price("pre_close", date, o.code)
        hi = self._price("high", date, o.code)
        lo = self._price("low", date, o.code)
        if op is None:
            carried.append(o); return
        if not can_sell(o.code, op, pre, hi, lo):
            carried.append(o); return  # 一字跌停卖不出，顺延

        # 流动性约束
        day_vol = self._price("volume", date, o.code) or 0
        max_shares = self.risk.liquidity_cap_shares(day_vol)
        sell_shares = pos.shares if o.target_value is None \
            else min(pos.shares, o.target_value / op)
        sell_shares = min(sell_shares, max_shares) if max_shares > 0 else 0
        if sell_shares <= 0:
            carried.append(o); return

        fill_price = op * (1 - RULES.slippage)   # 卖出滑点向下
        amount = sell_shares * fill_price
        cost = trade_cost(amount, "sell")
        self.cash += amount - cost
        self.turnover_traded += amount

        pnl_pct = fill_price / pos.entry_price - 1
        self.trade_log.append({
            "date": date, "code": o.code, "side": "sell",
            "price": fill_price, "shares": sell_shares,
            "pnl_pct": pnl_pct, "reason": o.reason,
        })
        pos.shares -= sell_shares
        if pos.shares <= 1e-6:
            del self.positions[o.code]
        # 未卖完的剩余量顺延
        if o.target_value is None and pos.code in self.positions:
            carried.append(Order(o.code, "sell", None, o.reason))

    def _fill_buy(self, date, o: Order, carried: list):
        op = self._price("open", date, o.code)
        pre = self._price("pre_close", date, o.code)
        hi = self._price("high", date, o.code)
        lo = self._price("low", date, o.code)
        if op is None:
            carried.append(o); return
        if not can_buy(o.code, op, pre, hi, lo):
            carried.append(o); return  # 一字涨停买不到，顺延

        fill_price = op * (1 + RULES.slippage)   # 买入滑点向上
        budget = min(o.target_value, self.cash * 0.99)
        if budget <= 0:
            return
        target_shares = budget / fill_price

        # 流动性约束
        day_vol = self._price("volume", date, o.code) or 0
        max_shares = self.risk.liquidity_cap_shares(day_vol)
        shares = min(target_shares, max_shares) if max_shares > 0 else 0
        # A股按手（100股）成交
        shares = np.floor(shares / 100) * 100
        if shares <= 0:
            return

        amount = shares * fill_price
        cost = trade_cost(amount, "buy")
        if amount + cost > self.cash:
            shares = np.floor((self.cash / (1 + 0.001)) / fill_price / 100) * 100
            if shares <= 0:
                return
            amount = shares * fill_price
            cost = trade_cost(amount, "buy")
        self.cash -= (amount + cost)
        self.turnover_traded += amount

        if o.code in self.positions:
            pos = self.positions[o.code]
            total = pos.shares + shares
            pos.entry_price = (pos.entry_price * pos.shares +
                               fill_price * shares) / total
            pos.shares = total
        else:
            self.positions[o.code] = PositionState(
                code=o.code, entry_price=fill_price, shares=shares,
                peak_price=fill_price)
        self.trade_log.append({
            "date": date, "code": o.code, "side": "buy",
            "price": fill_price, "shares": shares, "reason": o.reason,
        })
        # 未买够的剩余预算顺延
        remaining = o.target_value - amount
        if remaining > op * 100:   # 还够买至少一手
            carried.append(Order(o.code, "buy", remaining, o.reason))

    # ------------------------------------------------------------------
    def _generate_risk_sells(self, date) -> list[Order]:
        """收盘后基于个股风控生成卖出指令（次日开盘执行）。"""
        orders = []
        for code, pos in list(self.positions.items()):
            close = self._price("close", date, code)
            if close is None:
                continue
            pos.update_peak(close)
            ma = self.ma_exit.at[date, code] if (
                date in self.ma_exit.index and code in self.ma_exit.columns
            ) else np.nan

            if self.risk.check_stop_loss(pos, close):
                orders.append(Order(code, "sell", None, "stop_loss"))
            elif self.risk.check_trailing_stop(pos, close):
                orders.append(Order(code, "sell", None, "trailing_stop"))
            elif pd.notna(ma) and close < ma:
                orders.append(Order(code, "sell", None, "break_ma60"))
        return orders

    def _generate_rebalance(self, date, target_exposure: float) -> list[Order]:
        """收盘后基于综合得分生成调仓指令（次日开盘执行）。"""
        if date not in self.score.index:
            return []
        score_row = self.score.loc[date]
        trend_row = self.trend.loc[date] if date in self.trend.index \
            else pd.Series(True, index=score_row.index)
        vol_row = self.vol.loc[date] if date in self.vol.index \
            else pd.Series(index=score_row.index)

        target_w = self.build_weights(score_row, trend_row, vol_row,
                                      self.industry)
        if target_w.empty:
            return []

        equity = self._portfolio_value(date)
        investable = equity * target_exposure
        orders = []

        # 卖出：跌出综合得分前 exit_rank_pct 的持仓（且不在新目标里）
        ranked = score_row.dropna().rank(ascending=False, pct=True)
        for code in list(self.positions.keys()):
            r = ranked.get(code, 1.0)
            if code not in target_w.index and r > PORTFOLIO.exit_rank_pct:
                orders.append(Order(code, "sell", None, "rank_exit"))

        # 调整到目标权重
        for code, w in target_w.items():
            tgt_val = investable * w
            cur_val = 0.0
            if code in self.positions:
                px = self._price("close", date, code)
                cur_val = self.positions[code].shares * (px or 0)
            diff = tgt_val - cur_val
            if diff > equity * 0.005:        # 加仓阈值，过滤碎单
                orders.append(Order(code, "buy", diff, "rebalance"))
            elif diff < -equity * 0.005:
                orders.append(Order(code, "sell", -diff, "rebalance_trim"))
        return orders

    # ------------------------------------------------------------------
    def run(self) -> dict:
        prev_equity = self.init_capital
        equity_hist = []

        for i, date in enumerate(self.dates):
            # 1) 开盘撮合昨日生成的订单
            self.risk.maybe_resume(i)
            self._execute_open(date)

            # 2) 收盘计算净值
            equity = self._portfolio_value(date)
            self.equity_curve[date] = equity
            equity_hist.append(equity)

            # 3) 组合层面风控（基于当日收益与周回撤）
            daily_ret = equity / prev_equity - 1 if prev_equity else 0
            week_min = min(equity_hist[-5:]) if len(equity_hist) >= 1 else equity
            week_peak = max(equity_hist[-5:]) if len(equity_hist) >= 1 else equity
            weekly_dd_ref = week_peak
            action = self.risk.on_daily_close(i, daily_ret, equity, weekly_dd_ref)
            prev_equity = equity

            # 4) 生成次日订单
            self.pending = []
            if self.risk.is_halted(i):
                # 暂停期：清仓，不开新仓
                for code in list(self.positions.keys()):
                    self.pending.append(Order(code, "sell", None, "risk_halt"))
                continue

            # 个股风控卖出始终生效
            self.pending.extend(self._generate_risk_sells(date))

            if action["action"] == "halve":
                # 减仓至半仓：对每个持仓卖出一半市值
                for code, pos in self.positions.items():
                    px = self._price("close", date, code)
                    if px:
                        self.pending.append(
                            Order(code, "sell", pos.shares * px * 0.5,
                                  "daily_loss_halve"))

            # 调仓日重算目标
            if date in self.rebal:
                self.pending.extend(
                    self._generate_rebalance(date, action["target_exposure"]))

        equity_series = pd.Series(self.equity_curve).sort_index()
        trades = pd.DataFrame(self.trade_log)
        # 年化换手率 = 累计单边成交额 / 平均净值 / 年数
        years = max(len(equity_series) / 252, 1e-9)
        avg_equity = equity_series.mean()
        turnover = (self.turnover_traded / avg_equity / years
                    if avg_equity else np.nan)

        return {"equity": equity_series, "trades": trades,
                "turnover": turnover}
