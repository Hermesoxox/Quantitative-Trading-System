"""
交易成本与 A股交易规则约束（涨跌停、停牌、T+1）。

成本拆解（双边/单边见 config.TradingRules）
-------------------------------------------
买入成本 = 成交额 * (佣金 + 经手费 + 过户费) + 滑点
卖出成本 = 成交额 * (佣金 + 经手费 + 过户费 + 印花税) + 滑点
其中印花税仅卖出单边 0.05%，是 A股最大的一块显性成本。
"""

from __future__ import annotations

from config import RULES
from src.data.universe import is_star_or_gem


def trade_cost(amount: float, side: str) -> float:
    """
    计算单笔交易的总成本（元）。amount 为成交金额（>0），side ∈ {buy, sell}。
    滑点在撮合价中体现，这里只算税费佣金；佣金不足 5 元按 5 元。
    """
    fee = amount * (RULES.handling_fee + RULES.transfer_fee)
    commission = max(amount * RULES.commission, RULES.commission_min)
    fee += commission
    if side == "sell":
        fee += amount * RULES.stamp_tax
    return fee


def price_limit(code: str) -> float:
    """返回该股票的涨跌停幅度（主板 10%，创业板/科创板 20%）。"""
    return RULES.limit_star_gem if is_star_or_gem(code) else RULES.limit_main_board


def can_buy(code: str, open_price: float, pre_close: float,
            high: float, low: float) -> bool:
    """
    次日开盘能否买入：开盘即涨停（开盘价 >= 涨停价且当日未打开）则无法买入。
    简化判定：若开盘价已达涨停价，视为无法成交（一字板）。
    """
    if pre_close is None or pre_close <= 0:
        return False
    limit = price_limit(code)
    limit_up = round(pre_close * (1 + limit), 2)
    # 开盘即在涨停价、且最低价也未低于涨停（全天封板）-> 买不到
    if open_price >= limit_up - 1e-6 and low >= limit_up - 1e-6:
        return False
    return True


def can_sell(code: str, open_price: float, pre_close: float,
             high: float, low: float) -> bool:
    """
    次日开盘能否卖出：开盘即跌停且全天封死（一字跌停）则无法卖出。
    """
    if pre_close is None or pre_close <= 0:
        return False
    limit = price_limit(code)
    limit_down = round(pre_close * (1 - limit), 2)
    if open_price <= limit_down + 1e-6 and high <= limit_down + 1e-6:
        return False
    return True
