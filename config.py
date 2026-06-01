"""
全局配置：交易规则、成本、风控阈值、回测区间、因子权重。

把所有"魔法数字"集中到一处，是防止过拟合与方便参数扰动测试的工程基础。
任何一次参数变更都应在版本控制中留痕，便于事后归因。
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ----------------------------------------------------------------------------
# 1. 交易规则与成本（A股 T+1、涨跌停、税费）
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TradingRules:
    # 印花税：卖出单边收取
    stamp_tax: float = 0.0005          # 0.05%
    # 经手费（沪深交易所）：买卖双边
    handling_fee: float = 0.0000487    # 0.00487%
    # 过户费：买卖双边（沪市原仅 A 股，现统一双边）
    transfer_fee: float = 0.00001      # 0.001%
    # 券商佣金：双边，按万 2.5 估计（含规费），不足 5 元按 5 元
    commission: float = 0.00025
    commission_min: float = 5.0
    # 滑点假设（限价单 + 次日开盘成交的综合冲击）
    slippage: float = 0.001            # 0.1%

    # 涨跌停幅度
    limit_main_board: float = 0.10     # 主板 ±10%
    limit_star_gem: float = 0.20       # 创业板(300)/科创板(688) ±20%

    # T+1：当日买入次日才可卖
    settlement_days: int = 1


# ----------------------------------------------------------------------------
# 2. 股票池过滤规则
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseFilter:
    min_listing_days: int = 60                 # 上市满 60 个自然交易日
    min_avg_amount: float = 3.0e7              # 20日日均成交额 >= 3000 万
    exclude_st: bool = True                    # 剔除 ST / *ST
    amount_window: int = 20                    # 计算日均成交额的窗口


# ----------------------------------------------------------------------------
# 3. 组合构建参数
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioParams:
    # ★ 集中持仓模式（≤5 只）。集中度高 -> 单票风险大，必须靠 regime/波动率目标
    #   叠加层 + 更严个股止损来守回撤；详见 docs/ADVANCED_STRATEGY.md。
    n_holdings_min: int = 3
    n_holdings_max: int = 5                     # 最多持有 5 只
    top_n_buy: int = 5                          # 综合得分前 N 进入候选
    exit_rank_pct: float = 0.20                # 跌出前 20% 卖出
    max_weight_per_stock: float = 0.30         # 单票上限 30%（5 只各≈20%，留缓冲）
    max_weight_per_industry: float = 0.60      # 行业暴露上限 60%（最多≈3 只同业）
    weighting: str = "risk_parity"             # "equal" 或 "risk_parity"
    trend_ma_long: int = 200                   # 趋势过滤：价格 > 200 日均线
    exit_ma: int = 60                          # 跌破 60 日均线卖出
    # --- 换手率控制（降低交易成本与净值抖动） ---
    score_smooth_span: int = 5                 # 综合得分 EMA 平滑跨度（0=不平滑）
    rebalance_band: float = 0.03               # 无交易缓冲带：|目标-当前| 占净值
                                               # 比例低于此值则不调，过滤微小漂移
    hold_buffer_rank: float = 0.40             # 持仓滞后：得分仍在前 40% 就不因
                                               # 排名卖出（比建仓门槛宽，减少来回）


# ----------------------------------------------------------------------------
# 3b. 高级叠加层：市场状态识别 + 波动率目标（SOTA 风格，集中持仓的回撤防线）
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OverlayParams:
    use_regime: bool = True                    # 启用市场状态(趋势)过滤
    regime_ma: int = 200                       # 基准 MA200 判牛熊
    use_vol_target: bool = True                # 启用波动率目标
    target_vol: float = 0.10                   # 目标年化波动 10%（集中持仓宜更低）
    vol_window: int = 20
    exposure_smooth: int = 5                   # 总仓位平滑，防频繁满/空切换
    # ML 因子合成
    use_ml_combiner: bool = True               # 用梯度提升合成(否则线性加权)
    ml_horizon: int = 5                        # 预测未来收益窗口
    ml_retrain_freq: int = 60                  # 每 60 交易日重训
    ml_embargo_days: int = 10                  # 净化禁运缓冲


# ----------------------------------------------------------------------------
# 4. 风险管理阈值
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskParams:
    # 个股层面
    stop_loss: float = 0.08                    # 建仓后 -8% 无条件止损
    trailing_activate: float = 0.15            # 盈利 >15% 启动移动止盈
    trailing_drawdown_ratio: float = 0.30      # 回撤超过盈利的 30% 止盈
    # 组合层面
    daily_loss_cut: float = 0.05               # 当日亏损 >5% 减仓至半仓
    weekly_drawdown_halt: float = 0.08         # 周回撤 >8% 清仓并停一周
    halt_days: int = 5                         # 暂停交易的交易日数
    # 流动性
    max_pct_of_volume: float = 0.05            # 单票成交不超过当日成交量 5%
    # ★ 回撤守卫(集中持仓把最大回撤压到 20% 以内的硬约束)
    #   从净值峰值起算的回撤分档限仓：软档减半仓、硬档清仓持币，
    #   冷却 N 日后重置峰值并由 regime 决定是否重新入场。
    dd_guard_soft: float = 0.10                # 回撤 >10% -> 总仓位降至 50%
    dd_guard_hard: float = 0.15                # 回撤 >15% -> 清仓持币
    dd_guard_cooldown: int = 10                # 清仓后冷却交易日数，再重置峰值
    # 注：回撤守卫只能在"有真实趋势(熊市持续下跌)"时有效护盘；在无趋势的随机
    # 波动上过度收紧只会反复止损-再入场(whipsaw)，反而加深回撤。切勿在噪声/
    # 合成数据上调这三个参数——那是典型的过拟合陷阱。应在真实数据上检验。


# ----------------------------------------------------------------------------
# 5. 回测区间
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestPeriod:
    in_sample_start: str = "2016-01-01"
    in_sample_end: str = "2021-12-31"
    out_sample_start: str = "2022-01-01"
    out_sample_end: str = "2025-12-31"
    rebalance_freq: int = 10                   # 每 10 个交易日调仓一次（持仓周期
                                               # 居 5-20 日中段，兼顾换手与时效）
    # 滚动优化：3 年训练 + 1 年验证，每 12 个月滚动一次
    roll_train_months: int = 36
    roll_valid_months: int = 12
    roll_step_months: int = 12


# ----------------------------------------------------------------------------
# 6. 因子权重（初始值；滚动优化阶段会被覆盖）
#    五大类等权打底，组内再细分，避免对单一因子过度押注。
# ----------------------------------------------------------------------------
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    # 价值（0.20）
    "ep": 0.10,            # 盈利收益率 E/P
    "bp": 0.10,            # 账面市值比 B/P
    # 动量（0.20）
    "mom_20": 0.05,
    "mom_60": 0.07,
    "mom_12_1": 0.08,      # 12 个月动量剔除最近 1 个月
    # 反转（0.15）
    "rev_5": 0.08,         # 5 日反转
    "boll_lower_bounce": 0.07,
    # 波动率（0.15，负向：低波动溢价）
    "vol_20": 0.08,
    "atr_ratio": 0.07,
    # 资金流（0.15）
    "mf_3d_ratio": 0.08,   # 3 日主力净流入比率
    "vol_surge": 0.07,     # 成交量异常放大
    # 质量（0.15）
    "roe": 0.06,
    "gross_margin": 0.05,
    "earnings_stability": 0.04,
}

# 因子方向：+1 表示因子值越大越好，-1 表示越小越好
FACTOR_DIRECTION: Dict[str, int] = {
    "ep": +1, "bp": +1,
    "mom_20": +1, "mom_60": +1, "mom_12_1": +1,
    "rev_5": -1,                  # 短期跌得多 -> 反转买入，故方向为负
    "boll_lower_bounce": +1,
    "vol_20": -1, "atr_ratio": -1,
    "mf_3d_ratio": +1, "vol_surge": +1,
    "roe": +1, "gross_margin": +1, "earnings_stability": +1,
}

FACTOR_GROUPS: Dict[str, List[str]] = {
    "value": ["ep", "bp"],
    "momentum": ["mom_20", "mom_60", "mom_12_1"],
    "reversal": ["rev_5", "boll_lower_bounce"],
    "volatility": ["vol_20", "atr_ratio"],
    "moneyflow": ["mf_3d_ratio", "vol_surge"],
    "quality": ["roe", "gross_margin", "earnings_stability"],
}


# 单例式访问
RULES = TradingRules()
UNIVERSE = UniverseFilter()
PORTFOLIO = PortfolioParams()
RISK = RiskParams()
PERIOD = BacktestPeriod()
OVERLAY = OverlayParams()
