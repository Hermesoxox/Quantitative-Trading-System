"""
非线性因子合成器(对标 SOTA：梯度提升 + 净化滚动训练)。

为什么用梯度提升而非深度学习
----------------------------
* 表格型、低信噪比的截面因子数据上，梯度提升树(LightGBM)长期是机构主力，
  比深度网络更稳、更省样本、更可解释(特征重要度)，且能自动捕捉因子间的
  非线性与交互(线性加权做不到)。
* 深度学习在A股这种样本量/信噪比下极易过拟合、容量受限收益有限，故不采用。

防过拟合工程(关键)
------------------
* 滚动训练 + 净化禁运：每个预测截面只用"标签已结束且早于预测期(留禁运缓冲)"的
  历史样本训练，杜绝未来函数(见 cv.purged_train_index)。
* 标签用"截面收益排名"(学习排序思想)，对极端值稳健、跨期可比。
* 强正则：浅树、低学习率、列/行采样、最小叶子样本数，限制模型复杂度。
* 若无 lightgbm/sklearn，自动回退到岭回归线性合成，保证可运行。

输出与线性合成一致：score 宽表(date x code)，可直接喂给回测引擎。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import purged_train_index

# 可用性探测，决定模型后端
try:
    import lightgbm as lgb
    _BACKEND = "lightgbm"
except Exception:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        _BACKEND = "sklearn"
    except Exception:
        _BACKEND = "linear"


def _stack_features(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把 {因子名: 宽表} 堆叠成长表特征矩阵 (index=[date,code], col=因子)。"""
    cols = {}
    for name, wide in factors.items():
        cols[name] = wide.stack(future_stack=True) if hasattr(pd.DataFrame, "stack") \
            else wide.stack()
    df = pd.DataFrame(cols)
    df.index.names = ["date", "code"]
    return df


def _forward_return_label(close: pd.DataFrame, horizon: int) -> pd.Series:
    """未来 horizon 日收益，截面 rank 标准化为 [-0.5,0.5]，作为学习排序标签。"""
    fwd = close.shift(-horizon) / close - 1
    # 逐日截面 rank -> [0,1] -> 去均值
    ranked = fwd.rank(axis=1, pct=True) - 0.5
    s = ranked.stack(future_stack=True) if True else ranked.stack()
    s.index.names = ["date", "code"]
    return s


class MLCombiner:
    """梯度提升因子合成器(带回退)。"""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.03,
                 max_depth: int = 3, subsample: float = 0.8,
                 colsample: float = 0.8, min_child: int = 200,
                 seed: int = 42):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           max_depth=max_depth, subsample=subsample,
                           colsample=colsample, min_child=min_child, seed=seed)
        self.model = None
        self.backend = _BACKEND
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 用 numpy 训练，保证 fit/predict 口径一致(避免 sklearn 特征名告警)
        self.feature_names = list(X.columns)
        Xv, yv = np.asarray(X.values, dtype=float), np.asarray(y.values, dtype=float)
        p = self.params
        if self.backend == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                max_depth=p["max_depth"], num_leaves=2 ** p["max_depth"],
                subsample=p["subsample"], subsample_freq=1,
                colsample_bytree=p["colsample"], min_child_samples=p["min_child"],
                reg_lambda=1.0, random_state=p["seed"], n_jobs=-1, verbose=-1)
            self.model.fit(Xv, yv)
        elif self.backend == "sklearn":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                max_depth=p["max_depth"], l2_regularization=1.0,
                min_samples_leaf=p["min_child"], random_state=p["seed"])
            self.model.fit(Xv, yv)
        else:  # 岭回归线性回退
            A = np.c_[np.ones(len(Xv)), Xv]
            lam = 1.0
            ATA = A.T @ A + lam * np.eye(A.shape[1])
            self.coef_ = np.linalg.solve(ATA, A.T @ yv)
            self.model = "linear"
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = np.asarray(X.values, dtype=float)
        if self.backend == "linear":
            A = np.c_[np.ones(len(Xv)), Xv]
            return A @ self.coef_
        return self.model.predict(Xv)

    def feature_importance(self) -> pd.Series | None:
        if self.backend == "lightgbm" and self.model is not None:
            return pd.Series(self.model.feature_importances_,
                             index=self.feature_names).sort_values(ascending=False)
        return None


def rolling_ml_scores(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizon: int = 5,
    retrain_freq: int = 60,
    train_window_days: int = 756,   # 约3年
    embargo_days: int = 10,
    min_train_rows: int = 2000,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    滚动净化训练，产出综合得分宽表(date x code)。

    流程(每 retrain_freq 个交易日重训一次)：
      1. 取训练样本：观测日在 [predict_date - train_window, ...] 且
         标签结束日 < predict_date - embargo(净化+禁运)。
      2. 用 MLCombiner 拟合 (因子 -> 截面收益排名)。
      3. 用该模型预测后续 retrain_freq 天的截面得分，直到下次重训。

    返回 (score_wide, last_feature_importance)。
    """
    feat_cols = list(factors.keys())
    feats = _stack_features(factors)
    label = _forward_return_label(close, horizon)
    data = feats.join(label.rename("_y"), how="inner")
    # 只按"标签"过滤；缺失因子按 0(中性)填充，而非整行删除——否则某些股票
    # (如银行无毛利率)会被全部剔除，训练样本骤减致模型无法训练。因子已横截面
    # 标准化，均值≈0，填 0 即"该维度无信息"。至少需一个因子非缺失才保留该行。
    data = data.dropna(subset=["_y"])
    data = data[data[feat_cols].notna().any(axis=1)]
    data[feat_cols] = data[feat_cols].fillna(0.0)
    if data.empty:
        return pd.DataFrame(index=close.index, columns=close.columns), None

    obs_dates = data.index.get_level_values("date")
    all_dates = close.index
    score = pd.DataFrame(index=all_dates, columns=close.columns, dtype=float)

    model = None
    last_imp = None
    # 预测日集合(交易日)，从有足够训练数据处开始
    start_i = 0
    for i, pdate in enumerate(all_dates):
        retrain = (model is None) or (i % retrain_freq == 0)
        if retrain:
            win_start = pdate - pd.Timedelta(days=int(train_window_days * 1.45))
            in_win = (obs_dates >= win_start) & (obs_dates <= pdate)
            sub = data[in_win]
            if len(sub) >= min_train_rows:
                st = sub.index.get_level_values("date").to_series().reset_index(drop=True)
                mask = purged_train_index(st, pdate, horizon, embargo_days)
                tr = sub[mask.tolist() if hasattr(mask, "tolist") else mask]
                if len(tr) >= min_train_rows:
                    model = MLCombiner().fit(tr[feat_cols], tr["_y"])
                    last_imp = model.feature_importance()
        if model is None:
            continue
        # 预测当日截面
        if pdate in factors[feat_cols[0]].index:
            row = pd.DataFrame({c: factors[c].loc[pdate] for c in feat_cols})
            row = row[row.notna().any(axis=1)].fillna(0.0)  # 缺失因子按中性0填充
            if not row.empty:
                pred = model.predict(row)
                score.loc[pdate, row.index] = pred
    return score, last_imp
