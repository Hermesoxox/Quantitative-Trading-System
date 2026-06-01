"""
带"净化(purge)"与"禁运(embargo)"的时间序列交叉验证。

为什么需要它(对标 SOTA)
------------------------
普通 K-Fold 在金融时序上会**信息泄露**：标签是"未来 h 日收益"，跨越一段时间。
若训练集里含有标签窗口与测试集重叠的样本，模型就"偷看"了测试期信息，
导致样本内夏普虚高、上线即失效。这是 López de Prado《Advances in Financial
Machine Learning》指出的头号过拟合来源。

两个修正：
* Purge(净化)：从训练集剔除"标签窗口与测试集时间重叠"的样本。
* Embargo(禁运)：在测试集之后再留一段缓冲(embargo)，剔除紧随其后的训练样本，
  防止序列自相关造成的间接泄露。

本模块提供：
  - purged_train_index: 滚动"训练→预测"场景下，给定预测日，返回可用的训练样本掩码。
  - PurgedKFold: 兼容 sklearn 的分折器，用于超参选择与过拟合诊断(PBO)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def purged_train_index(
    sample_times: pd.Series,
    predict_time: pd.Timestamp,
    horizon_days: int,
    embargo_days: int,
) -> np.ndarray:
    """
    滚动场景：预测 predict_time 的截面时，哪些历史样本可用于训练？

    sample_times : 每个训练样本的"特征观测日"(index 任意, 值为日期)
    规则：样本的"标签结束日 = 观测日 + horizon" 必须早于
          predict_time - embargo，才不泄露未来。
    返回布尔数组(与 sample_times 等长)。
    """
    label_end = sample_times + pd.Timedelta(days=horizon_days)
    cutoff = predict_time - pd.Timedelta(days=embargo_days)
    return (label_end < cutoff).values


class PurgedKFold:
    """
    净化+禁运的 K 折分割器(时间有序)。用于超参搜索 / PBO 诊断。

    用法与 sklearn 一致：for tr, te in PurgedKFold(...).split(X, t1=label_end)
      X  : 特征(行按时间升序)
      t1 : 每个样本的标签结束时间(pd.Series, 与 X 同长, 升序对应)
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, t1: pd.Series):
        n = len(X)
        idx = np.arange(n)
        embargo = int(n * self.embargo_pct)
        # 连续时间块作为测试折
        test_folds = np.array_split(idx, self.n_splits)
        t1_vals = t1.values

        for fold in test_folds:
            test_start, test_end = fold[0], fold[-1]
            test_idx = idx[test_start:test_end + 1]
            # 测试期的时间范围
            t0_test = t1.index[test_start] if hasattr(t1, "index") else test_start
            test_label_max = t1_vals[test_start:test_end + 1].max()

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end + 1] = False
            # Purge：训练样本的标签结束时间若落入测试区间，剔除
            test_time_min = t1.index[test_start]
            for i in range(n):
                if not train_mask[i]:
                    continue
                # 样本 i 的观测时间(用 index)与标签结束时间(值)
                obs_i = t1.index[i]
                lbl_i = t1_vals[i]
                # 若样本标签窗与测试观测窗重叠则净化
                if (obs_i <= t1.index[test_end]) and (lbl_i >= test_time_min):
                    train_mask[i] = False
            # Embargo：测试折之后的 embargo 个样本也剔除
            emb_end = min(n, test_end + 1 + embargo)
            train_mask[test_end + 1:emb_end] = False

            yield idx[train_mask], test_idx
