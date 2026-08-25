"""分钟级 IC 显著性检验（降采样 + Newey-West HAC）。

高频铁律：相邻分钟的 IC 样本自相关极高，普通 t 检验会系统性虚高显著性。
必须先降采样（每 N 根取一个样本）再用 Newey-West HAC 修正剩余自相关。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# 默认检验参数（分钟级经验值）
DOWNSAMPLE_BARS = 15   # 降采样间隔（每 N 根取一个 IC 样本）
NW_LAG = 30            # Newey-West 滞后阶数（建议 30~60）


def newey_west_tstat(ic: pd.Series, downsample: int = DOWNSAMPLE_BARS,
                     lag: int = NW_LAG) -> dict[str, float]:
    """IC 均值显著性检验：降采样后用 OLS(HAC) 检验均值是否显著非零。

    Args:
        ic: IC 时序（compute_ic 输出）
        downsample: 降采样间隔（每 N 根取一个样本再检验）
        lag: Newey-West 滞后阶数

    Returns:
        {'t_stat': HAC t 统计量, 'p_value': p 值, 'n_eff': 降采样后样本数}

    Raises:
        RuntimeError: statsmodels 不可用，或降采样后样本不足（< lag + 10）
    """
    ic_ds = ic.dropna().iloc[::downsample]  # 降采样
    if len(ic_ds) < lag + 10:
        raise RuntimeError(f'IC 样本不足（降采样后 {len(ic_ds)} < {lag + 10}）')
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise RuntimeError('newey_west_tstat 需要 statsmodels（pip install statsmodels）') from e
    y = ic_ds.to_numpy(dtype=float)
    x = np.ones((len(y), 1))
    model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    return {'t_stat': float(model.tvalues[0]),
            'p_value': float(model.pvalues[0]),
            'n_eff': len(ic_ds)}
