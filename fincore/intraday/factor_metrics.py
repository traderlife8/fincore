"""宽表向量化因子评价指标（分钟级日内研究核心）。

所有函数的输入口径统一为宽表：index=bar_time，columns=instrument。
向量化实现（rank/mean 沿 axis=1），禁止 apply(axis=1)——分钟级数据
（数百时点 x 数十品种 x 数万根 K 线）下 groupby/apply 口径慢一个量级。

IC 显著性铁律：相邻分钟 IC 序列自相关极高，普通 t 检验系统性虚高，
显著性论证必须用 ``fincore.intraday.newey_west_tstat``（降采样 + HAC）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ic(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame,
               min_width: int = 5) -> pd.Series:
    """每根 K 线横截面 Rank IC（Spearman），向量化实现。

    数值口径：rank 后的 Pearson 相关。协方差与标准差统一用总体矩
    （ddof=0），与 scipy.stats.spearmanr 逐行一致——若混用 ddof=1
    会导致 IC 被系统性低估 (n-1)/n 倍。

    Args:
        factor_df: 因子宽表（bar_time x instrument）
        fwd_ret: 远期收益宽表（同形状，必须是 t -> t+N 的远期收益）
        min_width: 有效截面宽度下限（低于该宽度时点返回 NaN；
            小截面冒烟测试可降低）

    Returns:
        IC 时序（index=bar_time）
    """
    mask = ~(factor_df.isna() | fwd_ret.isna())
    # 有效截面宽度下限：沿 axis=0 广播行级条件，避免 DataFrame-Series 列对齐陷阱
    valid_rows = mask.sum(axis=1).ge(min_width)
    mask = mask.where(valid_rows, False, axis=0)
    # where(mask) 后无效处已是 NaN，rank 保持 NaN，后续统计无需再套掩码
    f_rank = factor_df.where(mask).rank(axis=1)
    r_rank = fwd_ret.where(mask).rank(axis=1)
    # 向量化 Spearman：rank 后的 Pearson 相关（总体矩口径，ddof=0）
    cov = (f_rank * r_rank).mean(axis=1) - f_rank.mean(axis=1) * r_rank.mean(axis=1)
    denom = (f_rank.std(axis=1, ddof=0) * r_rank.std(axis=1, ddof=0)).replace(0, pd.NA)
    return (cov / denom).where(mask.any(axis=1))


def compute_tail_ic(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame,
                    tail_pct: float = 0.10) -> dict[str, float]:
    """尾部 IC：只对远期收益分布极端样本计算 Spearman 相关。

    均值 IC 接近 0 但尾部 IC 强的因子对买方仍有效——买方靠尾部爆发盈利。

    纯 numpy 路径实现：不经过 pandas stack()，超大面板（数十万 bar x
    数十品种）下避免长表展开的内存翻倍与对齐开销。

    Args:
        factor_df: 因子宽表（bar_time x instrument）
        fwd_ret: 远期收益宽表（同形状）
        tail_pct: 尾部比例（默认 top/bottom 各 10%）

    Returns:
        {'tail_ic': 尾部样本 IC, 'mean_ic': 全样本 IC 均值,
         'tail_ratio': |尾部 IC| / |均值 IC|（>=2 视为尾部增强因子）}

    Raises:
        RuntimeError: 尾部有效样本不足（< 30 对）
    """
    from scipy.stats import spearmanr  # 惰性导入：scipy.stats 加载较重

    # 先按索引/列内连接对齐（与 compute_ic 的 pandas 隐式对齐行为一致），
    # 允许两表行集存在细微差异（如因子 parquet 与价格截断后 bar 集不同）
    factor_aligned, fwd_aligned = factor_df.align(fwd_ret, join='inner')
    f_vals = factor_aligned.to_numpy(dtype=float)
    r_vals = fwd_aligned.to_numpy(dtype=float)
    valid = ~(np.isnan(f_vals) | np.isnan(r_vals))
    q_lo, q_hi = np.quantile(r_vals[valid], [tail_pct, 1 - tail_pct])
    tail_mask = valid & ((r_vals <= q_lo) | (r_vals >= q_hi))
    if tail_mask.sum() < 30:
        raise RuntimeError('尾部 IC 样本不足（< 30 对），无法计算')
    tail_ic = float(spearmanr(f_vals[tail_mask], r_vals[tail_mask]).statistic)
    mean_ic = float(compute_ic(factor_df, fwd_ret).mean())
    ratio = abs(tail_ic) / abs(mean_ic) if abs(mean_ic) > 1e-9 else float('inf')
    return {'tail_ic': tail_ic, 'mean_ic': mean_ic, 'tail_ratio': ratio}


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """IC 时序摘要：均值 / 波动 / IR / 胜率 / 样本数。

    Args:
        ic: compute_ic 输出的 IC 时序

    Returns:
        {'ic_mean', 'ic_std', 'ir', 'win_rate', 'n'}

    Raises:
        RuntimeError: IC 序列为空
    """
    ic = ic.dropna()
    if ic.empty:
        raise RuntimeError('IC 序列为空，无法汇总')
    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    return {'ic_mean': ic_mean, 'ic_std': ic_std, 'ir': ir,
            'win_rate': float((ic > 0).mean()), 'n': len(ic)}


def compute_quantile_returns(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame,
                             n_quantiles: int = 5) -> pd.DataFrame:
    """分位组合收益：每个时点按因子值分组，计算各组远期收益均值。

    向量化实现：pct rank x n_quantiles 取整得组号（NaN 自然传播）。

    Args:
        factor_df: 因子宽表（bar_time x instrument）
        fwd_ret: 远期收益宽表（同形状）
        n_quantiles: 分组数（默认 5；截面宽度不足时组内均值自然为 NaN）

    Returns:
        DataFrame：index=bar_time，columns=Q1..QN，values=组内远期收益均值
    """
    mask = ~(factor_df.isna() | fwd_ret.isna())
    f_rank = factor_df.where(mask).rank(axis=1, pct=True)
    # 组号：pct in (0,1] -> 0..n_q-1（clip 防止 pct=1 越界；NaN 传播）
    q_val = np.floor(f_rank * n_quantiles).clip(upper=n_quantiles - 1)
    ret_by_q = {f'Q{q + 1}': fwd_ret.where(q_val == q).mean(axis=1)
                for q in range(n_quantiles)}
    return pd.DataFrame(ret_by_q)


def quantile_summary(quant_ret: pd.DataFrame) -> dict:
    """分位收益摘要：多空价差 + 单调性。

    Args:
        quant_ret: compute_quantile_returns 输出

    Returns:
        {'long_short_bp': 多空价差（bp/时点）,
         'monotonicity': Spearman(组号, 组均收益),
         'quant_means': 各组均值 dict}

    Raises:
        RuntimeError: 有效分位组不足 2 个
    """
    from scipy.stats import spearmanr  # 惰性导入：scipy.stats 加载较重

    quant_means = quant_ret.mean().dropna()
    if len(quant_means) < 2:
        raise RuntimeError('分位收益组数不足，无法计算单调性')
    ls_spread = float(quant_means.iloc[-1] - quant_means.iloc[0])
    group_ids = np.arange(1, len(quant_means) + 1)
    rho, _ = spearmanr(group_ids, quant_means.to_numpy())
    return {'long_short_bp': ls_spread * 1e4,
            'monotonicity': float(rho),
            'quant_means': quant_means.to_dict()}


def compute_turnover(factor_df: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    """因子分组换手率（向量化代理口径）：分位 rank 的逐根变化幅度。

    口径：pct rank 的相邻时点绝对差分截面均值 x n_quantiles，
    近似等于 top/bottom 组成员更换比例（rank 全变 = 完全换组）。

    Args:
        factor_df: 因子宽表（bar_time x instrument）
        n_quantiles: 分组数（用于尺度归一到 [0, 1] 近似）

    Returns:
        换手率时序（index=bar_time，首个时点为 NaN）
    """
    f_rank = factor_df.rank(axis=1, pct=True)
    # rank 变化幅度 x 分组数：pct rank 变化 1/n_q 约等于跨过一个组
    turnover = f_rank.diff().abs().mean(axis=1) * n_quantiles
    return turnover.clip(upper=1.0)


def turnover_summary(turnover: pd.Series, bars_per_day: int = 345) -> dict[str, float]:
    """换手率摘要（换算为每日口径）。

    Args:
        turnover: compute_turnover 输出
        bars_per_day: 每日 K 线数（默认 345，约等于国内期货日盘分钟数）

    Returns:
        {'turnover_per_bar': 每根均值, 'turnover_per_day': 每日均值}

    Raises:
        RuntimeError: 换手率序列为空
    """
    t = turnover.dropna()
    if t.empty:
        raise RuntimeError('换手率序列为空，无法汇总')
    per_bar = float(t.mean())
    return {'turnover_per_bar': per_bar,
            'turnover_per_day': per_bar * bars_per_day}


def ic_decay_matrix(factor_df: pd.DataFrame, prices: pd.DataFrame,
                    periods: tuple[int, ...] = (1, 3, 5, 15, 30, 60)) -> pd.Series:
    """不同持有期的 IC 均值（IC 衰减曲线）。

    Args:
        factor_df: 因子宽表（bar_time x instrument）
        prices: 价格宽表（bar_time x instrument，close 口径）
        periods: 持有期集合（根数；1 分钟数据下 1 = 1 分钟）

    Returns:
        Series：index=持有期，values=该持有期 IC 均值
    """
    ic_by_period = {
        n: float(compute_ic(factor_df, prices.pct_change(n).shift(-n)).mean())
        for n in periods
    }
    return pd.Series(ic_by_period).sort_index()


def acf_half_life(ic: pd.Series, max_lag: int = 60) -> dict[str, float]:
    """IC 序列自相关半衰期（ACF 首次衰减到 0.5 的 lag）。

    半衰期到策略频率的映射由调用方定义（如 <5min 高频、5-15min 中频、
    >60min 不适合日内）。

    Args:
        ic: IC 时序
        max_lag: 最大滞后阶数（默认 60）

    Returns:
        {'half_life': 半衰期 lag（未衰减到 0.5 时为 max_lag）,
         'acf_at_max': lag=max_lag 处的 ACF 值}

    Raises:
        RuntimeError: statsmodels 不可用或样本不足
    """
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError as e:
        raise RuntimeError('acf_half_life 需要 statsmodels（pip install statsmodels）') from e
    ic = ic.dropna()
    if len(ic) < max_lag + 10:
        raise RuntimeError(f'IC 样本不足（{len(ic)} < {max_lag + 10}），无法算 ACF')
    acf_vals = acf(ic.to_numpy(dtype=float), nlags=max_lag)
    half_life = float(max_lag)
    below = np.where(acf_vals[1:] <= 0.5)[0]
    if len(below) > 0:
        half_life = float(below[0] + 1)  # lag 从 1 开始计
    return {'half_life': half_life, 'acf_at_max': float(acf_vals[max_lag])}
