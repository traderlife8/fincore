"""分钟级多周期远期收益（forward returns）与日内会话截断。

与 ``fincore.factor_analysis.data.compute_forward_returns``（alphalens
交易日历口径）互补：本模块面向分钟 bar，纯 shift 口径构造标签，
不涉及日历对齐。

防标签泄漏铁律：fwd_ret_N(t) = price(t+N) / price(t) - 1 只用未来价格
做标签；评估时因子必须已 shift(1)，保证因子 t 对应收益 t -> t+N。
"""
from __future__ import annotations

import pandas as pd


def truncate_session(prices: pd.DataFrame, flat_time: str = '14:55',
                     night_start: str = '17:00') -> pd.DataFrame:
    """日内会话截断：每日 flat_time 之后的价格行剔除（模拟收盘强平）。

    Args:
        prices: 宽表（index=bar_time, columns=instrument）
        flat_time: 强平时刻（HH:MM），该时刻之后（含之前）保留 <= flat_time 的行
        night_start: 夜盘起始时刻（HH:MM）；时间 >= night_start 的行视为夜盘，
            不受日盘强平影响（如国内期货夜盘 21:00 起）

    Returns:
        截断后的宽表
    """
    t_flat = pd.to_datetime(flat_time).time()
    t_night = pd.to_datetime(night_start).time()
    idx_time = prices.index.time
    keep = (idx_time <= t_flat) | (idx_time >= t_night)
    return prices[keep]


def forward_returns(prices: pd.DataFrame,
                    periods: tuple[int, ...] = (1, 5, 15)) -> dict[int, pd.DataFrame]:
    """多周期远期收益宽表：fwd_ret_N(t) = price(t+N)/price(t) - 1。

    Args:
        prices: 价格宽表（index=bar_time, columns=instrument），
            内部先按时间排序
        periods: 持有期集合（根数；1 分钟数据下 1 = 1 分钟）

    Returns:
        dict：{period: 远期收益宽表（与输入同形状）}

    Raises:
        ValueError: periods 含非正整数
    """
    if any(not isinstance(n, int) or n <= 0 for n in periods):
        raise ValueError(f'periods 必须为正整数集合，得到 {periods!r}')
    wide = prices.sort_index()
    # pct_change(n).shift(-n)：t 时刻对齐 t -> t+n 收益（shift 是标签构造，
    # 非未来函数——因子侧必须自行保证只用了 t 及之前的信息）
    return {n: wide.pct_change(n).shift(-n) for n in periods}
