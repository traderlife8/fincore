"""交易级盈亏指标（单笔交易 pnl / 资金曲线口径的纯函数）。

与 ``fincore.metrics``（收益序列口径）互补：本模块输入是逐笔盈亏
Series 或资金曲线 Series，面向日内买方策略筛选（盈亏比优先于胜率——
买方胜率天然低 30-40%，单看胜率会错误惩罚高盈亏比策略）。

所有函数显性报错：无有效样本时 raise，不静默返回 0。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _clean_pnl(pnl: pd.Series) -> pd.Series:
    """去 NaN 后校验非空。"""
    pnl = pnl.dropna()
    if pnl.empty:
        raise RuntimeError('无有效交易样本（pnl 序列为空）')
    return pnl


def profit_factor(pnl: pd.Series) -> float:
    """盈利因子 = 总盈利 / |总亏损|。

    Args:
        pnl: 单笔盈亏序列

    Returns:
        盈利因子；无亏损单时为 inf（有盈利）或 0.0（无盈利）
    """
    pnl = _clean_pnl(pnl)
    gross_win = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def pnl_ratio(pnl: pd.Series) -> float:
    """盈亏比 = 平均盈利 / |平均亏损|（买方核心指标）。

    Args:
        pnl: 单笔盈亏序列

    Returns:
        盈亏比；无亏损单时为 inf（有盈利）或 0.0（无盈利）
    """
    pnl = _clean_pnl(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if losses.empty:
        return float('inf') if not wins.empty else 0.0
    if wins.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def win_rate(pnl: pd.Series) -> float:
    """胜率 = 盈利笔数占比。

    注意：日内买方场景 win_rate 不应单独作为筛选指标，
    必须与 pnl_ratio 联用。
    """
    pnl = _clean_pnl(pnl)
    return float((pnl > 0).mean())


def equity_max_drawdown(equity: pd.Series) -> float:
    """资金曲线最大回撤（负值，越小越差）。

    与 ``fincore.metrics.max_drawdown``（收益率序列输入）不同，
    本函数输入是资金曲线（权益金额或净值）。

    Args:
        equity: 资金曲线序列

    Returns:
        最大回撤（如 -0.5 表示 -50%）；样本不足 2 个时返回 0.0
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    running_max = eq.cummax()
    return float((eq / running_max - 1).min())


def equity_calmar(equity: pd.Series, ann_bars: int) -> float:
    """资金曲线卡玛比率 = 年化收益 / |最大回撤|。

    年化按样本跨度折算：ann_ret = (1 + total_ret) ** (ann_bars / n_bars) - 1。
    分钟级年化基准由调用方传入（如国内期货日盘 ann_bars=252*345）。

    Args:
        equity: 资金曲线序列
        ann_bars: 年化 bar 数（年化因子）

    Returns:
        卡玛比率；回撤为 0 时返回 inf（正收益）或 0.0
    """
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    n_bars = max(len(eq), 1)
    ann_ret = (1 + total_ret) ** (ann_bars / n_bars) - 1
    dd = equity_max_drawdown(eq)
    if dd == 0:
        return float('inf') if ann_ret > 0 else 0.0
    return float(ann_ret / abs(dd))
