"""Intraday (minute-bar) factor and trade analytics.

宽表向量化口径：index=bar_time，columns=instrument（品种/合约）。
面向期货/期权日内研究设计，与库内其他表面互补：

- ``fincore.metrics``：收益序列口径的绩效指标（日频年化默认）
- ``fincore.factor_analysis``：alphalens 长表口径的因子分析
- ``fincore.intraday``：分钟级宽表因子评价 + 交易级盈亏指标

本子包只依赖核心依赖（numpy/pandas/scipy），statsmodels 相关函数
（newey_west_tstat / acf_half_life）在首次调用时惰性导入。
"""
from fincore.intraday.factor_metrics import (
    acf_half_life,
    compute_ic,
    compute_quantile_returns,
    compute_tail_ic,
    compute_turnover,
    ic_decay_matrix,
    ic_summary,
    quantile_summary,
    turnover_summary,
)
from fincore.intraday.forward_returns import forward_returns, truncate_session
from fincore.intraday.significance import newey_west_tstat
from fincore.intraday.trades import (
    equity_calmar,
    equity_max_drawdown,
    pnl_ratio,
    profit_factor,
    win_rate,
)

__all__ = [
    'acf_half_life',
    'compute_ic',
    'compute_quantile_returns',
    'compute_tail_ic',
    'compute_turnover',
    'equity_calmar',
    'equity_max_drawdown',
    'forward_returns',
    'ic_decay_matrix',
    'ic_summary',
    'newey_west_tstat',
    'pnl_ratio',
    'profit_factor',
    'quantile_summary',
    'truncate_session',
    'turnover_summary',
    'win_rate',
]
