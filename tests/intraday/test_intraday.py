"""fincore.intraday 自检测试：宽表向量化指标与交易级指标的数值正确性。"""
import numpy as np
import pandas as pd
import pytest

from fincore.intraday import (
    acf_half_life,
    compute_ic,
    compute_quantile_returns,
    compute_tail_ic,
    compute_turnover,
    equity_calmar,
    equity_max_drawdown,
    forward_returns,
    ic_summary,
    newey_west_tstat,
    pnl_ratio,
    profit_factor,
    quantile_summary,
    truncate_session,
    turnover_summary,
    win_rate,
)


def _wide(values, freq='min', start='2024-01-02 09:00'):
    """从二维数组构造分钟级宽表（bar_time x instrument）。"""
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame(values, index=idx,
                        columns=[f'i{k}' for k in range(len(values[0]))])


@pytest.mark.p0
class TestComputeIC:
    def test_perfect_monotonic_factor_ic_is_one(self):
        """因子与远期收益完全同序 -> 每行 Rank IC = 1（与 scipy.spearmanr 一致）。"""
        base = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (4, 1))
        factor = _wide(base)
        fwd = _wide(base * 0.01)  # 同序不同尺度
        ic = compute_ic(factor, fwd, min_width=3)
        assert np.allclose(ic.dropna(), 1.0)

    def test_reversed_factor_ic_is_minus_one(self):
        base = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (4, 1))
        ic = compute_ic(_wide(base), _wide(base[:, ::-1]), min_width=3)
        assert np.allclose(ic.dropna(), -1.0)

    def test_matches_scipy_spearmanr_with_noise(self):
        """含噪声随机数据下与 scipy.stats.spearmanr(axis=1) 逐行一致。"""
        from scipy.stats import rankdata
        rng = np.random.default_rng(11)
        f = rng.normal(size=(20, 8))
        r = rng.normal(size=(20, 8))
        ic = compute_ic(_wide(f), _wide(r), min_width=3)
        for i in range(len(f)):
            expected = np.corrcoef(rankdata(f[i]), rankdata(r[i]))[0, 1]
            assert ic.iloc[i] == pytest.approx(expected, abs=1e-12)

    def test_nan_propagation(self):
        factor = _wide([[np.nan, 2.0, 3.0]] * 5)
        fwd = _wide([[1.0, 2.0, 3.0]] * 5)
        ic = compute_ic(factor, fwd, min_width=3)
        # 有效截面宽度只有 2 < min_width=3 -> 全 NaN
        assert ic.isna().all()


@pytest.mark.p1
class TestQuantile:
    def test_quantile_returns_two_groups(self):
        """8 资产 2 组：pct rank 边界口径下 Q1=前 3 位、Q2=后 5 位
        （pct=0.5 时 floor(0.5*2)=1 已进 Q2；pct=1.0 被 clip 回末组）。"""
        factor = _wide([list(range(1, 9))] * 3)
        fwd = _wide([[0.01 * k for k in range(1, 9)]] * 3)
        qr = compute_quantile_returns(factor, fwd, n_quantiles=2)
        assert np.allclose(qr['Q1'], np.mean([0.01, 0.02, 0.03]))
        assert np.allclose(qr['Q2'], np.mean([0.04, 0.05, 0.06, 0.07, 0.08]))

    def test_quantile_summary_spread_and_monotonicity(self):
        factor = _wide([list(range(1, 9))] * 3)
        fwd = _wide([[0.01 * k for k in range(1, 9)]] * 3)
        qr = compute_quantile_returns(factor, fwd, n_quantiles=2)
        s = quantile_summary(qr)
        # 完美单调：Q2-Q1 = 0.04 -> 400 bp；Spearman(组号,收益)=1
        assert abs(s['long_short_bp'] - 400.0) < 1e-6
        assert abs(s['monotonicity'] - 1.0) < 1e-9


@pytest.mark.p1
class TestTurnover:
    def test_constant_factor_zero_turnover(self):
        factor = _wide(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (10, 1)))
        t = compute_turnover(factor).dropna()
        assert np.allclose(t, 0.0)

    def test_full_shuffle_turnover_near_one(self):
        """相邻时点 rank 完全反转 -> 换手接近上限 1。"""
        rows = [[1.0, 2.0, 3.0, 4.0] if k % 2 == 0 else [4.0, 3.0, 2.0, 1.0]
                for k in range(6)]
        t = compute_turnover(_wide(rows)).dropna()
        assert (t > 0.9).all()

    def test_turnover_summary_daily_scale(self):
        s = pd.Series([0.1] * 10)
        summary = turnover_summary(s, bars_per_day=100)
        assert abs(summary['turnover_per_day'] - 10.0) < 1e-9


@pytest.mark.p1
class TestICTools:
    def test_ic_summary_values(self):
        ic = pd.Series([0.1, -0.1, 0.1, 0.1])
        s = ic_summary(ic)
        assert abs(s['ic_mean'] - 0.05) < 1e-12
        assert s['win_rate'] == 0.75
        assert s['n'] == 4

    def test_newey_west_insufficient_samples_raises(self):
        with pytest.raises(RuntimeError, match='样本不足'):
            newey_west_tstat(pd.Series([0.1] * 20), downsample=15, lag=30)

    def test_newey_west_positive_mean_significant(self):
        rng = np.random.default_rng(7)
        ic = pd.Series(rng.normal(0.05, 0.01, 600))
        r = newey_west_tstat(ic, downsample=1, lag=5)
        assert r['t_stat'] > 5
        assert r['p_value'] < 1e-5

    def test_acf_half_life_white_noise_is_one(self):
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.normal(0, 1, 300))
        r = acf_half_life(ic, max_lag=10)
        assert r['half_life'] == 1.0  # 白噪声 lag=1 即低于 0.5

    def test_acf_half_life_insufficient_raises(self):
        with pytest.raises(RuntimeError, match='样本不足'):
            acf_half_life(pd.Series([0.1] * 5), max_lag=10)


@pytest.mark.p1
class TestTailIC:
    def test_tail_ic_needs_thirty_pairs(self):
        rng = np.random.default_rng(1)
        with pytest.raises(RuntimeError, match='尾部 IC 样本不足'):
            compute_tail_ic(_wide(rng.normal(size=(5, 4))),
                            _wide(rng.normal(size=(5, 4))))  # 20 对 < 30

    def test_tail_ic_noisy_proxy_of_returns(self):
        rng = np.random.default_rng(3)
        n_t, n_a = 60, 5  # 300 对
        ret = rng.normal(0, 0.01, size=(n_t, n_a))
        # 因子 = 收益 + 微噪声 -> 尾部 IC 应显著为正
        factor = ret + rng.normal(0, 1e-4, size=(n_t, n_a))
        out = compute_tail_ic(_wide(factor), _wide(ret), tail_pct=0.15)
        assert out['tail_ic'] > 0.9


@pytest.mark.p1
class TestForwardReturns:
    def test_one_period(self):
        prices = _wide([[100.0], [110.0], [121.0]])
        out = forward_returns(prices, periods=(1,))
        fr = out[1]
        assert np.allclose(fr.iloc[:2, 0], [0.1, 0.1])
        assert np.isnan(fr.iloc[2, 0])  # 最后一根无未来价格

    def test_multi_period_keys_sorted_input(self):
        prices = _wide([[100.0], [101.0], [102.0], [103.0]])
        out = forward_returns(prices, periods=(1, 2))
        assert set(out) == {1, 2}
        # t=0 的 2 期收益 = 102/100 - 1
        assert np.allclose(out[2].iloc[0, 0], 0.02, atol=1e-9)

    def test_invalid_periods_raise(self):
        with pytest.raises(ValueError, match='periods'):
            forward_returns(_wide([[1.0]]), periods=(0,))


@pytest.mark.p1
class TestTruncateSession:
    def test_force_flat_and_night_kept(self):
        idx = pd.to_datetime([
            '2024-01-02 14:50', '2024-01-02 14:55', '2024-01-02 14:56',
            '2024-01-02 21:00',
        ])
        prices = pd.DataFrame({'a': [1.0] * 4}, index=idx)
        out = truncate_session(prices, flat_time='14:55', night_start='17:00')
        # 14:56 被截断；夜盘 21:00 保留
        assert list(out.index.time) == [pd.Timestamp('14:50').time(),
                                        pd.Timestamp('14:55').time(),
                                        pd.Timestamp('21:00').time()]


@pytest.mark.p0
class TestTradeMetrics:
    def test_profit_factor(self):
        pnl = pd.Series([100.0, -50.0, 200.0, -50.0])
        assert profit_factor(pnl) == pytest.approx(3.0)

    def test_pnl_ratio_no_losses_inf(self):
        assert pnl_ratio(pd.Series([10.0, 20.0])) == float('inf')

    def test_pnl_ratio_empty_raises(self):
        with pytest.raises(RuntimeError, match='无有效交易样本'):
            pnl_ratio(pd.Series([np.nan]))

    def test_win_rate(self):
        assert win_rate(pd.Series([1.0, -1.0, 2.0])) == pytest.approx(2 / 3)

    def test_equity_max_drawdown(self):
        eq = pd.Series([100.0, 120.0, 60.0, 90.0])
        assert equity_max_drawdown(eq) == pytest.approx(-0.5)

    def test_equity_calmar_no_dd_is_inf(self):
        """资金曲线单调上行无回撤 -> inf。"""
        assert equity_calmar(pd.Series([100.0, 200.0]), ann_bars=100) == float('inf')

    def test_equity_calmar_with_dd(self):
        # [100, 150, 75]: total_ret=-0.25, dd=-0.5; n=3, ann_bars=3 -> ann_ret=-0.25
        c = equity_calmar(pd.Series([100.0, 150.0, 75.0]), ann_bars=3)
        assert c == pytest.approx((-0.25) / 0.5)
