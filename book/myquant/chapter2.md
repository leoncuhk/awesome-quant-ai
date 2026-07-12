# Chapter 2: Mean Reversion Strategies

**Overview**:
Mean reversion strategies rest on the hypothesis that prices revert toward a long-run mean. They trade against extreme deviations from that mean and profit when the deviation closes. These strategies tend to perform well in range-bound markets but can suffer large losses in strongly trending markets.

**Theoretical foundations**:
- **Mean reversion**: prices fluctuate around intrinsic value and revert after deviating from it
- **Statistical arbitrage**: exploiting statistical regularities in the relationships between prices
- **Cointegration**: identifying price relationships that are stable in the long run

## 2.1 Statistical Arbitrage Strategies

### 2.1.1 Pairs Trading

**Strategy rationale**:
Select two stocks whose prices have historically moved together. When the spread between them deviates from its historical mean, trade against the deviation and wait for the spread to revert. Pairs trading is a market-neutral strategy: in principle its P&L does not depend on the direction of the overall market.

**Core components**:
- **Pair selection**: similar fundamentals, same industry, high historical correlation, cointegrated prices
- **Spread construction**: Spread = Stock1 − β × Stock2, where β is the hedge ratio
- **Signal generation**: trade when the z-score of the spread crosses a threshold
- **Risk controls**: z-score stop-loss, time-based stop, ongoing monitoring of the pair relationship

**Avoiding look-ahead bias**: the hedge ratio β is estimated by OLS on an in-sample (training) window only, and trading takes place exclusively on the subsequent out-of-sample (test) window. The rolling z-score uses only past observations by construction, so the whole signal path is causal.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from typing import Tuple, Dict, List


class PairsTradingStrategy:
    """Pairs trading with an explicit in-sample / out-of-sample split.

    The hedge ratio is estimated by OLS on the training window only;
    signals are generated and performance is evaluated on the
    subsequent test window, so no future data leaks into the backtest.
    """

    def __init__(self, lookback_period=60, entry_threshold=2.0, exit_threshold=0.5,
                 stop_loss=3.0, max_holding_days=30, train_fraction=0.5):
        """
        Args:
            lookback_period: rolling window for the spread z-score
            entry_threshold: z-score level that triggers an entry
            exit_threshold: z-score level that triggers an exit
            stop_loss: z-score level that triggers a stop-loss
            max_holding_days: maximum holding period in calendar days,
                not trading days (time stop)
            train_fraction: fraction of the sample used to estimate the
                hedge ratio (in-sample); the remainder is traded out-of-sample
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        self.train_fraction = train_fraction

    def find_cointegrated_pairs(self, price_data: pd.DataFrame, min_correlation=0.7) -> List[Dict]:
        """Scan a universe for cointegrated pairs.

        Note: in production, run this scan on in-sample data only —
        selecting pairs on the same data you later backtest on is itself
        a form of look-ahead (selection) bias.
        """
        n = price_data.shape[1]
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                stock1 = price_data.iloc[:, i].dropna()
                stock2 = price_data.iloc[:, j].dropna()

                # Require enough overlapping history
                common_index = stock1.index.intersection(stock2.index)
                if len(common_index) < 100:  # at least 100 trading days
                    continue

                stock1_aligned = stock1.loc[common_index]
                stock2_aligned = stock2.loc[common_index]

                # Correlation filter
                correlation = stock1_aligned.corr(stock2_aligned)
                if abs(correlation) < min_correlation:
                    continue

                # Engle-Granger cointegration test
                try:
                    score, pvalue, _ = coint(stock1_aligned, stock2_aligned)
                    if pvalue < 0.05:  # 5% significance level
                        pairs.append({
                            'stock1': price_data.columns[i],
                            'stock2': price_data.columns[j],
                            'pvalue': pvalue,
                            'correlation': correlation,
                            'adf_stat': score
                        })
                except Exception:
                    continue

        return sorted(pairs, key=lambda x: x['pvalue'])

    def estimate_hedge_ratio(self, stock1_train: pd.Series,
                             stock2_train: pd.Series) -> Tuple[float, Dict]:
        """Estimate the hedge ratio by OLS on the training window only."""
        X = sm.add_constant(stock2_train)
        model = sm.OLS(stock1_train, X).fit()
        hedge_ratio = float(model.params.iloc[1])

        stats = {
            'hedge_ratio': hedge_ratio,
            'intercept': float(model.params.iloc[0]),
            'r_squared': model.rsquared,
            'p_value': float(model.pvalues.iloc[1]),
            'std_error': float(model.bse.iloc[1])
        }
        return hedge_ratio, stats

    def calculate_rolling_zscore(self, spread: pd.Series) -> pd.Series:
        """Rolling z-score of the spread (uses past observations only)."""
        rolling_mean = spread.rolling(window=self.lookback_period).mean()
        rolling_std = spread.rolling(window=self.lookback_period).std()
        return (spread - rolling_mean) / rolling_std

    def generate_signals(self, stock1: pd.Series, stock2: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """Generate positions on the out-of-sample window.

        The hedge ratio comes from the training window; positions are
        forced to zero during the training window and only trade
        out-of-sample.
        """
        # Align the two series
        common_index = stock1.index.intersection(stock2.index)
        stock1_aligned = stock1.loc[common_index]
        stock2_aligned = stock2.loc[common_index]

        # In-sample / out-of-sample split
        split_idx = int(len(common_index) * self.train_fraction)
        test_start = common_index[split_idx]
        hedge_ratio, stats = self.estimate_hedge_ratio(
            stock1_aligned.iloc[:split_idx], stock2_aligned.iloc[:split_idx]
        )
        stats['train_end'] = common_index[split_idx - 1]
        stats['test_start'] = test_start

        # Spread with the fixed, in-sample hedge ratio.
        # Computing it over the full sample lets the rolling z-score warm up
        # on training data, so the test window has valid signals from day one.
        spread = stock1_aligned - hedge_ratio * stock2_aligned
        zscore = self.calculate_rolling_zscore(spread)

        signals = pd.DataFrame(index=common_index)
        signals['spread'] = spread
        signals['zscore'] = zscore

        # Raw signal conditions (NaN z-scores compare as False)
        signals['long_entry'] = zscore < -self.entry_threshold   # spread too low: buy the spread
        signals['short_entry'] = zscore > self.entry_threshold   # spread too high: sell the spread
        signals['exit'] = zscore.abs() < self.exit_threshold     # spread reverted: close
        signals['stop_loss'] = zscore.abs() > self.stop_loss

        # Build the position path (state machine); trade only out-of-sample.
        # After a stop-loss or time stop, re-entry is locked out until the
        # z-score first crosses back inside the entry band — otherwise the
        # entry rule would re-establish the stopped-out position on the very
        # same bar (|z| > 3 implies |z| > 2), making the stops dead code.
        positions = []
        current_pos = 0
        entry_date = None
        locked_out = False

        for date, row in signals.iterrows():
            if date < test_start:
                positions.append(0)
                continue

            # Time stop (note: (date - entry_date).days counts calendar days,
            # not trading days/bars, so weekends shorten the effective limit)
            if current_pos != 0 and entry_date is not None \
                    and (date - entry_date).days > self.max_holding_days:
                current_pos, entry_date, locked_out = 0, None, True

            # Stop-loss
            if current_pos != 0 and row['stop_loss']:
                current_pos, entry_date, locked_out = 0, None, True

            # Exit on reversion
            if current_pos != 0 and row['exit']:
                current_pos, entry_date = 0, None

            # Release the lockout once the z-score has reverted inside the band
            if locked_out and abs(row['zscore']) < self.entry_threshold:
                locked_out = False

            # Entries
            if current_pos == 0 and not locked_out:
                if row['long_entry']:
                    current_pos, entry_date = 1, date
                elif row['short_entry']:
                    current_pos, entry_date = -1, date

            positions.append(current_pos)

        signals['position'] = positions
        return signals, stats

    def calculate_portfolio_returns(self, stock1: pd.Series, stock2: pd.Series,
                                    signals: pd.DataFrame, hedge_ratio: float) -> pd.DataFrame:
        """Daily strategy returns from the position path."""
        ret1 = stock1.pct_change(fill_method=None)
        ret2 = stock2.pct_change(fill_method=None)

        # Position is shifted one day: today's position earns tomorrow's spread return.
        # Simplified P&L: unit notional in stock1 against hedge_ratio notional in stock2.
        pair_returns = signals['position'].shift(1) * (ret1 - hedge_ratio * ret2)

        portfolio_stats = pd.DataFrame(index=signals.index)
        portfolio_stats['stock1_returns'] = ret1
        portfolio_stats['stock2_returns'] = ret2
        portfolio_stats['pair_returns'] = pair_returns
        portfolio_stats['cumulative_returns'] = (1 + pair_returns.fillna(0)).cumprod()
        portfolio_stats['position'] = signals['position']
        portfolio_stats['spread'] = signals['spread']
        portfolio_stats['zscore'] = signals['zscore']
        return portfolio_stats

    def backtest_pairs(self, stock1: pd.Series, stock2: pd.Series) -> Dict:
        """Backtest the pair; performance is reported on the test window only."""
        signals, spread_stats = self.generate_signals(stock1, stock2)
        hedge_ratio = spread_stats['hedge_ratio']
        portfolio_stats = self.calculate_portfolio_returns(stock1, stock2, signals, hedge_ratio)

        # Evaluate out-of-sample only
        test_start = spread_stats['test_start']
        returns = portfolio_stats.loc[portfolio_stats.index >= test_start, 'pair_returns'].dropna()

        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        risk_free_rate = 0.03  # annualized risk-free rate
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) else 0.0

        # Trade statistics
        positions = signals['position']
        trades = (positions != positions.shift(1)) & (positions != 0)
        num_trades = int(trades.sum())

        # Per-trade returns and win rate
        trade_returns = []
        current_trade_start = None
        pair_ret = portfolio_stats['pair_returns']

        for i, pos in enumerate(positions):
            if pos != 0 and current_trade_start is None:
                current_trade_start = i
            elif pos == 0 and current_trade_start is not None:
                # Include bar i: positions are shifted one day, so the bar on
                # which the position flattens still carries the trade's
                # final-day return.
                trade_returns.append(pair_ret.iloc[current_trade_start:i + 1].sum())
                current_trade_start = None
        if current_trade_start is not None:  # close any trade still open at sample end
            trade_returns.append(pair_ret.iloc[current_trade_start:].sum())

        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0

        return {
            'signals': signals,
            'portfolio_stats': portfolio_stats,
            'spread_stats': spread_stats,
            'performance': {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_trade_return': np.mean(trade_returns) if trade_returns else 0
            }
        }


def demo_pairs_trading():
    """Pairs trading demo on synthetic data for illustration."""
    # Two correlated price series sharing a common stochastic trend
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    common_trend = np.cumsum(np.random.normal(0.0005, 0.01, 500))

    # Stock 1: base price x common trend x idiosyncratic noise
    stock1_specific = np.cumsum(np.random.normal(0, 0.008, 500))
    stock1_prices = 100 * np.exp(common_trend + stock1_specific * 0.3)

    # Stock 2: base price x common trend x idiosyncratic noise
    stock2_specific = np.cumsum(np.random.normal(0, 0.008, 500))
    stock2_prices = 80 * np.exp(common_trend + stock2_specific * 0.3)

    stock1 = pd.Series(stock1_prices, index=dates)
    stock2 = pd.Series(stock2_prices, index=dates)

    strategy = PairsTradingStrategy(
        lookback_period=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        max_holding_days=30,
        train_fraction=0.5   # first half estimates beta, second half is traded
    )

    results = strategy.backtest_pairs(stock1, stock2)

    print("Pairs trading backtest (out-of-sample from "
          f"{results['spread_stats']['test_start'].date()}):")
    for key, value in results['performance'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    pairs_results = demo_pairs_trading()
```

**Example pairs in practice**:

1. **A-share classics**:
   - Ping An Insurance (601318) vs China Life (601628) — insurance
   - Kweichow Moutai (600519) vs Wuliangye (000858) — liquor
   - China Merchants Bank (600036) vs Ping An Bank (000001) — banking

2. **US tech pairs**:
   - Apple (AAPL) vs Microsoft (MSFT)
   - Alphabet (GOOGL) vs Meta (META)
   - Amazon (AMZN) vs Netflix (NFLX)

3. **ETF pairs**:
   - SPY vs QQQ (broad market vs tech)
   - XLF vs XLE (financials vs energy)
   - IWM vs QQQ (small caps vs large-cap tech)

**Pair selection criteria**:

```python
import pandas as pd
from statsmodels.tsa.stattools import coint
from typing import Tuple, Dict


class PairSelectionCriteria:
    """Screening rules for candidate pairs."""

    @staticmethod
    def correlation_test(stock1: pd.Series, stock2: pd.Series, min_corr=0.7) -> bool:
        """Correlation filter."""
        correlation = stock1.corr(stock2)
        return abs(correlation) >= min_corr

    @staticmethod
    def cointegration_test(stock1: pd.Series, stock2: pd.Series,
                           significance=0.05) -> Tuple[bool, float]:
        """Engle-Granger cointegration test."""
        try:
            _, pvalue, _ = coint(stock1, stock2)
            return pvalue < significance, pvalue
        except Exception:
            return False, 1.0

    @staticmethod
    def fundamental_similarity(stock1_info: Dict, stock2_info: Dict) -> float:
        """Fundamental similarity score in [0, 1]."""
        # Industry match
        industry_score = 1.0 if stock1_info.get('industry') == stock2_info.get('industry') else 0.0

        # Market-cap similarity
        mc1, mc2 = stock1_info.get('market_cap', 0), stock2_info.get('market_cap', 0)
        if mc1 > 0 and mc2 > 0:
            size_score = min(mc1, mc2) / max(mc1, mc2)
        else:
            size_score = 0.0

        return industry_score * 0.6 + size_score * 0.4
```

### 2.1.2 ETF Arbitrage

**Strategy rationale**:
Exploit deviations between an ETF's secondary-market price and its net asset value (NAV). ETF arbitrage takes two main forms: primary-market creation/redemption arbitrage and secondary-market spread trading.

**Core mechanics**:
- **Primary market**: large investors can create or redeem ETF shares against a basket of the underlying stocks
- **Secondary market**: ETF shares trade on the exchange like ordinary stocks
- **Arbitrage opportunity**: when the ETF price deviates materially from NAV, buy the cheap side and sell the rich side

```python
import numpy as np
import pandas as pd


class ETFArbitrageStrategy:
    """ETF premium/discount arbitrage."""

    def __init__(self, threshold=0.5, transaction_cost=0.002, min_arbitrage_amount=1000000):
        """
        Args:
            threshold: premium/discount threshold (%)
            transaction_cost: one-way transaction cost (as a fraction)
            min_arbitrage_amount: minimum notional per arbitrage trade
        """
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.min_arbitrage_amount = min_arbitrage_amount

    def calculate_theoretical_nav(self, constituent_data: pd.DataFrame,
                                  weights: pd.Series) -> pd.Series:
        """Theoretical NAV as the weighted sum of constituent prices."""
        aligned_data = constituent_data.reindex(columns=weights.index, fill_value=0)
        return (aligned_data * weights).sum(axis=1)

    def calculate_premium_discount(self, etf_price: pd.Series, nav: pd.Series) -> pd.Series:
        """ETF premium (+) / discount (−) in percent."""
        return (etf_price - nav) / nav * 100

    def calculate_arbitrage_profit(self, premium: float, trade_amount: float) -> float:
        """Arbitrage profit net of round-trip transaction costs."""
        gross_profit = abs(premium) / 100 * trade_amount
        transaction_costs = self.transaction_cost * trade_amount * 2  # buy + sell
        return max(0, gross_profit - transaction_costs)

    def generate_arbitrage_signals(self, etf_data: pd.DataFrame,
                                   constituent_data: pd.DataFrame,
                                   weights: pd.Series) -> pd.DataFrame:
        """Generate arbitrage signals from the premium/discount series."""
        theoretical_nav = self.calculate_theoretical_nav(constituent_data, weights)
        premium = self.calculate_premium_discount(etf_data['price'], theoretical_nav)

        etf_volume = etf_data.get('volume', pd.Series(0, index=etf_data.index))

        signals = pd.DataFrame(index=etf_data.index)
        signals['etf_price'] = etf_data['price']
        signals['theoretical_nav'] = theoretical_nav
        signals['premium'] = premium
        signals['etf_volume'] = etf_volume

        # Trade conditions (the threshold comparison already guarantees a
        # large enough premium, so no separate magnitude filter is needed)
        sufficient_volume = etf_volume > self.min_arbitrage_amount / etf_data['price']

        signals['arbitrage_long'] = (
            (premium < -self.threshold)      # ETF trades below NAV
            & sufficient_volume
        )
        signals['arbitrage_short'] = (
            (premium > self.threshold)       # ETF trades above NAV
            & sufficient_volume
        )

        # Potential profit per opportunity
        signals['potential_profit'] = signals.apply(
            lambda row: self.calculate_arbitrage_profit(
                row['premium'],
                min(row['etf_volume'] * row['etf_price'], self.min_arbitrage_amount)
            ), axis=1
        )

        return signals

    def intraday_arbitrage_strategy(self, etf_data: pd.DataFrame,
                                    constituent_data: pd.DataFrame,
                                    weights: pd.Series,
                                    time_window: str = '5min') -> pd.DataFrame:
        """Intraday variant with wider thresholds around the open and close."""
        # Resample to the target frequency
        etf_resampled = etf_data.resample(time_window).last()
        constituent_resampled = constituent_data.resample(time_window).last()

        signals = self.generate_arbitrage_signals(etf_resampled, constituent_resampled, weights)

        # Flag the open and the run-up to the close
        market_open = signals.index.time == pd.Timestamp('09:30').time()
        market_close = signals.index.time >= pd.Timestamp('14:50').time()

        signals['market_open'] = market_open
        signals['market_close'] = market_close

        # Volatility is higher around the open/close, so require a wider premium
        adjusted_threshold = np.where(
            market_open | market_close,
            self.threshold * 1.5,
            self.threshold
        )

        signals['arbitrage_long'] = (
            (signals['premium'] < -adjusted_threshold)
            & (signals['etf_volume'] > 0)
        )
        signals['arbitrage_short'] = (
            (signals['premium'] > adjusted_threshold)
            & (signals['etf_volume'] > 0)
        )

        return signals

    def risk_management(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Overlay risk controls on the raw arbitrage signals."""
        signals = signals.copy()

        # 1. Limit consecutive trades (avoid over-trading)
        signals['consecutive_trades'] = (
            (signals['arbitrage_long'] | signals['arbitrage_short'])
            .rolling(window=10)
            .sum()
        )

        # 2. Volatility filter: pause arbitrage in the most volatile regimes.
        # The 90th-percentile cutoff is computed on an expanding window of
        # past data only, so the filter is free of look-ahead bias.
        price_volatility = signals['etf_price'].pct_change(fill_method=None).rolling(20).std()
        vol_cutoff = price_volatility.expanding(min_periods=60).quantile(0.9)
        high_volatility = price_volatility > vol_cutoff

        # 3. Apply the controls
        signals['risk_adjusted_long'] = (
            signals['arbitrage_long']
            & (signals['consecutive_trades'] < 3)
            & ~high_volatility
        )
        signals['risk_adjusted_short'] = (
            signals['arbitrage_short']
            & (signals['consecutive_trades'] < 3)
            & ~high_volatility
        )

        return signals


def demo_etf_arbitrage():
    """ETF arbitrage demo on synthetic data for illustration."""
    np.random.seed(42)

    # Two consecutive 390-minute trading sessions (09:30-15:59), built from
    # business days plus intraday offsets, so the open/close flags line up
    # with real session boundaries instead of labelling overnight bars
    session_days = pd.bdate_range('2020-01-02', periods=2)
    dates = pd.DatetimeIndex([
        day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=m)
        for day in session_days for m in range(390)
    ])

    # Five synthetic constituents
    n_stocks = 5
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1],
                        index=[f'stock_{i}' for i in range(n_stocks)])

    # Constituent prices driven by a common factor plus idiosyncratic noise
    base_price = 100
    stock_prices = {}

    for i, stock in enumerate(weights.index):
        common_factor = np.cumsum(np.random.normal(0, 0.001, len(dates)))
        idiosyncratic = np.cumsum(np.random.normal(0, 0.0005, len(dates)))
        prices = base_price * (1 + i * 0.2) * np.exp(common_factor * 0.7 + idiosyncratic * 0.3)
        stock_prices[stock] = prices

    constituent_data = pd.DataFrame(stock_prices, index=dates)

    # ETF price = theoretical NAV plus pricing noise (~0.2% std)
    theoretical_nav = (constituent_data * weights).sum(axis=1)
    noise = np.random.normal(0, 0.002, len(dates))
    etf_prices = theoretical_nav * (1 + noise)

    etf_data = pd.DataFrame({
        'price': etf_prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)

    strategy = ETFArbitrageStrategy(threshold=0.3, transaction_cost=0.001)
    signals = strategy.intraday_arbitrage_strategy(etf_data, constituent_data, weights)
    signals = strategy.risk_management(signals)

    arbitrage_opportunities = (
        signals['risk_adjusted_long'] | signals['risk_adjusted_short']
    ).sum()

    avg_premium = signals['premium'].abs().mean()
    max_potential_profit = signals['potential_profit'].max()

    print("ETF arbitrage analysis:")
    print(f"Number of arbitrage opportunities: {arbitrage_opportunities}")
    print(f"Average absolute premium: {avg_premium:.3f}%")
    print(f"Maximum potential profit: ${max_potential_profit:,.2f}")

    return signals


if __name__ == "__main__":
    etf_results = demo_etf_arbitrage()
```

**Types of ETF arbitrage**:

1. **Cash arbitrage**:
   - Applies to cash-creation ETFs
   - Exploits the gap between creation/redemption prices and market prices
   - Lower risk, steadier returns

2. **In-kind arbitrage**:
   - Requires assembling the basket of constituent stocks
   - Typically offers a larger arbitrage margin
   - Demands more capital and operational capability

3. **Cross-market arbitrage**:
   - Price gaps for the same underlying exposure across markets
   - E.g., a Hong Kong-listed ETF vs an A-share ETF on the same index
   - Currency risk must be managed

## 2.2 Cointegration Strategies

### 2.2.1 Vector Error Correction Model (VECM) Strategy

**Strategy rationale**:
A VECM describes both the short-run dynamics and the long-run equilibrium among several cointegrated series. When prices drift away from the long-run equilibrium, the error-correction mechanism pulls them back — and that predictable pull is the trading signal.

**Key concepts**:
- **Cointegration**: a stable long-run relationship among multiple time series
- **Error-correction term (ECT)**: the current deviation from the long-run equilibrium
- **Adjustment speed (α)**: how quickly prices revert toward equilibrium

**Avoiding look-ahead bias**: at each date the model is fitted on a trailing window that ends the previous day; the resulting cointegration vector is applied to the current price to compute the signal, and the signal is lagged one day before it earns returns. For speed, the model is re-fitted only every `refit_interval` days and the cointegration vector is held fixed in between — a pedagogical simplification; production systems also monitor for cointegration breakdown.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from typing import Tuple, Dict


class VECMStrategy:
    """Trading strategy based on a vector error correction model."""

    def __init__(self, lag_order=1, lookback_window=252):
        """
        Args:
            lag_order: number of lagged differences in the VECM
            lookback_window: trailing window used to fit the model
        """
        self.lag_order = lag_order
        self.lookback_window = lookback_window
        self.model = None

    def test_cointegration(self, price_data: pd.DataFrame, significance_level=0.05) -> Dict:
        """Johansen cointegration test."""
        try:
            result = coint_johansen(price_data.dropna(), det_order=0, k_ar_diff=self.lag_order)

            trace_stats = result.lr1            # trace statistics
            max_eigen_stats = result.lr2        # maximum-eigenvalue statistics
            crit_values_trace = result.cvt      # trace critical values
            crit_values_max_eigen = result.cvm  # max-eigenvalue critical values

            # Number of cointegrating relationships at the chosen level
            significance_idx = {'10%': 0, '5%': 1, '1%': 2}[f'{int(significance_level * 100)}%']

            n_coint_trace = sum(trace_stats > crit_values_trace[:, significance_idx])
            n_coint_max_eigen = sum(max_eigen_stats > crit_values_max_eigen[:, significance_idx])

            return {
                'n_variables': len(price_data.columns),
                'n_coint_trace': n_coint_trace,
                'n_coint_max_eigen': n_coint_max_eigen,
                'trace_stats': trace_stats,
                'max_eigen_stats': max_eigen_stats,
                'eigenvectors': result.evec,
                'eigenvalues': result.eig
            }
        except Exception as e:
            print(f"Cointegration test failed: {e}")
            return None

    def fit_vecm(self, price_data: pd.DataFrame, coint_rank=1) -> Tuple[object, Dict]:
        """Fit a VECM on the given window."""
        try:
            # Determine the cointegration rank first
            coint_test = self.test_cointegration(price_data)
            if coint_test is None:
                return None, {}

            recommended_rank = min(coint_test['n_coint_trace'], coint_test['n_coint_max_eigen'])
            actual_rank = min(coint_rank, recommended_rank, len(price_data.columns) - 1)

            if actual_rank <= 0:
                # No cointegrating relationship found in this window
                return None, {}

            self.model = VECM(
                price_data.dropna(),
                k_ar_diff=self.lag_order,
                coint_rank=actual_rank,
                deterministic='ci'  # constant inside the cointegration relation
            )
            vecm_result = self.model.fit()

            model_info = {
                'alpha': vecm_result.alpha,  # adjustment speeds
                'beta': vecm_result.beta,    # cointegration vectors
                'gamma': vecm_result.gamma,  # short-run coefficients
                'coint_rank': actual_rank,
                'log_likelihood': vecm_result.llf
            }
            return vecm_result, model_info

        except Exception as e:
            print(f"VECM fit failed: {e}")
            return None, {}

    def calculate_error_correction_term(self, price_data: pd.DataFrame,
                                        beta: np.ndarray) -> pd.Series:
        """Error-correction term: beta' * prices (first cointegration vector)."""
        coint_vector = beta[:, 0] if beta.ndim > 1 else beta
        error_correction = price_data.values @ coint_vector
        return pd.Series(error_correction, index=price_data.index)

    def generate_trading_signals(self, price_data: pd.DataFrame,
                                 entry_threshold=1.5, exit_threshold=0.5,
                                 refit_interval=21) -> pd.DataFrame:
        """Generate signals from the z-scored error-correction term.

        At date t the model is fitted on the window [t - lookback, t),
        i.e. on past data only; the signal at t is then applied to the
        return at t+1 in the backtest (via shift(1)).
        """
        n = len(price_data)
        sig_arr = np.zeros(n)
        ect_arr = np.full(n, np.nan)
        z_arr = np.full(n, np.nan)
        alpha_arr = np.full(n, np.nan)

        prev_signal = 0.0
        beta = None
        ect_mean = ect_std = None
        alpha_speed = np.nan

        for i in range(self.lookback_window, n):
            # Re-fit periodically on the trailing window ending at i-1
            if beta is None or (i - self.lookback_window) % refit_interval == 0:
                window_data = price_data.iloc[i - self.lookback_window:i]
                vecm_result, model_info = self.fit_vecm(window_data)
                if vecm_result is not None:
                    beta = model_info['beta']
                    hist_ect = self.calculate_error_correction_term(window_data, beta)
                    ect_mean = hist_ect.mean()
                    ect_std = hist_ect.std()
                    alpha_speed = float(model_info['alpha'][0, 0])
                # If the fit fails, keep the last valid model (if any)

            if beta is None:
                prev_signal = 0.0
                continue

            # Apply the (past-fitted) cointegration vector to today's prices
            ect_now = self.calculate_error_correction_term(price_data.iloc[[i]], beta).iloc[0]
            ect_zscore = (ect_now - ect_mean) / ect_std if ect_std and ect_std > 0 else 0.0

            if ect_zscore > entry_threshold:
                signal = -1.0  # ECT above equilibrium: short the portfolio, expect reversion
            elif ect_zscore < -entry_threshold:
                signal = 1.0   # ECT below equilibrium: long the portfolio, expect reversion
            elif abs(ect_zscore) < exit_threshold:
                signal = 0.0   # near equilibrium: flat
            else:
                signal = prev_signal  # in between: hold the current position

            sig_arr[i] = signal
            ect_arr[i] = ect_now
            z_arr[i] = ect_zscore
            alpha_arr[i] = alpha_speed
            prev_signal = signal

        return pd.DataFrame({
            'signal': sig_arr,
            'ect': ect_arr,
            'ect_zscore': z_arr,
            'alpha_speed': alpha_arr
        }, index=price_data.index)

    def portfolio_weights_from_cointegration(self, beta: np.ndarray,
                                             normalize=True) -> np.ndarray:
        """Portfolio weights from the first cointegration vector."""
        weights = beta[:, 0] if beta.ndim > 1 else beta
        if normalize:
            weights = weights / np.sum(np.abs(weights))
        return weights

    def backtest_vecm_strategy(self, price_data: pd.DataFrame) -> Dict:
        """Backtest the VECM signal."""
        signals = self.generate_trading_signals(price_data)

        returns = price_data.pct_change(fill_method=None)

        # Pedagogical simplification: trade an equally weighted basket. Note
        # that the signal is the z-scored ECT of the *cointegration* portfolio,
        # so trading the equal-weight basket severs the direct link between the
        # signal and the instrument actually traded; a production
        # implementation trades the cointegration weights instead
        # (see portfolio_weights_from_cointegration).
        portfolio_returns = returns.mean(axis=1)

        # Lag the signal one day so it only uses information available at entry
        strategy_returns = signals['signal'].shift(1) * portfolio_returns

        total_return = (1 + strategy_returns).prod() - 1
        annual_return = strategy_returns.mean() * 252
        volatility = strategy_returns.std() * np.sqrt(252)
        risk_free_rate = 0.03  # annualized risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + strategy_returns.fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'signals': signals,
            'strategy_returns': strategy_returns,
            'performance': {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }


def demo_vecm_strategy():
    """VECM strategy demo on synthetic data for illustration."""
    # Three cointegrated price series sharing a common stochastic trend
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    common_trend = np.cumsum(np.random.normal(0, 0.01, 500))

    # Long-run cointegration with short-run deviations
    stock1 = 100 * np.exp(common_trend + np.cumsum(np.random.normal(0, 0.005, 500)))
    stock2 = 80 * np.exp(common_trend * 1.2 + np.cumsum(np.random.normal(0, 0.005, 500)))
    stock3 = 120 * np.exp(common_trend * 0.8 + np.cumsum(np.random.normal(0, 0.005, 500)))

    price_data = pd.DataFrame({
        'stock1': stock1,
        'stock2': stock2,
        'stock3': stock3
    }, index=dates)

    strategy = VECMStrategy(lag_order=1, lookback_window=120)
    results = strategy.backtest_vecm_strategy(price_data)

    print("VECM strategy backtest:")
    for key, value in results['performance'].items():
        print(f"{key}: {value:.4f}")

    return results


if __name__ == "__main__":
    vecm_results = demo_vecm_strategy()
```
