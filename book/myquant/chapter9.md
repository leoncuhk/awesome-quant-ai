# Chapter 9: Macro Strategies

Macro strategies trade the broad drivers of asset prices — interest rates, currencies, inflation, and growth — rather than individual securities. Their signals come from the shape of the yield curve, cross-country interest rate and inflation differentials, and shifts in volatility regimes. This chapter implements two representative families: interest rate strategies built on yield-curve level, slope, and curvature, and currency strategies built on carry, purchasing power parity, and regime-dependent trading rules. All rolling statistics use only past data, so the signals are free of look-ahead bias.

## 9.1 Interest Rate Strategies

The yield curve can be summarized by three factors: level (the average of yields), slope (long minus short yields), and curvature (the belly relative to the wings). Slope trades bet on steepening or flattening, level trades bet on mean reversion of overall rates, and butterfly trades bet on the curvature normalizing. The class below computes these indicators, converts them into z-score and percentile-based signals, and constructs a simple duration-targeted portfolio from a short-maturity and a long-maturity bond.

```python
import pandas as pd
import numpy as np


class InterestRateStrategy:
    def __init__(self):
        self.yield_curve_data = {}

    def calculate_yield_curve_indicators(self, yield_data):
        """Compute yield-curve level, slope, and curvature indicators.

        `yield_data` is a DataFrame with tenor columns such as
        '3M', '2Y', '5Y', '10Y', '30Y'.
        """
        indicators = pd.DataFrame(index=yield_data.index)

        # Slope: long-end yield minus short-end yield
        indicators['slope_2y10y'] = yield_data['10Y'] - yield_data['2Y']
        indicators['slope_3m10y'] = yield_data['10Y'] - yield_data['3M']
        indicators['slope_5y30y'] = yield_data['30Y'] - yield_data['5Y']

        # Curvature: wings minus twice the belly (butterfly)
        indicators['curvature'] = (yield_data['2Y'] + yield_data['10Y']) - 2 * yield_data['5Y']

        # Yield changes and rates of change per tenor
        for tenor in ['3M', '2Y', '5Y', '10Y', '30Y']:
            if tenor in yield_data.columns:
                indicators[f'{tenor}_change'] = yield_data[tenor].diff()
                indicators[f'{tenor}_roc'] = yield_data[tenor].pct_change()

        # Level: average of the belly of the curve
        indicators['level'] = yield_data[['2Y', '5Y', '10Y']].mean(axis=1)

        return indicators

    def yield_curve_trading_signals(self, yield_indicators):
        """Generate trading signals from yield-curve indicators."""
        signals = pd.DataFrame(index=yield_indicators.index)

        # Slope signals: compare the current slope to its one-year band
        slope_mean = yield_indicators['slope_2y10y'].rolling(252).mean()
        slope_std = yield_indicators['slope_2y10y'].rolling(252).std()

        # Steepening / flattening when the slope breaks out of the band
        signals['steepening'] = yield_indicators['slope_2y10y'] > slope_mean + slope_std
        signals['flattening'] = yield_indicators['slope_2y10y'] < slope_mean - slope_std

        # Level signals: rolling percentile of the overall rate level
        level_percentile = yield_indicators['level'].rolling(252).rank(pct=True)
        signals['rates_low'] = level_percentile < 0.2
        signals['rates_high'] = level_percentile > 0.8

        # Curvature signals: butterfly trades on extreme z-scores
        curvature_zscore = (
            (yield_indicators['curvature'] -
             yield_indicators['curvature'].rolling(252).mean()) /
            yield_indicators['curvature'].rolling(252).std()
        )
        signals['butterfly_long'] = curvature_zscore < -2   # curve unusually humped
        signals['butterfly_short'] = curvature_zscore > 2   # curve unusually dished

        return signals

    def duration_hedged_strategy(self, bond_prices, duration_data, target_duration=5):
        """Build a two-bond portfolio matching a target duration.

        Combines a short-maturity bond (2Y) and a long-maturity bond (10Y)
        so that the weighted portfolio duration equals `target_duration`.
        """
        portfolio_weights = pd.DataFrame(index=bond_prices.index)

        short_duration = duration_data['2Y_duration']
        long_duration = duration_data['10Y_duration']

        # Solve w_long * D_long + (1 - w_long) * D_short = target_duration
        w_long = (target_duration - short_duration) / (long_duration - short_duration)
        w_short = 1 - w_long

        portfolio_weights['long_bond'] = w_long
        portfolio_weights['short_bond'] = w_short

        return portfolio_weights
```

## 9.2 Currency Strategies

Currency markets offer several persistent, economically motivated effects. The carry trade earns the interest rate differential by holding high-yield currencies against low-yield ones — combining it with momentum helps avoid carry crashes. Purchasing power parity (PPP) says exchange rates should offset inflation differentials over the long run, so large deviations create mean-reversion opportunities. Finally, FX behavior differs by volatility regime: mean reversion tends to work in turbulent markets, momentum in quiet ones. Currency pair columns follow the standard six-letter convention (e.g. 'EURUSD' = base 'EUR', quote 'USD').

```python
import pandas as pd
import numpy as np


class CurrencyStrategy:
    def __init__(self):
        self.currency_pairs = {}

    def carry_trade_strategy(self, fx_data, interest_rate_data):
        """Carry trade: hold the higher-yielding currency, filtered by momentum."""
        signals = pd.DataFrame(index=fx_data.index)

        for pair in fx_data.columns:
            base_currency = pair[:3]
            quote_currency = pair[3:]

            if (base_currency in interest_rate_data.columns and
                    quote_currency in interest_rate_data.columns):
                # Interest rate differential (base minus quote)
                interest_diff = (interest_rate_data[base_currency] -
                                 interest_rate_data[quote_currency])

                # Raw carry signal: long when the differential exceeds 1%
                signals[f'{pair}_carry'] = np.where(
                    interest_diff > 0.01, 1,
                    np.where(interest_diff < -0.01, -1, 0)
                )

                # 20-day exchange rate momentum as a confirmation filter
                fx_momentum = fx_data[pair].pct_change(20)

                # Combined signal: trade carry only when momentum agrees
                signals[f'{pair}_combined'] = np.where(
                    (signals[f'{pair}_carry'] == 1) & (fx_momentum > 0), 1,
                    np.where((signals[f'{pair}_carry'] == -1) & (fx_momentum < 0), -1, 0)
                )

        return signals

    def purchasing_power_parity_strategy(self, fx_data, inflation_data):
        """PPP mean reversion: fade large deviations from inflation-implied fair value.

        Unit assumption: `inflation_data` must contain *per-period* inflation
        rates sampled at the same frequency as `fx_data` (e.g. daily rates for
        a daily FX index), so that `cumsum()` gives the cumulative inflation
        differential. Passing annualized figures such as CPI YoY on a daily
        index would overstate the PPP-implied move by roughly the
        annualization factor (~250x for daily data); convert them first,
        e.g. `daily_rate = (1 + annual_rate)**(1/252) - 1`.
        """
        signals = pd.DataFrame(index=fx_data.index)

        for pair in fx_data.columns:
            base_currency = pair[:3]
            quote_currency = pair[3:]

            if (base_currency in inflation_data.columns and
                    quote_currency in inflation_data.columns):
                # Relative inflation differential
                inflation_diff = (inflation_data[base_currency] -
                                  inflation_data[quote_currency])

                # PPP-implied cumulative exchange rate change
                theoretical_fx_change = inflation_diff.cumsum()

                # Actual (log) exchange rate change since the start of the sample
                actual_fx_change = np.log(fx_data[pair] / fx_data[pair].iloc[0])

                # Deviation from PPP fair value
                ppp_deviation = actual_fx_change - theoretical_fx_change

                # Mean-reversion signal on the deviation z-score
                ppp_zscore = (
                    (ppp_deviation - ppp_deviation.rolling(252).mean()) /
                    ppp_deviation.rolling(252).std()
                )

                signals[f'{pair}_ppp_signal'] = np.where(
                    ppp_zscore > 2, -1,               # overvalued: short
                    np.where(ppp_zscore < -2, 1, 0)   # undervalued: long
                )

        return signals

    def volatility_regime_strategy(self, fx_data, vol_threshold=0.01):
        """Switch between mean reversion and momentum by volatility regime."""
        signals = pd.DataFrame(index=fx_data.index)

        for pair in fx_data.columns:
            # Rolling realized volatility
            returns = fx_data[pair].pct_change()
            rolling_vol = returns.rolling(20).std()

            # Regime identification via rolling volatility quantiles
            high_vol_regime = rolling_vol > rolling_vol.rolling(252).quantile(0.75)
            low_vol_regime = rolling_vol < rolling_vol.rolling(252).quantile(0.25)

            # High-volatility regime: mean reversion (fade recent moves)
            mean_revert_signal = np.where(returns.rolling(5).mean() > 0, -1, 1)

            # Low-volatility regime: momentum (follow recent moves)
            momentum_signal = np.where(returns.rolling(10).mean() > 0, 1, -1)

            # Combined regime-dependent signal; flat in the middle regime
            signals[f'{pair}_regime'] = np.where(
                high_vol_regime, mean_revert_signal,
                np.where(low_vol_regime, momentum_signal, 0)
            )

        return signals
```
