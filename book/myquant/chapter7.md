# Chapter 7: Volatility Strategies

Volatility is the one dimension of markets that is genuinely forecastable. Returns are close to unpredictable day to day, but their *magnitude* clusters: turbulent days follow turbulent days, calm follows calm. That persistence, plus the fact that options let you trade volatility directly, supports a whole family of strategies: forecasting realized volatility with GARCH-type models, trading mispricings on the implied volatility surface, and harvesting the variance risk premium.

## 7.1 GARCH Volatility Forecasting

**Strategy rationale.**
GARCH(p, q) models capture volatility clustering by letting today's conditional variance depend on recent squared shocks (the ARCH terms) and on its own recent values (the GARCH terms). A fitted model yields a one-step-ahead volatility forecast at each date; comparing that forecast to current realized volatility produces signals — e.g. scale down position size when forecast volatility spikes (volatility targeting), or trade the gap when forecast and realized volatility diverge.

The rolling loop below refits the model at each step using **only the trailing window**, so forecasts never see future data. Requires the `arch` package (`pip install arch`).

```python
import numpy as np
import pandas as pd
from arch import arch_model


class GARCHVolatilityStrategy:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None

    def fit_garch_model(self, returns):
        """Fit a GARCH(p, q) model to a return series."""
        # Drop missing values and clip extreme outliers (> 5 sigma)
        returns_clean = returns.dropna()
        returns_clean = returns_clean[
            np.abs(returns_clean) < returns_clean.std() * 5
        ]

        # Returns are scaled by 100: the optimizer is numerically more
        # stable when variance is not tiny
        model = arch_model(returns_clean * 100, vol='Garch', p=self.p, q=self.q)
        self.model = model.fit(disp='off')
        return self.model

    def forecast_volatility(self, horizon=1):
        """Forecast volatility `horizon` steps ahead."""
        forecast = self.model.forecast(horizon=horizon)
        # Undo the x100 scaling: variance scales by 100^2
        predicted_variance = forecast.variance.iloc[-1, :] / 10000
        return np.sqrt(predicted_variance)

    def calculate_volatility_signals(self, returns, lookback=252):
        """Generate trading signals from rolling GARCH forecasts.

        At each date t, the model is fit on returns from t-lookback to
        t-1 and forecasts volatility for date t — no look-ahead.
        """
        signals = pd.DataFrame(index=returns.index)
        volatility_forecasts = []
        realized_volatilities = []

        for i in range(lookback, len(returns)):
            window_returns = returns.iloc[i - lookback:i]

            try:
                self.fit_garch_model(window_returns)
                vol_forecast = self.forecast_volatility(horizon=1).iloc[0]
                volatility_forecasts.append(vol_forecast)

                # Trailing 20-day realized volatility for comparison
                realized_vol = window_returns.iloc[-20:].std()
                realized_volatilities.append(realized_vol)

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Optimizer failures on degenerate windows: record NaN
                # and move on rather than aborting the whole backtest
                volatility_forecasts.append(np.nan)
                realized_volatilities.append(np.nan)

        # Pad the warm-up period so the series align with the index
        volatility_forecasts = [np.nan] * lookback + volatility_forecasts
        realized_volatilities = [np.nan] * lookback + realized_volatilities

        signals['vol_forecast'] = volatility_forecasts
        signals['vol_realized'] = realized_volatilities
        signals['vol_ratio'] = signals['vol_forecast'] / signals['vol_realized']

        # Regime signals from the trailing distribution of forecasts
        signals['vol_breakout'] = (
            signals['vol_forecast']
            > signals['vol_forecast'].rolling(60).quantile(0.8)
        )
        signals['vol_mean_revert'] = (
            signals['vol_forecast']
            < signals['vol_forecast'].rolling(60).quantile(0.2)
        )

        return signals
```

**Practical notes.**

- Refitting on every bar is expensive (a full maximum-likelihood estimation per day). In practice, refit weekly or monthly and only update the forecast daily — parameter estimates move slowly.
- Plain GARCH treats positive and negative shocks symmetrically. Equity volatility reacts more to drawdowns (the leverage effect); GJR-GARCH (`o=1` in `arch_model`) or EGARCH usually forecast equity volatility better.
- The most robust production use of these forecasts is not directional trading but **volatility targeting**: size positions inversely to forecast volatility so the portfolio runs at roughly constant risk.

## 7.2 The Implied Volatility Surface

**Strategy rationale.**
While GARCH forecasts volatility from historical returns, option prices reveal the market's *implied* volatility. Inverting the Black-Scholes formula for each strike and expiry produces the **implied volatility surface**. Its shape — the skew across strikes and the term structure across expiries — encodes market expectations, and distortions in that shape (put-call IV gaps, kinked skews) can flag relative-value trades.

```python
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


class VolatilitySurfaceStrategy:
    def __init__(self):
        self.vol_surface = {}

    def calculate_implied_volatility(self, option_prices, stock_price, strike_prices,
                                     time_to_expiry, risk_free_rate, option_type='call'):
        """Back out implied volatility from market option prices."""

        def black_scholes_price(vol, strike):
            """Black-Scholes price for a European option."""
            d1 = (np.log(stock_price / strike)
                  + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry) \
                / (vol * np.sqrt(time_to_expiry))
            d2 = d1 - vol * np.sqrt(time_to_expiry)

            if option_type == 'call':
                price = (stock_price * norm.cdf(d1)
                         - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:
                price = (strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                         - stock_price * norm.cdf(-d1))
            return price

        implied_vols = []
        for i, market_price in enumerate(option_prices):
            try:
                strike = strike_prices[i]

                def objective(vol, k=strike):
                    return black_scholes_price(vol, k) - market_price

                # Root-find IV in [1%, 500%]; fails if the market price
                # violates no-arbitrage bounds
                iv = brentq(objective, 0.01, 5.0)
                implied_vols.append(iv)
            except ValueError:
                implied_vols.append(np.nan)

        return np.array(implied_vols)

    def build_volatility_surface(self, option_data):
        """Build an IV surface from an option chain.

        `option_data` needs columns: expiry, strike, call_price, put_price,
        underlying_price, time_to_expiry, risk_free_rate.
        """
        vol_surface = {}

        for expiry in option_data['expiry'].unique():
            # Sort by strike: np.interp below requires ascending x-values
            expiry_data = (option_data[option_data['expiry'] == expiry]
                           .sort_values('strike'))

            strikes = expiry_data['strike'].values
            call_prices = expiry_data['call_price'].values
            put_prices = expiry_data['put_price'].values

            stock_price = expiry_data['underlying_price'].iloc[0]
            time_to_expiry = expiry_data['time_to_expiry'].iloc[0]
            risk_free_rate = expiry_data['risk_free_rate'].iloc[0]

            call_ivs = self.calculate_implied_volatility(
                call_prices, stock_price, strikes, time_to_expiry, risk_free_rate, 'call'
            )
            put_ivs = self.calculate_implied_volatility(
                put_prices, stock_price, strikes, time_to_expiry, risk_free_rate, 'put'
            )

            # ATM IV by linear interpolation. Precondition: strikes are
            # sorted ascending (done above) and NaN IVs (failed root-finds)
            # are dropped — np.interp silently returns nonsense otherwise.
            valid = ~np.isnan(call_ivs)
            atm_iv = (np.interp(stock_price, strikes[valid], call_ivs[valid])
                      if valid.any() else np.nan)

            vol_surface[expiry] = {
                'strikes': strikes,
                'call_ivs': call_ivs,
                'put_ivs': put_ivs,
                'atm_strike': strikes[np.argmin(np.abs(strikes - stock_price))],
                'atm_iv': atm_iv,
            }

        return vol_surface

    def detect_volatility_arbitrage(self, vol_surface):
        """Flag potential relative-value distortions on the surface."""
        arbitrage_signals = []

        for expiry, surface_data in vol_surface.items():
            strikes = surface_data['strikes']
            call_ivs = surface_data['call_ivs']
            put_ivs = surface_data['put_ivs']

            # Put-call parity check: call and put IVs at the same strike
            # should match for European options
            iv_spread = call_ivs - put_ivs
            abnormal_spread = np.abs(iv_spread) > 0.02  # 2 vol-point threshold

            # Skew anomaly: compare local slope on each side of ATM. This
            # needs a neighbor strike on BOTH sides, so it is skipped when
            # the ATM strike sits at the edge of the strike grid — but the
            # parity-spread signal above does not depend on it and is
            # recorded either way.
            atm_strike_idx = np.argmin(np.abs(strikes - surface_data['atm_strike']))

            abnormal_skew = None  # not computable at the grid edge
            if 0 < atm_strike_idx < len(call_ivs) - 1:
                left_skew = call_ivs[atm_strike_idx] - call_ivs[atm_strike_idx - 1]
                right_skew = call_ivs[atm_strike_idx + 1] - call_ivs[atm_strike_idx]
                abnormal_skew = abs(left_skew - right_skew) > 0.05

            arbitrage_signals.append({
                'expiry': expiry,
                'abnormal_spread': abnormal_spread.any(),
                'abnormal_skew': abnormal_skew,
                'max_spread': np.nanmax(np.abs(iv_spread)),
            })

        return arbitrage_signals
```

**Caveats.** A call-put IV gap is not free money: for American options, or when dividends and stock-borrow costs are material, put-call parity holds only approximately, and much of the apparent spread is those carry costs rather than mispricing. Always check whether a "signal" survives the bid-ask spread — options quotes are wide precisely where surfaces look most distorted.

## 7.3 The Variance Risk Premium

**Strategy rationale.**
Averaged over long samples, index implied volatility exceeds the realized volatility that subsequently materializes. Option buyers systematically overpay for variance because options are insurance against crashes, and insurance carries a premium. The **variance risk premium (VRP)** is that gap:

```
VRP_t = IV_t^2  −  E_t[RV_{t→t+h}^2]
```

i.e. implied variance minus expected future realized variance. Selling variance — via short straddles, short VIX futures, or variance swaps — collects this premium on average, in exchange for rare, violent losses when volatility spikes. It is the option-market analogue of selling insurance: steady income punctuated by catastrophes, so position sizing and tail hedging matter more than the signal itself.

Two measurement details are easy to get wrong:

- **Ex-ante vs. ex-post.** A *tradable* signal at time t may use only IV at t and realized variance measured over the *past* h days. Subtracting *future* realized variance gives the ex-post premium — useful for evaluation, never for signal construction (look-ahead bias).
- **Variance, not volatility.** The premium is defined in variance units (vol squared); comparing vols directly understates the effect because of Jensen's inequality.

```python
import numpy as np
import pandas as pd


class VarianceRiskPremiumStrategy:
    def __init__(self, realized_window=21, ann_factor=252):
        self.realized_window = realized_window  # ~1 month of trading days
        self.ann_factor = ann_factor

    def realized_variance(self, returns):
        """Trailing annualized realized variance (backward-looking)."""
        return returns.pow(2).rolling(self.realized_window).sum() \
            * (self.ann_factor / self.realized_window)

    def compute_signals(self, returns, implied_vol):
        """Ex-ante VRP signal: implied variance minus TRAILING realized
        variance. Uses only information available at each date.

        `implied_vol` is an annualized IV series (e.g. a VIX-style index
        divided by 100), aligned on the same dates as `returns`.
        """
        df = pd.DataFrame({'returns': returns, 'implied_vol': implied_vol}).dropna()

        df['implied_var'] = df['implied_vol'] ** 2
        df['realized_var'] = self.realized_variance(df['returns'])
        df['vrp'] = df['implied_var'] - df['realized_var']

        # Short variance when the premium is comfortably positive
        # (its usual state); stand aside when it inverts, which tends
        # to happen exactly when short-vol positions are bleeding
        df['short_vol_signal'] = (df['vrp'] > 0).astype(int)

        # Ex-post premium actually earned over the NEXT window — the
        # shift(-window) makes this forward-looking, so it is for
        # EVALUATION ONLY, never a tradable input
        future_rv = df['realized_var'].shift(-self.realized_window)
        df['vrp_expost'] = df['implied_var'] - future_rv

        return df


# --- Demo on synthetic data for illustration ---
# Simulate returns with volatility clustering, and an implied vol series
# that quotes trailing realized vol plus a positive premium.
rng = np.random.default_rng(7)
n = 1500

vol = np.zeros(n)
vol[0] = 0.01
for t in range(1, n):  # simple stochastic volatility process
    vol[t] = 0.94 * vol[t - 1] + 0.06 * 0.01 + 0.002 * abs(rng.standard_normal())
returns = pd.Series(rng.standard_normal(n) * vol,
                    index=pd.bdate_range('2020-01-01', periods=n))

trailing_vol_ann = returns.rolling(21).std() * np.sqrt(252)
implied_vol = trailing_vol_ann * 1.15 + rng.normal(0, 0.01, n)  # ~15% premium

strategy = VarianceRiskPremiumStrategy()
result = strategy.compute_signals(returns, implied_vol).dropna()

print(f"Mean ex-ante VRP:  {result['vrp'].mean():.4f}")
print(f"Mean ex-post VRP:  {result['vrp_expost'].mean():.4f}")
print(f"Share of days short-vol signal is on: {result['short_vol_signal'].mean():.1%}")
```

**Practical notes.** On real index data, use a model-free implied variance measure (the VIX squared, for the S&P 500) rather than a single ATM option's IV — that is what variance-swap replication actually prices. And respect the payoff asymmetry: historical VRP strategies show high Sharpe ratios with occasional drawdowns of many years of accumulated premium (February 2018 and March 2020 are the canonical examples). Sizing as if the premium were a normal alpha is how short-vol funds blow up.
