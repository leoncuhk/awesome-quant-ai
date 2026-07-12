# Chapter 6: Fundamental Quantitative Strategies

Fundamental quant sits between discretionary stock picking and purely price-driven signals: it takes the inputs a fundamental analyst would use — valuations, profitability, growth, balance-sheet quality — and applies them systematically across a wide cross-section of stocks. The two workhorses of this style are **factor models**, which explain and forecast returns with a small set of common drivers, and **event-driven signals**, which exploit predictable price behavior around discrete corporate events.

## 6.1 Multi-Factor Models

### 6.1.1 The Fama-French Five-Factor Model

**Strategy rationale.**
The Fama-French five-factor model (2015) explains cross-sectional stock returns with five systematic factors:

| Factor | Name | Long / short legs |
|---|---|---|
| MKT | Market | Market portfolio minus the risk-free rate |
| SMB | Size (Small Minus Big) | Small-cap stocks minus large-cap stocks |
| HML | Value (High Minus Low) | High book-to-market minus low book-to-market |
| RMW | Profitability (Robust Minus Weak) | High-ROE firms minus low-ROE firms |
| CMA | Investment (Conservative Minus Aggressive) | Low asset growth minus high asset growth |

A strategy built on this model can be used two ways: (1) estimate each stock's **alpha** — the return not explained by the five factors — and buy stocks with persistently positive alpha; (2) tilt the portfolio toward factor exposures that have historically carried a premium (small, cheap, profitable, conservatively investing firms).

The implementation below expects `stock_data` as a long-format DataFrame with columns `date`, `stock_id`, `returns`, `market_cap`, `book_to_market`, `roe`, `asset_growth`, and `market_data` as a DataFrame indexed by date with a `returns` column for the market portfolio.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm


class FamaFrenchStrategy:
    def __init__(self, annual_risk_free_rate=0.03, min_obs=60):
        self.risk_free_daily = annual_risk_free_rate / 252
        self.min_obs = min_obs  # minimum observations required per regression

    def calculate_market_factor(self, market_returns):
        """Market factor (MKT): market return minus the risk-free rate."""
        return market_returns - self.risk_free_daily

    def calculate_size_factor(self, stock_data):
        """Size factor (SMB) - Small Minus Big.

        Note: simplified implementation. The standard Fama-French
        construction uses 2x3 double sorts (size x value). Here we split
        each date's cross-section at the median market cap and return the
        long-short spread as a time series indexed by date.
        """
        def _daily_smb(group):
            median_cap = group['market_cap'].median()
            small = group[group['market_cap'] <= median_cap]
            big = group[group['market_cap'] > median_cap]
            return small['returns'].mean() - big['returns'].mean()

        return stock_data.groupby('date')[['market_cap', 'returns']].apply(_daily_smb)

    def calculate_value_factor(self, stock_data):
        """Value factor (HML) - High Minus Low.

        Splits each date's cross-section at the 30th/70th percentiles of
        book-to-market and returns the long-short spread time series.
        """
        def _daily_hml(group):
            q70 = group['book_to_market'].quantile(0.7)
            q30 = group['book_to_market'].quantile(0.3)
            high_bm = group[group['book_to_market'] >= q70]
            low_bm = group[group['book_to_market'] <= q30]
            return high_bm['returns'].mean() - low_bm['returns'].mean()

        return stock_data.groupby('date')[['book_to_market', 'returns']].apply(_daily_hml)

    def calculate_profitability_factor(self, stock_data):
        """Profitability factor (RMW) - Robust Minus Weak.

        Splits each date's cross-section at the 30th/70th percentiles of ROE.
        """
        def _daily_rmw(group):
            q70 = group['roe'].quantile(0.7)
            q30 = group['roe'].quantile(0.3)
            robust = group[group['roe'] >= q70]
            weak = group[group['roe'] <= q30]
            return robust['returns'].mean() - weak['returns'].mean()

        return stock_data.groupby('date')[['roe', 'returns']].apply(_daily_rmw)

    def calculate_investment_factor(self, stock_data):
        """Investment factor (CMA) - Conservative Minus Aggressive.

        Splits each date's cross-section at the 30th/70th percentiles of
        asset growth. Conservative (low-growth) firms are the long leg.
        """
        def _daily_cma(group):
            q30 = group['asset_growth'].quantile(0.3)
            q70 = group['asset_growth'].quantile(0.7)
            conservative = group[group['asset_growth'] <= q30]
            aggressive = group[group['asset_growth'] >= q70]
            return conservative['returns'].mean() - aggressive['returns'].mean()

        return stock_data.groupby('date')[['asset_growth', 'returns']].apply(_daily_cma)

    def build_factor_model(self, stock_data, market_data):
        """Estimate the five-factor regression for every stock.

        All factor series and per-stock return series are aligned
        explicitly on dates before the regression; rows with missing
        values on any regressor are dropped.
        """
        # Assemble the factor panel, indexed by date
        factors = pd.DataFrame({
            'MKT': self.calculate_market_factor(market_data['returns']),
            'SMB': self.calculate_size_factor(stock_data),
            'HML': self.calculate_value_factor(stock_data),
            'RMW': self.calculate_profitability_factor(stock_data),
            'CMA': self.calculate_investment_factor(stock_data),
        })

        results = {}
        for stock, group in stock_data.groupby('stock_id'):
            # Excess returns for this stock, indexed by date
            stock_returns = (
                group.set_index('date')['returns'] - self.risk_free_daily
            ).rename('excess_ret')

            # Explicit date alignment: inner-join returns against factors
            merged = factors.join(stock_returns, how='inner').dropna()
            if len(merged) < self.min_obs:
                continue  # not enough overlapping history for a stable fit

            X = sm.add_constant(merged[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])
            model = sm.OLS(merged['excess_ret'], X).fit()

            results[stock] = {
                'alpha': model.params['const'],
                'beta_mkt': model.params['MKT'],
                'beta_smb': model.params['SMB'],
                'beta_hml': model.params['HML'],
                'beta_rmw': model.params['RMW'],
                'beta_cma': model.params['CMA'],
                'alpha_tstat': model.tvalues['const'],
                'r_squared': model.rsquared,
            }

        return results

    def generate_signals(self, factor_results):
        """Generate stock-selection signals from estimated loadings."""
        signals = pd.DataFrame(
            index=list(factor_results.keys()),
            columns=['alpha_signal', 'factor_score', 'final_signal'],
            dtype=float,
        )

        for stock, factors in factor_results.items():
            # Alpha-based selection: positive alpha with a reasonable fit
            alpha_ok = factors['alpha'] > 0 and factors['r_squared'] > 0.3
            signals.loc[stock, 'alpha_signal'] = 1.0 if alpha_ok else 0.0

            # Loading-based selection: reward exposure to premia that
            # have historically been positive
            score = 0
            if factors['beta_smb'] > 0:  # small-cap tilt
                score += 1
            if factors['beta_hml'] > 0:  # value tilt
                score += 1
            if factors['beta_rmw'] > 0:  # profitability tilt
                score += 1
            if factors['beta_cma'] > 0:  # conservative-investment tilt
                score += 1

            signals.loc[stock, 'factor_score'] = score
            signals.loc[stock, 'final_signal'] = 1.0 if score >= 3 else 0.0

        return signals
```

**Practical notes.**

- The regressions are estimated over the full sample here for clarity. In production you should estimate loadings on a **rolling window** and use only data available at the rebalance date, otherwise stock selection quietly conditions on future information.
- Alpha estimates are noisy; the `alpha_tstat` field is included so you can require statistical significance (e.g. |t| > 2) rather than just a positive point estimate.
- The simplified median/tercile factor construction is fine for teaching; if you need research-grade factors for US equities, use the pre-built series from Kenneth French's data library.

### 6.1.2 A Custom Multi-Factor Framework

The five-factor model constrains you to five academic factors. A practical multi-factor stock-selection system typically combines a broader library — valuation, growth, quality, momentum, liquidity — through a standard pipeline: **compute raw factors → neutralize → score → weight**.

```python
import numpy as np
import pandas as pd


class CustomMultiFactorStrategy:
    def __init__(self):
        self.factor_weights = {}

    def calculate_fundamental_factors(self, financial_data):
        """Fundamental factors from (point-in-time) financial statements.

        `financial_data` is a quarterly DataFrame for one stock, or a
        panel processed per stock. To avoid look-ahead bias, statement
        data must be lagged to its public release date, not the fiscal
        period end date.
        """
        factors = pd.DataFrame(index=financial_data.index)

        # Valuation factors
        factors['PE'] = financial_data['market_cap'] / financial_data['net_income']
        factors['PB'] = financial_data['market_cap'] / financial_data['book_value']
        factors['PS'] = financial_data['market_cap'] / financial_data['revenue']
        factors['PCF'] = financial_data['market_cap'] / financial_data['operating_cashflow']
        factors['EV_EBITDA'] = financial_data['enterprise_value'] / financial_data['ebitda']

        # A loss-maker has negative earnings and hence a negative PE, which
        # the inverted rank in factor_scoring would crown as the "cheapest"
        # stock. Mask ratios with non-positive denominators to NaN so they
        # drop out of the ranking instead of topping the valuation score.
        for ratio, denominator in [('PE', 'net_income'), ('PB', 'book_value'),
                                   ('PS', 'revenue'), ('PCF', 'operating_cashflow'),
                                   ('EV_EBITDA', 'ebitda')]:
            factors.loc[financial_data[denominator] <= 0, ratio] = np.nan

        # Growth factors (year-over-year on quarterly data)
        factors['revenue_growth'] = financial_data['revenue'].pct_change(periods=4, fill_method=None)
        factors['earnings_growth'] = financial_data['net_income'].pct_change(periods=4, fill_method=None)
        factors['book_value_growth'] = financial_data['book_value'].pct_change(periods=4, fill_method=None)
        factors['roa_change'] = financial_data['roa'].diff(periods=4)
        factors['roe_change'] = financial_data['roe'].diff(periods=4)

        # Quality factors
        factors['roa'] = financial_data['net_income'] / financial_data['total_assets']
        factors['roe'] = financial_data['net_income'] / financial_data['shareholders_equity']
        factors['gross_margin'] = financial_data['gross_profit'] / financial_data['revenue']
        factors['operating_margin'] = financial_data['operating_income'] / financial_data['revenue']
        factors['debt_to_equity'] = financial_data['total_debt'] / financial_data['shareholders_equity']
        factors['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']
        factors['asset_turnover'] = financial_data['revenue'] / financial_data['total_assets']

        # Cash-flow factors
        factors['fcf_yield'] = financial_data['free_cashflow'] / financial_data['market_cap']
        factors['ocf_to_sales'] = financial_data['operating_cashflow'] / financial_data['revenue']
        factors['capex_to_sales'] = financial_data['capex'] / financial_data['revenue']

        return factors

    def calculate_technical_factors(self, price_data):
        """Price-based factors (momentum, reversal, volatility, liquidity)."""
        factors = pd.DataFrame(index=price_data.index)
        returns = price_data['close'].pct_change(fill_method=None)

        # Momentum factors (approx. 21 trading days per month)
        for period in [1, 3, 6, 12]:
            factors[f'momentum_{period}m'] = price_data['close'].pct_change(
                periods=period * 21, fill_method=None
            )

        # Short-term reversal
        factors['reversal_1m'] = -price_data['close'].pct_change(periods=21, fill_method=None)

        # Volatility factors
        for period in [1, 3, 6]:
            factors[f'volatility_{period}m'] = returns.rolling(period * 21).std()

        # Liquidity factors
        factors['turnover'] = price_data['volume'] / price_data['shares_outstanding']
        factors['amihud_illiq'] = returns.abs() / (price_data['volume'] * price_data['close'])

        # Technical-indicator factors
        factors['rsi'] = self.calculate_rsi(price_data['close'])
        factors['macd_signal'] = self.calculate_macd_signal(price_data['close'])

        return factors

    def factor_neutralization(self, factors, industry_codes, market_cap):
        """Neutralize factors against industry and size.

        Without neutralization, a "value" score is often just a bet on
        cheap industries (banks, energy) and small caps. Demeaning within
        industries and removing the linear size exposure isolates the
        stock-specific signal.
        """
        neutralized = factors.copy()

        for factor in factors.columns:
            if factors[factor].isna().all():
                continue

            # Industry neutralization: demean within each industry
            for industry in industry_codes.unique():
                mask = industry_codes == industry
                if mask.sum() > 1:
                    industry_mean = factors.loc[mask, factor].mean()
                    neutralized.loc[mask, factor] = (
                        neutralized.loc[mask, factor] - industry_mean
                    )

            # Size neutralization: remove the linear log-market-cap exposure
            log_cap = np.log(market_cap)
            correlation = neutralized[factor].corr(log_cap)
            if not np.isnan(correlation) and log_cap.std() > 0:
                beta = correlation * neutralized[factor].std() / log_cap.std()
                neutralized[factor] = neutralized[factor] - beta * (log_cap - log_cap.mean())

        return neutralized

    def factor_scoring(self, factors):
        """Combine factors into style scores via cross-sectional ranks.

        Percentile ranks are robust to outliers and put every factor on
        the same [0, 1] scale before aggregation.
        """
        scores = pd.DataFrame(index=factors.index)

        # Valuation score (cheaper = higher score, so ranks are inverted).
        # Ratios masked to NaN (non-positive denominators) are skipped per
        # stock: the score is the mean of the ranks that ARE available, so
        # one masked ratio does not void a stock's whole valuation score.
        valuation_factors = ['PE', 'PB', 'PS', 'PCF', 'EV_EBITDA']
        valuation_ranks = [
            1 - factors[factor].rank(pct=True)
            for factor in valuation_factors if factor in factors.columns
        ]
        scores['valuation_score'] = pd.concat(valuation_ranks, axis=1).mean(axis=1)

        # Growth score (faster growth = higher score)
        growth_factors = ['revenue_growth', 'earnings_growth', 'book_value_growth',
                          'roa_change', 'roe_change']
        growth_score = 0
        for factor in growth_factors:
            if factor in factors.columns:
                growth_score += factors[factor].rank(pct=True)
        scores['growth_score'] = growth_score / len(growth_factors)

        # Quality score
        quality_factors = ['roa', 'roe', 'gross_margin', 'operating_margin',
                           'current_ratio', 'asset_turnover']
        quality_score = 0
        for factor in quality_factors:
            if factor in factors.columns:
                quality_score += factors[factor].rank(pct=True)

        # Negatively oriented quality factors (higher leverage = lower score)
        negative_quality_factors = ['debt_to_equity']
        for factor in negative_quality_factors:
            if factor in factors.columns:
                quality_score += 1 - factors[factor].rank(pct=True)

        scores['quality_score'] = quality_score / (
            len(quality_factors) + len(negative_quality_factors)
        )

        # Composite score (weights are a design choice; consider IC-weighting)
        scores['composite_score'] = (
            scores['valuation_score'] * 0.4
            + scores['growth_score'] * 0.3
            + scores['quality_score'] * 0.3
        )

        return scores

    def generate_portfolio_weights(self, scores, max_weight=0.05):
        """Build portfolio weights from composite scores."""
        # Select the top 30% of stocks by composite score
        threshold = scores['composite_score'].quantile(0.7)
        selected = scores[scores['composite_score'] >= threshold]

        # Score-proportional weights, capped per name and renormalized
        raw_weights = selected['composite_score'] / selected['composite_score'].sum()
        weights = np.minimum(raw_weights, max_weight)
        weights = weights / weights.sum()

        return weights

    def calculate_rsi(self, prices, window=14):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd_signal(self, prices):
        """MACD line vs. signal line (1 = bullish crossover state)."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd > signal).astype(int)
```

**Sign caveat.** Valuation ratios are only meaningful with positive denominators — without the NaN masking above, negative-PE loss-makers would rank as the cheapest stocks and top the valuation score.

**The two look-ahead traps in fundamental data.** First, *reporting lag*: Q4 financials dated December 31 are typically not public until February or March — always align statements to their release date. Second, *survivorship bias*: building factors on today's index constituents silently excludes the delisted losers, inflating backtested value and quality premia. Use point-in-time databases where possible.

## 6.2 Event-Driven Quant

Factor models capture slow-moving cross-sectional premia. Event-driven quant instead trades the predictable price dynamics around **discrete corporate events**, where information arrives in a lump and markets digest it imperfectly. Common systematic event strategies include:

- **Post-earnings announcement drift (PEAD)** — stocks with strongly positive (negative) earnings surprises tend to keep drifting up (down) for weeks after the announcement, one of the most persistent anomalies in the literature.
- **Index rebalancing** — additions to a major index attract forced buying from passive funds around the effective date; deletions see the reverse.
- **Merger arbitrage** — after a cash acquisition is announced, the target trades at a discount to the offer price; capturing the spread is a bet on deal completion.
- **Buybacks and insider activity** — announced repurchase programs and clusters of insider buying carry positive drift on average.

The standard research tool for any of these is the **event study**: align all events in event time (day 0 = announcement), compute abnormal returns relative to a benchmark, and average the cumulative abnormal return (CAR) across events.

```python
import numpy as np
import pandas as pd

# --- Event study of post-earnings announcement drift ---
# Synthetic data for illustration: we simulate abnormal returns around
# 200 earnings announcements with a small built-in drift, then recover
# it with the standard CAR methodology.

rng = np.random.default_rng(42)

n_events = 200
window = np.arange(-10, 21)  # event days t-10 .. t+20

# Standardized earnings surprise (SUE) for each event
sue = rng.standard_normal(n_events)

# Daily abnormal returns: noise + drift proportional to the surprise,
# concentrated on the announcement day and decaying afterwards
abnormal = pd.DataFrame(
    rng.normal(0, 0.01, size=(n_events, len(window))),
    columns=window,
)
announcement_jump = 0.02 * sue                      # day-0 reaction
post_drift = 0.0008 * sue                           # daily post-event drift
abnormal[0] = abnormal[0] + announcement_jump
for day in range(1, 21):
    abnormal[day] = abnormal[day] + post_drift

# Sort events into surprise quintiles and compare CARs
quintile = pd.qcut(sue, 5, labels=False)
car = abnormal.cumsum(axis=1)  # cumulative abnormal return per event

car_top = car[quintile == 4].mean()     # most positive surprises
car_bottom = car[quintile == 0].mean()  # most negative surprises

summary = pd.DataFrame({
    'CAR_top_quintile': car_top,
    'CAR_bottom_quintile': car_bottom,
    'long_short_spread': car_top - car_bottom,
})
print(summary.loc[[0, 5, 10, 20]].round(4))
```

A tradable PEAD strategy follows directly from this analysis: at each announcement, rank the surprise (e.g. standardized unexpected earnings, or the day-0 abnormal return itself as a proxy), go long the top quintile and short the bottom quintile, and hold for 20–60 trading days. The critical implementation details are the same as in the event study — **enter only after the announcement is public** (no look-ahead), use abnormal rather than raw returns so market moves don't contaminate the signal, and control for the fact that announcements cluster in earnings season, which concentrates risk in calendar time.
