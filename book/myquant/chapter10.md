# Chapter 10: Portfolio Construction and Risk Management

Individual strategies rarely survive on their own; the durable edge comes from combining weakly correlated return streams and controlling risk dynamically. This chapter covers both halves of that problem. First, a multi-strategy allocation framework that supports equal weighting, mean-variance optimization, and risk parity (equal risk contribution), with per-strategy allocation caps. Second, a dynamic risk manager that scales exposure through volatility targeting, market-regime filters, and drawdown control. In every adjustment rule, the scaling factor is shifted by one period before being applied to returns, so position sizes only ever depend on information available at the time — no look-ahead bias.

## 10.1 Multi-Strategy Portfolio Framework

The allocator treats each strategy's return stream as an asset. Equal weighting is the robust baseline; mean-variance maximizes the in-sample Sharpe ratio but is sensitive to estimation error; risk parity sizes positions so each strategy contributes equally to portfolio volatility, which avoids concentration in whichever strategy happened to have the best recent returns.

```python
import numpy as np
import pandas as pd
import scipy.optimize as optimize


class MultiStrategyPortfolio:
    def __init__(self, max_allocation_per_strategy=0.3):
        self.strategies = {}
        self.allocations = {}
        self.max_allocation = max_allocation_per_strategy

    def add_strategy(self, name, strategy_returns, sharpe_ratio, max_drawdown):
        """Register a strategy return stream with its summary statistics."""
        self.strategies[name] = {
            'returns': strategy_returns,
            'sharpe': sharpe_ratio,
            'max_dd': max_drawdown,
            'volatility': strategy_returns.std() * np.sqrt(252)
        }

    def optimize_allocations(self, method='equal_risk_contribution'):
        """Compute strategy weights using the chosen allocation method."""
        strategy_returns = pd.DataFrame({name: data['returns']
                                         for name, data in self.strategies.items()})

        if method == 'equal_weight':
            n_strategies = len(self.strategies)
            allocations = {name: 1 / n_strategies for name in self.strategies.keys()}

        elif method == 'risk_parity':
            allocations = self.risk_parity_optimization(strategy_returns)

        elif method == 'mean_variance':
            allocations = self.mean_variance_optimization(strategy_returns)

        elif method == 'equal_risk_contribution':
            allocations = self.equal_risk_contribution(strategy_returns)

        else:
            raise ValueError(f"unknown method: {method}")

        # Apply the per-strategy cap, then renormalize to sum to 1.
        # Note: this single pass is an approximation — renormalization can
        # push a weight slightly back above the cap. For exact caps, embed
        # the bound in the optimizer (as risk_parity_optimization does).
        for name in allocations:
            allocations[name] = min(allocations[name], self.max_allocation)

        total_allocation = sum(allocations.values())
        allocations = {name: weight / total_allocation
                       for name, weight in allocations.items()}

        self.allocations = allocations
        return allocations

    def risk_parity_optimization(self, returns):
        """Risk parity: equalize each strategy's contribution to portfolio risk."""
        cov_matrix = returns.cov() * 252  # annualized covariance matrix

        def risk_budget_objective(weights, cov_matrix):
            """Penalize deviations from equal risk contributions."""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib

            # Target: every strategy contributes the same share of risk
            target_risk = portfolio_vol / len(weights)
            return np.sum((contrib - target_risk)**2)

        n_assets = len(returns.columns)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, self.max_allocation) for _ in range(n_assets))

        result = optimize.minimize(
            risk_budget_objective,
            np.ones(n_assets) / n_assets,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return dict(zip(returns.columns, result.x))

    def mean_variance_optimization(self, returns):
        """Mean-variance: maximize the in-sample Sharpe ratio."""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)

        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -port_return / port_vol

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, self.max_allocation) for _ in range(n_assets))

        result = optimize.minimize(
            neg_sharpe,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return dict(zip(returns.columns, result.x))

    def equal_risk_contribution(self, returns):
        """Equal risk contribution (equivalent to risk parity here)."""
        return self.risk_parity_optimization(returns)

    def calculate_portfolio_metrics(self, risk_free_rate=0.03):
        """Compute performance metrics for the allocated portfolio.

        The Sharpe ratio subtracts `risk_free_rate` (annualized, default 3%),
        matching the convention used by Chapter 11's StrategyEvaluator.
        """
        if not self.allocations:
            return None

        # Weighted portfolio returns
        portfolio_returns = sum(self.allocations[name] * self.strategies[name]['returns']
                                for name in self.allocations)

        # Annualized return, volatility, and Sharpe ratio
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

        # Maximum drawdown from the cumulative equity curve
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annual_return / abs(max_drawdown)
        }
```

## 10.2 Dynamic Risk Management

Static weights leave the portfolio exposed to volatility clustering: the same nominal position is far riskier in a crisis than in a calm market. Three complementary overlays address this. Volatility targeting scales leverage inversely to realized volatility so the portfolio runs at a roughly constant risk level. Regime-based adjustment cuts exposure when stress indicators (VIX spikes, downtrends, liquidity stress) fire. Drawdown control halves the position whenever the equity curve falls beyond a threshold, limiting the depth of losing streaks. Each multiplier is applied with a one-period lag (`shift(1)`) to keep the rules implementable in real time.

```python
import numpy as np
import pandas as pd


class DynamicRiskManager:
    def __init__(self, target_volatility=0.15, lookback_window=60):
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window

    def calculate_portfolio_volatility(self, returns):
        """Annualized rolling realized volatility."""
        return returns.rolling(self.lookback_window).std() * np.sqrt(252)

    def volatility_targeting(self, strategy_returns):
        """Scale exposure so realized volatility tracks the target."""
        realized_vol = self.calculate_portfolio_volatility(strategy_returns)

        # Leverage factor, capped to a sensible range
        leverage_factor = self.target_volatility / realized_vol
        leverage_factor = leverage_factor.fillna(1).clip(0.5, 2.0)

        # Apply yesterday's leverage to today's return (no look-ahead)
        adjusted_returns = strategy_returns * leverage_factor.shift(1)

        return adjusted_returns, leverage_factor

    def regime_based_risk_adjustment(self, returns, market_indicators):
        """Reduce exposure when market stress indicators fire."""
        risk_multipliers = pd.Series(1.0, index=returns.index)

        # VIX regime: cut risk when VIX is in its top quintile
        if 'vix' in market_indicators.columns:
            vix_high = (market_indicators['vix'] >
                        market_indicators['vix'].rolling(252).quantile(0.8))
            risk_multipliers.loc[vix_high] *= 0.5

        # Trend regime: reduce risk in downtrends
        if 'market_trend' in market_indicators.columns:
            downtrend = market_indicators['market_trend'] < 0
            risk_multipliers.loc[downtrend] *= 0.7

        # Liquidity regime: cut risk sharply under liquidity stress
        if 'liquidity_stress' in market_indicators.columns:
            liquidity_stress = (
                market_indicators['liquidity_stress'] >
                market_indicators['liquidity_stress'].rolling(252).quantile(0.9)
            )
            risk_multipliers.loc[liquidity_stress] *= 0.3

        # Apply with a one-period lag (no look-ahead)
        adjusted_returns = returns * risk_multipliers.shift(1)
        return adjusted_returns, risk_multipliers

    def drawdown_control(self, returns, max_drawdown_threshold=0.1):
        """Halve the position while drawdown exceeds the threshold."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max

        # Risk-off whenever the drawdown breaches the threshold
        risk_off = drawdown < -max_drawdown_threshold

        position_multiplier = pd.Series(1.0, index=returns.index)
        position_multiplier.loc[risk_off] = 0.5

        # Apply with a one-period lag (no look-ahead)
        controlled_returns = returns * position_multiplier.shift(1)

        return controlled_returns, position_multiplier
```
