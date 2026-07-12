# Chapter 11: Performance Evaluation and Attribution

A strategy is only as good as the evidence behind it, and that evidence has to be measured carefully. This chapter builds a systematic evaluation framework: absolute metrics (annualized return, volatility, Sharpe ratio, skewness, kurtosis), drawdown analysis (depth and duration), trade statistics (win rate, profit factor), and benchmark-relative attribution (alpha, beta, tracking error, information ratio). Because full-sample averages can hide regime-dependent behavior, the framework also computes rolling metrics over a moving window, which reveal when a strategy's edge appeared, faded, or reversed.

## 11.1 Strategy Evaluation Framework

The evaluator takes a series of periodic (daily) returns and optionally a benchmark return series. Note that the alpha reported here is the simple annualized mean excess return over the benchmark, not a regression intercept; beta is estimated from the covariance of the aligned return series. The rolling maximum drawdown is computed with a rolling-window helper (`rolling(...).apply(...)`, which loops over windows in Python), an approach that is pandas 2.x-safe and free of chained-assignment pitfalls.

```python
import numpy as np
import pandas as pd


class StrategyEvaluator:
    def __init__(self):
        self.risk_free_rate = 0.03  # annualized risk-free rate

    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """Compute a comprehensive set of performance metrics."""
        metrics = {}

        # Basic return and risk metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = returns.mean() * 252
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = ((metrics['annual_return'] - self.risk_free_rate) /
                                   metrics['annual_volatility'])

        # Higher moments
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max

        metrics['max_drawdown'] = drawdown.min()
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])

        # Drawdown duration: length of each below-water episode
        drawdown_periods = []
        in_drawdown = False
        start_dd = None

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_dd = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if start_dd is not None:
                    drawdown_periods.append(i - start_dd)

        # An episode still open at the end of the sample counts too;
        # otherwise the duration is understated when the deepest
        # drawdown is the current one.
        if in_drawdown and start_dd is not None:
            drawdown_periods.append(len(drawdown) - start_dd)

        metrics['avg_drawdown_duration'] = (np.mean(drawdown_periods)
                                            if drawdown_periods else 0)
        metrics['max_drawdown_duration'] = (max(drawdown_periods)
                                            if drawdown_periods else 0)

        # Trade statistics
        metrics['win_rate'] = (returns > 0).mean()
        metrics['profit_factor'] = (returns[returns > 0].sum() /
                                    abs(returns[returns < 0].sum()))

        # Benchmark-relative metrics
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns

            # Alpha here is the annualized mean excess return
            # (not a regression intercept)
            metrics['alpha'] = excess_returns.mean() * 252
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = (
                metrics['alpha'] / metrics['tracking_error']
                if metrics['tracking_error'] > 0 else 0
            )

            # Beta from the covariance of the aligned return series
            aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
            covariance = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            benchmark_variance = aligned.iloc[:, 1].var()
            metrics['beta'] = (covariance / benchmark_variance
                               if benchmark_variance > 0 else 0)

        return metrics

    def rolling_performance_analysis(self, returns, window=252):
        """Compute performance metrics over a rolling window."""
        rolling_metrics = pd.DataFrame(index=returns.index)

        # Cumulative return over each window
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
        rolling_metrics['rolling_volatility'] = (returns.rolling(window).std() *
                                                 np.sqrt(252))

        # Sharpe from the annualized mean return within each window
        rolling_annual_return = returns.rolling(window).mean() * 252
        rolling_metrics['rolling_sharpe'] = (
            (rolling_annual_return - self.risk_free_rate) /
            rolling_metrics['rolling_volatility']
        )

        # Rolling maximum drawdown, computed within each window
        def window_max_drawdown(window_returns):
            cumulative = np.cumprod(1 + window_returns)
            running_max = np.maximum.accumulate(cumulative)
            return ((cumulative - running_max) / running_max).min()

        rolling_metrics['rolling_max_drawdown'] = returns.rolling(window).apply(
            window_max_drawdown, raw=True
        )

        return rolling_metrics
```
