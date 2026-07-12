# Chapter 1: Trend Following Strategies

Trend following is the oldest and most widely deployed family of systematic strategies: instead of predicting turning points, it identifies an established price trend and rides it until the evidence reverses. This chapter builds the classic toolkit step by step — moving-average crossovers (SMA, EMA, KAMA), channel breakouts (Donchian, Bollinger Bands), and momentum indicators (RSI, MACD) — each with a self-contained Python implementation. Throughout the chapter, `signal`/`position` columns use held-state semantics — the column holds +1 while a long position is on and -1 while a short is on (0 when flat), rather than emitting one-bar impulses — and every backtest lags the signal by one bar to avoid look-ahead bias. All demos run on synthetic price data for illustration only; none of the printed numbers are empirical performance claims.

## 1.1 Moving Average Strategy Family

### 1.1.1 Simple Moving Average (SMA) Strategy

**Strategy rationale**:
The SMA is the most basic trend-following indicator. It smooths price fluctuations by taking the arithmetic mean of the last N periods, revealing the direction of the underlying trend.

**Mathematical model**:
```
SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
```

**Key characteristics**:
- Lag: the SMA reacts slowly to price changes, which makes it good at filtering market noise
- Smoothness: it effectively dampens the impact of short-term price swings
- Trend following: a price crossing of the SMA generates a trend signal

**Implementation details**:
- **Parameter selection**:
  - Short-term SMA: 5-20 days, suited to short-term trading
  - Medium-term SMA: 20-60 days, suited to medium-term trend following
  - Long-term SMA: 100-200 days, suited to long-term investing
- **Signal generation**:
  - Golden cross: the short SMA crosses above the long SMA — buy signal
  - Death cross: the short SMA crosses below the long SMA — sell signal
- **Trend confirmation**: the slope of the SMA gauges trend strength

**Strategy implementation**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class SMAStrategy:
    """Simple moving average crossover strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the SMA strategy.

        Args:
            short_window: lookback of the short moving average
            long_window: lookback of the long moving average
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None

    def calculate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the moving averages."""
        data = data.copy()
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from the SMA crossover."""
        data = self.calculate_sma(data)

        # 1: long, -1: short, 0: flat (while the long SMA is still warming up)
        data['signal'] = 0
        valid = data['SMA_long'].notna()
        data.loc[valid, 'signal'] = np.where(
            data.loc[valid, 'SMA_short'] > data.loc[valid, 'SMA_long'], 1, -1
        )

        # Position changes
        data['positions'] = data['signal'].diff()

        # Mark entry points: any transition into the long (+1) or short (-1)
        # state, including the first entry out of the 0 warm-up state
        prev_signal = data['signal'].shift(1, fill_value=0)
        data['buy_signal'] = (data['signal'] == 1) & (prev_signal != 1)
        data['sell_signal'] = (data['signal'] == -1) & (prev_signal != -1)

        return data

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute strategy returns."""
        data = data.copy()

        # Daily returns
        data['market_returns'] = data['close'].pct_change()

        # Strategy returns: lag the signal by one bar so that a signal
        # observed at the close of day t only earns the return of day t+1
        # (no look-ahead bias)
        data['strategy_returns'] = data['signal'].shift(1) * data['market_returns']

        # Cumulative returns
        data['cumulative_market_returns'] = (1 + data['market_returns']).cumprod()
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()

        return data

    def backtest(self, data: pd.DataFrame) -> Dict:
        """Run the backtest."""
        signals = self.generate_signals(data)
        signals = self.calculate_returns(signals)

        # Performance metrics
        strategy_returns = signals['strategy_returns'].dropna()
        market_returns = signals['market_returns'].dropna()

        metrics = self.calculate_performance_metrics(strategy_returns, market_returns)

        self.signals = signals
        return {
            'signals': signals,
            'metrics': metrics
        }

    def calculate_performance_metrics(self, strategy_returns: pd.Series,
                                    market_returns: pd.Series) -> Dict:
        """Compute performance metrics."""
        # Annualized return
        strategy_annual_return = strategy_returns.mean() * 252
        market_annual_return = market_returns.mean() * 252

        # Annualized volatility
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        market_volatility = market_returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.03  # assume a 3% risk-free rate
        strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_volatility
        market_sharpe = (market_annual_return - risk_free_rate) / market_volatility

        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (strategy_returns > 0).mean()

        return {
            'strategy_annual_return': strategy_annual_return,
            'market_annual_return': market_annual_return,
            'strategy_volatility': strategy_volatility,
            'market_volatility': market_volatility,
            'strategy_sharpe': strategy_sharpe,
            'market_sharpe': market_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'excess_return': strategy_annual_return - market_annual_return
        }

    def plot_strategy(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot the strategy results."""
        if self.signals is None:
            raise ValueError("Run backtest() first")

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Panel 1: price and moving averages
        axes[0].plot(self.signals.index, self.signals['close'], label='Price', linewidth=1)
        axes[0].plot(self.signals.index, self.signals['SMA_short'],
                    label=f'SMA{self.short_window}', alpha=0.8)
        axes[0].plot(self.signals.index, self.signals['SMA_long'],
                    label=f'SMA{self.long_window}', alpha=0.8)

        # Mark entries and exits
        buy_points = self.signals[self.signals['buy_signal']]
        sell_points = self.signals[self.signals['sell_signal']]

        if not buy_points.empty:
            axes[0].scatter(buy_points.index, buy_points['close'],
                          color='green', marker='^', s=100, label='Buy')
        if not sell_points.empty:
            axes[0].scatter(sell_points.index, sell_points['close'],
                          color='red', marker='v', s=100, label='Sell')

        axes[0].set_title('SMA Strategy - Price and Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel 2: cumulative return comparison
        axes[1].plot(self.signals.index, self.signals['cumulative_market_returns'],
                    label='Market return', linewidth=1.5)
        axes[1].plot(self.signals.index, self.signals['cumulative_strategy_returns'],
                    label='Strategy return', linewidth=1.5)
        axes[1].set_title('Cumulative Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: strategy position
        axes[2].plot(self.signals.index, self.signals['signal'],
                    label='Position signal', linewidth=1)
        axes[2].set_title('Strategy Position')
        axes[2].set_ylabel('Position (1: long, -1: short, 0: flat)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Usage example
def demo_sma_strategy():
    """SMA strategy demo (synthetic data for illustration)."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Generate a price series with an upward drift
    returns = np.random.normal(0.0005, 0.02, 500)
    trend = np.linspace(0, 0.3, 500)  # add an upward trend
    prices = 100 * np.exp(np.cumsum(returns + trend/500))

    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)

    # Run the strategy
    strategy = SMAStrategy(short_window=10, long_window=30)
    results = strategy.backtest(data)

    # Print results
    print("SMA strategy backtest results:")
    for key, value in results['metrics'].items():
        print(f"{key}: {value:.4f}")

    # Plot
    strategy.plot_strategy()

    return results

# Run the demo
if __name__ == "__main__":
    demo_sma_strategy()
```

### 1.1.2 Exponential Moving Average (EMA) Strategy

**Strategy rationale**:
The EMA weights recent prices more heavily, making it more responsive to price changes and quicker to catch trend reversals.

**Mathematical model**:
```
EMA(t) = α × P(t) + (1-α) × EMA(t-1)
where α = 2/(n+1) and n is the period length
```

**Key characteristics**:
- Responsiveness: reacts faster to price changes, reducing lag
- Decaying weights: recent prices carry more weight, older prices progressively less
- Trend sensitivity: identifies trend turning points earlier

**Implementation details**:
```python
import pandas as pd
import numpy as np

class EMAStrategy:
    """Exponential moving average dual-crossover strategy."""

    def __init__(self, fast_period=12, slow_period=26):
        """
        Initialize the EMA strategy.

        Args:
            fast_period: fast EMA lookback
            slow_period: slow EMA lookback
        """
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Compute an exponential moving average."""
        return data.ewm(span=period, adjust=False).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from the EMA crossover."""
        data = data.copy()

        # Fast and slow EMAs
        data['EMA_fast'] = self.calculate_ema(data['close'], self.fast_period)
        data['EMA_slow'] = self.calculate_ema(data['close'], self.slow_period)

        # Held-state signal, same semantics as the SMA strategy:
        # +1 while the fast EMA is above the slow EMA, -1 while below
        data['signal'] = np.where(data['EMA_fast'] > data['EMA_slow'], 1, -1)

        # Stay flat until the slow EMA has seen a full lookback of data
        # (with adjust=False the EMA is defined from bar 0, but its early
        # values are dominated by the seed price)
        data.loc[data.index[:self.slow_period], 'signal'] = 0

        # Crossover markers (any transition into the long or short state)
        prev_signal = data['signal'].shift(1, fill_value=0)
        data['golden_cross'] = (data['signal'] == 1) & (prev_signal != 1)
        data['death_cross'] = (data['signal'] == -1) & (prev_signal != -1)

        return data

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the backtest."""
        signals = self.generate_signals(data)

        # Returns: lag the signal one bar to avoid look-ahead bias
        signals['returns'] = signals['close'].pct_change()
        signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
        signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()

        return signals

# Usage example
def demo_ema_strategy():
    """EMA strategy demo (synthetic data for illustration)."""
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)

    # Generate a price series
    returns = np.random.normal(0.001, 0.02, 500)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)

    # Run the strategy
    strategy = EMAStrategy(fast_period=12, slow_period=26)
    results = strategy.backtest(data)

    final_return = results['cumulative_returns'].iloc[-1] - 1
    print(f"EMA strategy backtest complete, cumulative return: {final_return:.2%}")
    return results

if __name__ == "__main__":
    demo_ema_strategy()
```

**EMA vs. SMA comparison**:

| Feature | EMA | SMA |
|------|-----|-----|
| Response speed | Fast | Slow |
| Lag | Smaller | Larger |
| Noise filtering | Weaker | Stronger |
| Trend tracking | Sensitive | Stable |
| False signals | More | Fewer |

### 1.1.3 Adaptive Moving Average (AMA / KAMA) Strategy

**Strategy rationale**:
The AMA adjusts its smoothing constant automatically based on market volatility: it responds quickly in trending markets and filters noise in choppy markets.

**KAMA implementation**:
```python
import pandas as pd
import numpy as np

class KAMAStrategy:
    """Kaufman Adaptive Moving Average strategy."""

    def __init__(self, period=20, fast_sc=2, slow_sc=30):
        """
        Initialize the KAMA strategy.

        Args:
            period: efficiency ratio lookback
            fast_sc: fast smoothing constant period
            slow_sc: slow smoothing constant period
        """
        self.period = period
        self.fast_sc = fast_sc
        self.slow_sc = slow_sc

    def calculate_kama(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the KAMA indicator."""
        data = data.copy()

        # Direction: net price move over the lookback
        data['Direction'] = data['close'].diff(self.period).abs()

        # Volatility: sum of absolute one-bar moves over the lookback
        data['Volatility'] = data['close'].diff().abs().rolling(self.period).sum()

        # Efficiency Ratio
        data['ER'] = data['Direction'] / data['Volatility']
        data['ER'] = data['ER'].fillna(0)

        # Smoothing Constant
        fast_alpha = 2 / (self.fast_sc + 1)
        slow_alpha = 2 / (self.slow_sc + 1)
        data['SC'] = (data['ER'] * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        # KAMA is recursive, so we compute it with an explicit loop
        # (SC is never NaN here because ER is fillna(0), so slow_alpha**2
        # is used during the warm-up bars)
        kama_values = [data['close'].iloc[0]]  # seed value

        for i in range(1, len(data)):
            kama_new = kama_values[-1] + data['SC'].iloc[i] * (data['close'].iloc[i] - kama_values[-1])
            kama_values.append(kama_new)

        data['KAMA'] = kama_values

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        data = self.calculate_kama(data)

        # Price relative to KAMA
        data['price_above_kama'] = data['close'] > data['KAMA']
        data['price_below_kama'] = data['close'] < data['KAMA']

        # Crossover events (fill_value keeps the shifted column boolean)
        data['signal'] = 0
        data.loc[data['price_above_kama'] & ~data['price_above_kama'].shift(1, fill_value=False), 'signal'] = 1
        data.loc[data['price_below_kama'] & ~data['price_below_kama'].shift(1, fill_value=False), 'signal'] = -1

        # Hold each state until the opposite crossover (held-state semantics,
        # consistent with the rest of the chapter)
        data['signal'] = data['signal'].where(data['signal'] != 0).ffill().fillna(0).astype(int)

        # KAMA slope signals
        data['kama_slope'] = data['KAMA'].diff()
        data['kama_rising'] = data['kama_slope'] > 0
        data['kama_falling'] = data['kama_slope'] < 0

        return data
```

**KAMA advantages**:
- Adaptivity: adjusts automatically to market efficiency
- Fewer false signals: more stable in choppy markets
- Trend sensitivity: responds quickly in trending markets

## 1.2 Breakout Strategy Family

### 1.2.1 Donchian Channel Breakout Strategy

**Strategy rationale**:
Detects trend breakouts when price exceeds the highest high (upper band) or lowest low (lower band) of the past N days — a classic trend-following approach.

**Core concepts**:
- Upper band: the highest high of the past N days
- Lower band: the lowest low of the past N days
- Breakout: price moving beyond a channel boundary

**Implementation details**:
```python
import pandas as pd
import numpy as np

class DonchianChannelStrategy:
    """Donchian channel breakout strategy."""

    def __init__(self, entry_period=20, exit_period=10):
        """
        Initialize strategy parameters.

        Args:
            entry_period: entry channel lookback
            exit_period: exit channel lookback
        """
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.position = 0  # current position: 1 long, -1 short, 0 flat

    def calculate_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the Donchian channels."""
        data = data.copy()

        # Entry channel
        data['upper_channel'] = data['high'].rolling(window=self.entry_period).max()
        data['lower_channel'] = data['low'].rolling(window=self.entry_period).min()
        data['middle_channel'] = (data['upper_channel'] + data['lower_channel']) / 2

        # Exit channel
        data['exit_upper'] = data['high'].rolling(window=self.exit_period).max()
        data['exit_lower'] = data['low'].rolling(window=self.exit_period).min()

        # Channel width
        data['channel_width'] = data['upper_channel'] - data['lower_channel']
        data['channel_width_pct'] = data['channel_width'] / data['middle_channel']

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        data = self.calculate_channels(data)

        # Entry signals: compare today's close with yesterday's channel so the
        # breakout level is fully known before the bar being tested
        data['long_entry'] = data['close'] > data['upper_channel'].shift(1)
        data['short_entry'] = data['close'] < data['lower_channel'].shift(1)

        # Exit signals
        data['long_exit'] = data['close'] < data['exit_lower'].shift(1)
        data['short_exit'] = data['close'] > data['exit_upper'].shift(1)

        # Build the position path (stateful, so we use a loop over a
        # NumPy array and assign the column once at the end)
        positions = np.zeros(len(data), dtype=int)
        current_position = 0

        for i in range(len(data)):
            if data['long_entry'].iloc[i] and current_position <= 0:
                current_position = 1
            elif data['short_entry'].iloc[i] and current_position >= 0:
                current_position = -1
            elif data['long_exit'].iloc[i] and current_position > 0:
                current_position = 0
            elif data['short_exit'].iloc[i] and current_position < 0:
                current_position = 0

            positions[i] = current_position

        data['position'] = positions

        # Position changes
        data['signal'] = data['position'].diff()

        return data

    def add_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add entry filters."""
        data = data.copy()

        # Volume filter
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_filter'] = data['volume'] > data['volume_ma'] * 1.5
        else:
            data['volume_filter'] = True

        # ATR filter (avoid trading in low-volatility regimes)
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(14).mean()
        data['atr_filter'] = data['atr'] > data['atr'].rolling(50).quantile(0.3)

        # Apply the filters (provided for extension; the basic backtest
        # below uses the unfiltered entries)
        data['filtered_long_entry'] = data['long_entry'] & data['volume_filter'] & data['atr_filter']
        data['filtered_short_entry'] = data['short_entry'] & data['volume_filter'] & data['atr_filter']

        return data

    def backtest_with_stops(self, data: pd.DataFrame, stop_loss_pct=0.02) -> pd.DataFrame:
        """Backtest with a stop-loss overlay."""
        data = self.generate_signals(data)
        data = self.add_filters(data)

        # Returns: lag the position one bar to avoid look-ahead bias
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']

        # Stop-loss logic (stateful loop; assign the column once at the end)
        stop_flags = np.zeros(len(data), dtype=bool)
        current_position = 0
        entry_price = np.nan

        for i in range(1, len(data)):
            if data['position'].iloc[i] != current_position:
                current_position = data['position'].iloc[i]
                entry_price = data['close'].iloc[i]

            # Check the stop
            if current_position > 0:  # long position
                if data['close'].iloc[i] < entry_price * (1 - stop_loss_pct):
                    stop_flags[i] = True
                    current_position = 0
            elif current_position < 0:  # short position
                if data['close'].iloc[i] > entry_price * (1 + stop_loss_pct):
                    stop_flags[i] = True
                    current_position = 0

        # For pedagogical simplicity the stop flags are diagnostic only;
        # a production backtest would flatten the position (and its returns)
        # on the bar after each stop fires.
        data['stop_loss'] = stop_flags

        return data

# Usage example
def demo_donchian_strategy():
    """Donchian channel strategy demo (synthetic data for illustration)."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Generate a trending price series
    returns = np.random.normal(0.0008, 0.025, 500)
    trend = np.sin(np.linspace(0, 4*np.pi, 500)) * 0.001  # add a cyclical trend
    prices = 100 * np.exp(np.cumsum(returns + trend))

    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)

    # Run the strategy
    strategy = DonchianChannelStrategy(entry_period=20, exit_period=10)
    results = strategy.backtest_with_stops(data, stop_loss_pct=0.03)

    # Performance metrics
    total_return = (1 + results['strategy_returns']).prod() - 1
    annual_return = results['strategy_returns'].mean() * 252
    volatility = results['strategy_returns'].std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    print("Donchian channel strategy backtest results:")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized return: {annual_return:.2%}")
    print(f"Annualized volatility: {volatility:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")

    return results
```

**Parameter tuning guidelines**:
- Entry period: 10-40 days, commonly 20
- Exit period: half the entry period, commonly 10
- Filters: volume expansion, ATR above a historical quantile

**Suitable markets**:
- Markets with strong trends
- Instruments with moderate volatility
- Liquid instruments

### 1.2.2 Bollinger Bands Breakout Strategy

**Strategy rationale**:
Identifies trend breakouts when price crosses the Bollinger Bands, while also exploiting mean-reversion behavior. The bands consist of a middle band (moving average), an upper band (middle + N standard deviations), and a lower band (middle - N standard deviations).

**Core concepts**:
- Middle band: typically a 20-day simple moving average
- Upper/lower bands: middle band ± 2 standard deviations
- Bandwidth: a gauge of market volatility
- %B: where price sits within the bands

```python
import pandas as pd
import numpy as np

class BollingerBandsStrategy:
    """Bollinger Bands breakout strategy."""

    def __init__(self, period=20, std_dev=2, volume_factor=1.5):
        """
        Initialize the Bollinger Bands strategy.

        Args:
            period: moving average lookback
            std_dev: standard deviation multiplier
            volume_factor: volume surge multiplier
        """
        self.period = period
        self.std_dev = std_dev
        self.volume_factor = volume_factor

    def calculate_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the Bollinger Bands."""
        data = data.copy()

        # Middle band (moving average)
        data['BB_middle'] = data['close'].rolling(window=self.period).mean()

        # Rolling standard deviation
        rolling_std = data['close'].rolling(window=self.period).std()

        # Upper and lower bands
        data['BB_upper'] = data['BB_middle'] + (rolling_std * self.std_dev)
        data['BB_lower'] = data['BB_middle'] - (rolling_std * self.std_dev)

        # Bandwidth
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']

        # Position of price within the bands (%B)
        data['BB_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])

        # Trailing percentile rank of the bandwidth
        data['BB_width_percentile'] = data['BB_width'].rolling(252).rank(pct=True)

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        data = self.calculate_bands(data)

        # Volume filter
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_surge'] = data['volume'] > data['volume_ma'] * self.volume_factor
        else:
            data['volume_surge'] = True

        # Breakout signals (high-volatility regime)
        high_volatility = data['BB_width_percentile'] > 0.7
        data['breakout_long'] = (
            (data['close'] > data['BB_upper']) &
            high_volatility &
            data['volume_surge']
        )
        data['breakout_short'] = (
            (data['close'] < data['BB_lower']) &
            high_volatility &
            data['volume_surge']
        )

        # Mean-reversion signals (low-volatility regime)
        low_volatility = data['BB_width_percentile'] < 0.3
        data['mean_revert_long'] = (
            (data['close'] < data['BB_lower']) &
            low_volatility
        )
        data['mean_revert_short'] = (
            (data['close'] > data['BB_upper']) &
            low_volatility
        )

        # Squeeze signal (expansion after a band contraction)
        data['squeeze'] = data['BB_width'] < data['BB_width'].rolling(50).quantile(0.2)
        data['squeeze_breakout'] = (
            data['squeeze'].shift(1, fill_value=False) &
            ~data['squeeze'] &
            data['volume_surge']
        )

        return data

    def backtest_dual_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Backtest the dual strategy (breakout + mean reversion)."""
        data = self.generate_signals(data)

        # Regime-based weights: breakout gets more weight when volatility is high
        data['trend_weight'] = data['BB_width_percentile']
        data['mean_revert_weight'] = 1 - data['trend_weight']

        # Breakout signal
        data['breakout_signal'] = 0
        data.loc[data['breakout_long'], 'breakout_signal'] = 1
        data.loc[data['breakout_short'], 'breakout_signal'] = -1

        # Mean-reversion signal
        data['mean_revert_signal'] = 0
        data.loc[data['mean_revert_long'], 'mean_revert_signal'] = 1
        data.loc[data['mean_revert_short'], 'mean_revert_signal'] = -1

        # Combined signal
        data['combined_signal'] = (
            data['breakout_signal'] * data['trend_weight'] +
            data['mean_revert_signal'] * data['mean_revert_weight']
        )

        # Returns: lag the signal one bar to avoid look-ahead bias
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['combined_signal'].shift(1) * data['returns']
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

        return data

# Usage example
def demo_bollinger_strategy():
    """Bollinger Bands strategy demo (synthetic data for illustration)."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Generate a price series with cyclical volatility
    # (amplitude kept below the base level so the std dev stays positive)
    base_trend = np.linspace(0, 0.2, 500)
    volatility_cycle = 0.015 + 0.01 * np.sin(np.linspace(0, 8*np.pi, 500))
    returns = np.random.normal(base_trend/500, volatility_cycle)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 500))),
        'volume': np.random.randint(500000, 5000000, 500)
    }, index=dates)

    # Run the strategy
    strategy = BollingerBandsStrategy(period=20, std_dev=2)
    results = strategy.backtest_dual_strategy(data)

    print("Bollinger Bands strategy backtest complete")
    return results
```

## 1.3 Momentum Strategy Family

### 1.3.1 RSI Momentum Strategy

**Strategy rationale**:
The RSI (Relative Strength Index) measures the speed and magnitude of price moves, identifying overbought/oversold conditions and momentum turning points. RSI oscillates between 0 and 100; readings above 70 are conventionally overbought and below 30 oversold.

**Key characteristics**:
- Oscillator: bounded between 0 and 100
- Overbought/oversold: flags price extremes
- Divergence: price and RSI moving in opposite directions

```python
import pandas as pd
import numpy as np

class RSIStrategy:
    """RSI momentum strategy."""

    def __init__(self, period=14, overbought=70, oversold=30, rsi_ma_period=5):
        """
        Initialize the RSI strategy.

        Args:
            period: RSI lookback
            overbought: overbought threshold
            oversold: oversold threshold
            rsi_ma_period: RSI moving average lookback
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.rsi_ma_period = rsi_ma_period

    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the RSI (Cutler's variant, using simple moving averages)."""
        data = data.copy()

        # Price changes
        delta = data['close'].diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        # Relative strength and RSI
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # RSI moving average
        data['RSI_MA'] = data['RSI'].rolling(window=self.rsi_ma_period).mean()

        # RSI momentum
        data['RSI_momentum'] = data['RSI'].diff(3)

        return data

    def detect_divergence(self, data: pd.DataFrame, lookback: int = 10,
                          max_pivot_gap: int = 50) -> pd.DataFrame:
        """Detect price/RSI divergences from confirmed swing points.

        A bar j is a swing high (low) when its close is the extreme of the
        window [j - lookback, j + lookback]. The pivot is only *confirmed*
        at bar i = j + lookback, once the right side of the window has been
        observed, and divergence signals are stamped on the confirmation
        bar — so no future data is used at signal time.
        """
        data = data.copy()
        n = len(data)
        close = data['close'].to_numpy()
        rsi = data['RSI'].to_numpy()

        bullish = np.zeros(n, dtype=bool)
        bearish = np.zeros(n, dtype=bool)

        swing_highs = []  # bar positions of confirmed swing highs
        swing_lows = []   # bar positions of confirmed swing lows

        for i in range(2 * lookback, n):
            j = i - lookback                       # candidate pivot bar
            if np.isnan(rsi[j]):
                continue
            window = close[j - lookback : i + 1]   # ends at the current bar i

            # Confirmed swing high at bar j
            if close[j] == window.max():
                swing_highs.append(j)
                if len(swing_highs) >= 2:
                    prev, curr = swing_highs[-2], swing_highs[-1]
                    # Higher high in price, lower high in RSI -> bearish divergence
                    if (curr - prev <= max_pivot_gap and
                            close[curr] > close[prev] and rsi[curr] < rsi[prev]):
                        bearish[i] = True

            # Confirmed swing low at bar j
            if close[j] == window.min():
                swing_lows.append(j)
                if len(swing_lows) >= 2:
                    prev, curr = swing_lows[-2], swing_lows[-1]
                    # Lower low in price, higher low in RSI -> bullish divergence
                    if (curr - prev <= max_pivot_gap and
                            close[curr] < close[prev] and rsi[curr] > rsi[prev]):
                        bullish[i] = True

        data['bullish_divergence'] = bullish
        data['bearish_divergence'] = bearish

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        data = self.calculate_rsi(data)
        data = self.detect_divergence(data)

        # Classic overbought/oversold signals (fire on entering the zone)
        data['oversold_signal'] = (data['RSI'] < self.oversold) & (data['RSI'].shift(1) >= self.oversold)
        data['overbought_signal'] = (data['RSI'] > self.overbought) & (data['RSI'].shift(1) <= self.overbought)

        # RSI midline crossings
        data['rsi_bull_cross'] = (data['RSI'] > 50) & (data['RSI'].shift(1) <= 50)
        data['rsi_bear_cross'] = (data['RSI'] < 50) & (data['RSI'].shift(1) >= 50)

        # RSI relative to its moving average
        data['rsi_above_ma'] = data['RSI'] > data['RSI_MA']
        data['rsi_below_ma'] = data['RSI'] < data['RSI_MA']

        # Combined signals
        data['signal'] = 0

        # Buy signals
        buy_conditions = (
            data['oversold_signal'] |
            data['bullish_divergence'] |
            (data['rsi_bull_cross'] & data['rsi_above_ma'])
        )
        data.loc[buy_conditions, 'signal'] = 1

        # Sell signals
        sell_conditions = (
            data['overbought_signal'] |
            data['bearish_divergence'] |
            (data['rsi_bear_cross'] & data['rsi_below_ma'])
        )
        data.loc[sell_conditions, 'signal'] = -1

        # Hold each state until the opposite trigger fires (held-state
        # semantics, consistent with the rest of the chapter)
        data['signal'] = data['signal'].where(data['signal'] != 0).ffill().fillna(0).astype(int)

        return data

# RSI strategy usage example
def demo_rsi_strategy():
    """RSI strategy demo (synthetic data for illustration)."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Generate a trending price series
    trend = np.cumsum(np.random.normal(0.001, 0.02, 500))
    noise = np.random.normal(0, 0.015, 500)
    prices = 100 * np.exp(trend + noise)

    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)

    strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    results = strategy.generate_signals(data)

    print("RSI strategy signal generation complete")
    return results
```

### 1.3.2 MACD Momentum Strategy

**Strategy rationale**:
MACD detects momentum shifts through the spread between fast and slow EMAs; the MACD histogram shows how momentum strength is evolving.

```python
import pandas as pd
import numpy as np

class MACDStrategy:
    """MACD momentum strategy."""

    def __init__(self, fast=12, slow=26, signal=9):
        """
        Initialize the MACD strategy.

        Args:
            fast: fast EMA lookback
            slow: slow EMA lookback
            signal: signal line EMA lookback
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the MACD indicator."""
        data = data.copy()

        # Fast and slow EMAs
        data['EMA_fast'] = data['close'].ewm(span=self.fast, adjust=False).mean()
        data['EMA_slow'] = data['close'].ewm(span=self.slow, adjust=False).mean()

        # MACD line
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']

        # Signal line
        data['MACD_signal'] = data['MACD'].ewm(span=self.signal, adjust=False).mean()

        # MACD histogram
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        data = self.calculate_macd(data)

        # MACD golden/death crosses
        data['golden_cross'] = (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))
        data['death_cross'] = (data['MACD'] < data['MACD_signal']) & (data['MACD'].shift(1) >= data['MACD_signal'].shift(1))

        # MACD zero-line crossings
        data['zero_cross_up'] = (data['MACD'] > 0) & (data['MACD'].shift(1) <= 0)
        data['zero_cross_down'] = (data['MACD'] < 0) & (data['MACD'].shift(1) >= 0)

        # Histogram slope signals. Note: the histogram crossing zero is the
        # same event as a golden/death cross, so the "turning" signals test
        # a slope reversal instead — momentum decelerating and then
        # re-accelerating — which is a genuinely distinct, earlier trigger.
        hist_slope = data['MACD_histogram'].diff()
        data['histogram_increasing'] = hist_slope > 0
        data['histogram_decreasing'] = hist_slope < 0
        data['histogram_turning_up'] = (hist_slope > 0) & (hist_slope.shift(1) <= 0)
        data['histogram_turning_down'] = (hist_slope < 0) & (hist_slope.shift(1) >= 0)

        # Combined signals
        data['signal'] = 0

        # Buy signals
        buy_conditions = (
            data['golden_cross'] |
            data['zero_cross_up'] |
            data['histogram_turning_up']
        )
        data.loc[buy_conditions, 'signal'] = 1

        # Sell signals
        sell_conditions = (
            data['death_cross'] |
            data['zero_cross_down'] |
            data['histogram_turning_down']
        )
        data.loc[sell_conditions, 'signal'] = -1

        # Hold each state until the opposite trigger fires (held-state
        # semantics, consistent with the rest of the chapter)
        data['signal'] = data['signal'].where(data['signal'] != 0).ffill().fillna(0).astype(int)

        return data
```
