
## 一、趋势跟踪策略体系

### 1.1 移动平均线策略族

#### 1.1.1 简单移动平均线(SMA)策略

**策略原理**：
SMA是最基础的趋势跟踪指标，通过计算过去N个周期的算术平均值来平滑价格波动，识别趋势方向。

**数学模型**：
```
SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
```

**核心特点**：
- 延迟性：SMA对价格变化反应较慢，适合过滤市场噪音
- 平滑性：能有效减少价格波动的干扰
- 趋势跟踪：当价格突破SMA时产生趋势信号

**实现细节**：
- **参数选择**：
  - 短期SMA: 5-20日，适用于短线交易
  - 中期SMA: 20-60日，适用于中期趋势跟踪
  - 长期SMA: 100-200日，适用于长期投资
- **信号生成**：
  - 金叉：短期SMA向上突破长期SMA，产生买入信号
  - 死叉：短期SMA向下跌破长期SMA，产生卖出信号
- **趋势确认**：SMA斜率判断趋势强度

**策略代码实现**：
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class SMAStrategy:
    """简单移动平均线策略"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        初始化SMA策略
        
        Args:
            short_window: 短期移动平均线周期
            long_window: 长期移动平均线周期
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None
    
    def calculate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        data = data.copy()
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_sma(data)
        
        # 初始化信号列
        data['signal'] = 0
        data['position'] = 0
        
        # 生成交易信号
        # 1: 买入/做多，-1: 卖出/做空，0: 持有
        data['signal'][self.short_window:] = np.where(
            data['SMA_short'][self.short_window:] > data['SMA_long'][self.short_window:], 1, -1
        )
        
        # 计算仓位变化
        data['positions'] = data['signal'].diff()
        
        # 识别买入和卖出点
        data['buy_signal'] = (data['positions'] == 2)  # 从-1变为1
        data['sell_signal'] = (data['positions'] == -2)  # 从1变为-1
        
        return data
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略收益"""
        data = data.copy()
        
        # 计算日收益率
        data['market_returns'] = data['close'].pct_change()
        
        # 计算策略收益（信号滞后一期）
        data['strategy_returns'] = data['signal'].shift(1) * data['market_returns']
        
        # 计算累积收益
        data['cumulative_market_returns'] = (1 + data['market_returns']).cumprod()
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
        
        return data
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """策略回测"""
        signals = self.generate_signals(data)
        signals = self.calculate_returns(signals)
        
        # 计算策略指标
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
        """计算业绩指标"""
        # 年化收益率
        strategy_annual_return = strategy_returns.mean() * 252
        market_annual_return = market_returns.mean() * 252
        
        # 年化波动率
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        market_volatility = market_returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率3%
        strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_volatility
        market_sharpe = (market_annual_return - risk_free_rate) / market_volatility
        
        # 最大回撤
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 胜率
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
        """绘制策略图表"""
        if self.signals is None:
            raise ValueError("请先运行backtest方法")
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 第一个子图：价格和移动平均线
        axes[0].plot(self.signals.index, self.signals['close'], label='价格', linewidth=1)
        axes[0].plot(self.signals.index, self.signals['SMA_short'], 
                    label=f'SMA{self.short_window}', alpha=0.8)
        axes[0].plot(self.signals.index, self.signals['SMA_long'], 
                    label=f'SMA{self.long_window}', alpha=0.8)
        
        # 标记买卖点
        buy_points = self.signals[self.signals['buy_signal']]
        sell_points = self.signals[self.signals['sell_signal']]
        
        if not buy_points.empty:
            axes[0].scatter(buy_points.index, buy_points['close'], 
                          color='green', marker='^', s=100, label='买入')
        if not sell_points.empty:
            axes[0].scatter(sell_points.index, sell_points['close'], 
                          color='red', marker='v', s=100, label='卖出')
        
        axes[0].set_title('SMA策略 - 价格与信号')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 第二个子图：累积收益对比
        axes[1].plot(self.signals.index, self.signals['cumulative_market_returns'], 
                    label='市场收益', linewidth=1.5)
        axes[1].plot(self.signals.index, self.signals['cumulative_strategy_returns'], 
                    label='策略收益', linewidth=1.5)
        axes[1].set_title('累积收益对比')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 第三个子图：策略仓位
        axes[2].plot(self.signals.index, self.signals['signal'], 
                    label='仓位信号', linewidth=1)
        axes[2].set_title('策略仓位')
        axes[2].set_ylabel('仓位 (1:多头, -1:空头, 0:空仓)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用示例
def demo_sma_strategy():
    """SMA策略演示"""
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 生成带趋势的价格数据
    returns = np.random.normal(0.0005, 0.02, 500)
    trend = np.linspace(0, 0.3, 500)  # 添加上升趋势
    prices = 100 * np.exp(np.cumsum(returns + trend/500))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    # 运行策略
    strategy = SMAStrategy(short_window=10, long_window=30)
    results = strategy.backtest(data)
    
    # 打印结果
    print("SMA策略回测结果:")
    for key, value in results['metrics'].items():
        print(f"{key}: {value:.4f}")
    
    # 绘制图表
    strategy.plot_strategy()
    
    return results

# 运行演示
if __name__ == "__main__":
    demo_sma_strategy()
```

### 1.1.2 指数移动平均线(EMA)策略

**策略原理**：
EMA给予近期价格更高权重，对价格变化更敏感，能更快捕捉趋势转变。

**数学模型**：
```
EMA(t) = α × P(t) + (1-α) × EMA(t-1)
其中：α = 2/(n+1)，n为周期数
```

**核心特点**：
- 响应性：对价格变化反应更快，减少滞后性
- 权重递减：近期价格权重更大，远期价格权重递减
- 趋势敏感：能更早识别趋势转换点

**实现细节**：
```python
class EMAStrategy:
    """指数移动平均线策略"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化EMA策略
        
        Args:
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期  
            signal_period: 信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        data = data.copy()
        
        # 计算快慢EMA
        data['EMA_fast'] = self.calculate_ema(data['close'], self.fast_period)
        data['EMA_slow'] = self.calculate_ema(data['close'], self.slow_period)
        
        # 计算MACD线
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        
        # 计算信号线
        data['Signal'] = self.calculate_ema(data['MACD'], self.signal_period)
        
        # 计算MACD柱状图
        data['Histogram'] = data['MACD'] - data['Signal']
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_macd(data)
        
        # 初始化信号
        data['signal'] = 0
        
        # MACD金叉死叉信号
        data['golden_cross'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
        data['death_cross'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
        
        # 生成交易信号
        data.loc[data['golden_cross'], 'signal'] = 1  # 买入信号
        data.loc[data['death_cross'], 'signal'] = -1  # 卖出信号
        
        # 零轴突破信号
        data['zero_cross_up'] = (data['MACD'] > 0) & (data['MACD'].shift(1) <= 0)
        data['zero_cross_down'] = (data['MACD'] < 0) & (data['MACD'].shift(1) >= 0)
        
        return data
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """策略回测"""
        signals = self.generate_signals(data)
        
        # 计算收益
        signals['returns'] = data['close'].pct_change()
        signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
        signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
        
        return signals

# 使用示例
def demo_ema_strategy():
    """EMA策略演示"""
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # 生成价格数据
    returns = np.random.normal(0.001, 0.02, 500)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    
    # 运行策略
    strategy = EMAStrategy(fast_period=12, slow_period=26, signal_period=9)
    results = strategy.backtest(data)
    
    print("EMA策略回测完成")
    return results
```

**EMA vs SMA对比**：

| 特征 | EMA | SMA |
|------|-----|-----|
| 响应速度 | 快 | 慢 |
| 滞后性 | 较小 | 较大 |
| 噪音过滤 | 较弱 | 较强 |
| 趋势跟踪 | 敏感 | 稳定 |
| 假信号 | 较多 | 较少 |

#### 1.1.3 自适应移动平均线(AMA, KAMA)策略

**策略原理**：
AMA根据市场波动性自动调整平滑常数，在趋势市场中快速响应，在震荡市场中减少噪音。

**KAMA算法实现**：
```python
class KAMAStrategy:
    """考夫曼自适应移动平均线策略"""
    
    def __init__(self, period=20, fast_sc=2, slow_sc=30):
        """
        初始化KAMA策略
        
        Args:
            period: 效率比率计算周期
            fast_sc: 快速平滑常数
            slow_sc: 慢速平滑常数
        """
        self.period = period
        self.fast_sc = fast_sc
        self.slow_sc = slow_sc
    
    def calculate_kama(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算KAMA指标"""
        data = data.copy()
        
        # 计算方向性(Direction)
        data['Direction'] = data['close'].diff(self.period).abs()
        
        # 计算波动性(Volatility)
        data['Volatility'] = data['close'].diff().abs().rolling(self.period).sum()
        
        # 计算效率比率(Efficiency Ratio)
        data['ER'] = data['Direction'] / data['Volatility']
        data['ER'] = data['ER'].fillna(0)
        
        # 计算平滑常数(Smoothing Constant)
        fast_alpha = 2 / (self.fast_sc + 1)
        slow_alpha = 2 / (self.slow_sc + 1)
        data['SC'] = (data['ER'] * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # 计算KAMA
        kama_values = [data['close'].iloc[0]]  # 初始值
        
        for i in range(1, len(data)):
            if pd.isna(data['SC'].iloc[i]):
                kama_values.append(kama_values[-1])
            else:
                kama_new = kama_values[-1] + data['SC'].iloc[i] * (data['close'].iloc[i] - kama_values[-1])
                kama_values.append(kama_new)
        
        data['KAMA'] = kama_values
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_kama(data)
        
        # 价格与KAMA的关系
        data['price_above_kama'] = data['close'] > data['KAMA']
        data['price_below_kama'] = data['close'] < data['KAMA']
        
        # 信号生成
        data['signal'] = 0
        data.loc[data['price_above_kama'] & ~data['price_above_kama'].shift(1), 'signal'] = 1
        data.loc[data['price_below_kama'] & ~data['price_below_kama'].shift(1), 'signal'] = -1
        
        # KAMA斜率信号
        data['kama_slope'] = data['KAMA'].diff()
        data['kama_rising'] = data['kama_slope'] > 0
        data['kama_falling'] = data['kama_slope'] < 0
        
        return data
```

**KAMA优势**：
- 自适应性：根据市场效率自动调整
- 减少假信号：在震荡市场中更稳定
- 趋势敏感：在趋势市场中响应迅速

### 1.2 突破策略族

#### 1.2.1 唐奇安通道突破策略

**策略原理**：
基于价格突破过去N天的最高价(上轨)或最低价(下轨)来判断趋势突破，是经典的趋势跟踪策略。

**核心概念**：
- 上轨：过去N天最高价
- 下轨：过去N天最低价
- 突破：价格超越通道边界

**实现细节**：
```python
class DonchianChannelStrategy:
    """唐奇安通道突破策略"""
    
    def __init__(self, entry_period=20, exit_period=10):
        """
        初始化策略参数
        
        Args:
            entry_period: 入场通道周期
            exit_period: 出场通道周期
        """
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.position = 0  # 当前仓位：1多头，-1空头，0空仓
    
    def calculate_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算唐奇安通道"""
        data = data.copy()
        
        # 入场通道
        data['upper_channel'] = data['high'].rolling(window=self.entry_period).max()
        data['lower_channel'] = data['low'].rolling(window=self.entry_period).min()
        data['middle_channel'] = (data['upper_channel'] + data['lower_channel']) / 2
        
        # 出场通道
        data['exit_upper'] = data['high'].rolling(window=self.exit_period).max()
        data['exit_lower'] = data['low'].rolling(window=self.exit_period).min()
        
        # 通道宽度
        data['channel_width'] = data['upper_channel'] - data['lower_channel']
        data['channel_width_pct'] = data['channel_width'] / data['middle_channel']
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_channels(data)
        
        # 入场信号
        data['long_entry'] = data['close'] > data['upper_channel'].shift(1)
        data['short_entry'] = data['close'] < data['lower_channel'].shift(1)
        
        # 出场信号
        data['long_exit'] = data['close'] < data['exit_lower'].shift(1)
        data['short_exit'] = data['close'] > data['exit_upper'].shift(1)
        
        # 生成仓位信号
        data['position'] = 0
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
            
            data['position'].iloc[i] = current_position
        
        # 计算信号变化
        data['signal'] = data['position'].diff()
        
        return data
    
    def add_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加过滤条件"""
        data = data.copy()
        
        # 成交量过滤
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_filter'] = data['volume'] > data['volume_ma'] * 1.5
        else:
            data['volume_filter'] = True
        
        # ATR过滤（避免在低波动期交易）
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(14).mean()
        data['atr_filter'] = data['atr'] > data['atr'].rolling(50).quantile(0.3)
        
        # 应用过滤条件
        data['filtered_long_entry'] = data['long_entry'] & data['volume_filter'] & data['atr_filter']
        data['filtered_short_entry'] = data['short_entry'] & data['volume_filter'] & data['atr_filter']
        
        return data
    
    def backtest_with_stops(self, data: pd.DataFrame, stop_loss_pct=0.02) -> pd.DataFrame:
        """带止损的回测"""
        data = self.generate_signals(data)
        data = self.add_filters(data)
        
        # 计算收益
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        # 止损逻辑
        data['stop_loss'] = False
        current_position = 0
        entry_price = 0
        
        for i in range(1, len(data)):
            if data['position'].iloc[i] != current_position:
                current_position = data['position'].iloc[i]
                entry_price = data['close'].iloc[i]
            
            # 检查止损
            if current_position > 0:  # 多头仓位
                if data['close'].iloc[i] < entry_price * (1 - stop_loss_pct):
                    data['stop_loss'].iloc[i] = True
                    current_position = 0
            elif current_position < 0:  # 空头仓位
                if data['close'].iloc[i] > entry_price * (1 + stop_loss_pct):
                    data['stop_loss'].iloc[i] = True
                    current_position = 0
        
        return data

# 使用示例
def demo_donchian_strategy():
    """唐奇安通道策略演示"""
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 生成趋势性价格数据
    returns = np.random.normal(0.0008, 0.025, 500)
    trend = np.sin(np.linspace(0, 4*np.pi, 500)) * 0.001  # 添加周期性趋势
    prices = 100 * np.exp(np.cumsum(returns + trend))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    # 运行策略
    strategy = DonchianChannelStrategy(entry_period=20, exit_period=10)
    results = strategy.backtest_with_stops(data, stop_loss_pct=0.03)
    
    # 计算业绩指标
    total_return = (1 + results['strategy_returns']).prod() - 1
    annual_return = results['strategy_returns'].mean() * 252
    volatility = results['strategy_returns'].std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    print(f"唐奇安通道策略回测结果:")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"年化波动率: {volatility:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    
    return results
```

**参数优化建议**：
- 入场周期：10-40天，常用20天
- 出场周期：入场周期的1/2，常用10天
- 过滤条件：成交量放大、ATR高于历史分位数

**适用市场**：
- 趋势性强的市场
- 波动率适中的品种
- 流动性好的标的

#### 1.2.2 布林带突破策略

**策略原理**：
基于价格突破布林带上下轨来识别趋势突破，结合均值回归特性。布林带由中轨(移动平均线)、上轨(中轨+N倍标准差)和下轨(中轨-N倍标准差)组成。

**核心概念**：
- 中轨：通常为20日简单移动平均线
- 上下轨：中轨±2倍标准差
- 带宽：反映市场波动性
- 带位：价格在布林带中的相对位置

```python
class BollingerBandsStrategy:
    """布林带突破策略"""
    
    def __init__(self, period=20, std_dev=2, volume_factor=1.5):
        """
        初始化布林带策略
        
        Args:
            period: 移动平均线周期
            std_dev: 标准差倍数
            volume_factor: 成交量放大倍数
        """
        self.period = period
        self.std_dev = std_dev
        self.volume_factor = volume_factor
    
    def calculate_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标"""
        data = data.copy()
        
        # 计算中轨(移动平均线)
        data['BB_middle'] = data['close'].rolling(window=self.period).mean()
        
        # 计算标准差
        rolling_std = data['close'].rolling(window=self.period).std()
        
        # 计算上下轨
        data['BB_upper'] = data['BB_middle'] + (rolling_std * self.std_dev)
        data['BB_lower'] = data['BB_middle'] - (rolling_std * self.std_dev)
        
        # 计算布林带宽度
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        
        # 计算价格在布林带中的位置(%B)
        data['BB_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        
        # 计算带宽的历史分位数
        data['BB_width_percentile'] = data['BB_width'].rolling(252).rank(pct=True)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_bands(data)
        
        # 成交量过滤
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_surge'] = data['volume'] > data['volume_ma'] * self.volume_factor
        else:
            data['volume_surge'] = True
        
        # 突破信号(高波动期)
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
        
        # 均值回归信号(低波动期)
        low_volatility = data['BB_width_percentile'] < 0.3
        data['mean_revert_long'] = (
            (data['close'] < data['BB_lower']) & 
            low_volatility
        )
        data['mean_revert_short'] = (
            (data['close'] > data['BB_upper']) & 
            low_volatility
        )
        
        # 挤压信号(布林带收缩后的扩张)
        data['squeeze'] = data['BB_width'] < data['BB_width'].rolling(50).quantile(0.2)
        data['squeeze_breakout'] = (
            data['squeeze'].shift(1) & 
            ~data['squeeze'] & 
            data['volume_surge']
        )
        
        return data
    
    def backtest_dual_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """双重策略回测(突破+均值回归)"""
        data = self.generate_signals(data)
        
        # 策略权重分配
        data['trend_weight'] = data['BB_width_percentile']  # 高波动期权重更大
        data['mean_revert_weight'] = 1 - data['trend_weight']
        
        # 突破策略信号
        data['breakout_signal'] = 0
        data.loc[data['breakout_long'], 'breakout_signal'] = 1
        data.loc[data['breakout_short'], 'breakout_signal'] = -1
        
        # 均值回归策略信号
        data['mean_revert_signal'] = 0
        data.loc[data['mean_revert_long'], 'mean_revert_signal'] = 1
        data.loc[data['mean_revert_short'], 'mean_revert_signal'] = -1
        
        # 组合信号
        data['combined_signal'] = (
            data['breakout_signal'] * data['trend_weight'] + 
            data['mean_revert_signal'] * data['mean_revert_weight']
        )
        
        # 计算收益
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['combined_signal'].shift(1) * data['returns']
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
        
        return data

# 使用示例
def demo_bollinger_strategy():
    """布林带策略演示"""
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 生成具有周期性波动的价格数据
    base_trend = np.linspace(0, 0.2, 500)
    volatility_cycle = 0.01 + 0.015 * np.sin(np.linspace(0, 8*np.pi, 500))
    returns = np.random.normal(base_trend/500, volatility_cycle)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 500))),
        'volume': np.random.randint(500000, 5000000, 500)
    }, index=dates)
    
    # 运行策略
    strategy = BollingerBandsStrategy(period=20, std_dev=2)
    results = strategy.backtest_dual_strategy(data)
    
    print("布林带策略回测完成")
    return results
```

### 1.3 动量策略族

#### 1.3.1 RSI动量策略

**策略原理**：
RSI(相对强弱指数)测量价格变动的速度和幅度，识别超买超卖条件和动量转换点。RSI值在0-100之间波动，通常70以上为超买，30以下为超卖。

**核心特点**：
- 震荡指标：在0-100区间内波动
- 超买超卖：识别价格极端位置
- 背离信号：价格与RSI走势不一致

```python
class RSIStrategy:
    """RSI动量策略"""
    
    def __init__(self, period=14, overbought=70, oversold=30, rsi_ma_period=5):
        """
        初始化RSI策略
        
        Args:
            period: RSI计算周期
            overbought: 超买阈值
            oversold: 超卖阈值
            rsi_ma_period: RSI移动平均周期
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.rsi_ma_period = rsi_ma_period
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标"""
        data = data.copy()
        
        # 计算价格变化
        delta = data['close'].diff()
        
        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        # 计算相对强度和RSI
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI移动平均线
        data['RSI_MA'] = data['RSI'].rolling(window=self.rsi_ma_period).mean()
        
        # RSI动量
        data['RSI_momentum'] = data['RSI'].diff(3)
        
        return data
    
    def detect_divergence(self, data: pd.DataFrame, lookback=10) -> pd.DataFrame:
        """检测价格与RSI背离"""
        data = data.copy()
        
        # 寻找局部高点和低点
        data['price_high'] = data['close'].rolling(window=lookback, center=True).max() == data['close']
        data['price_low'] = data['close'].rolling(window=lookback, center=True).min() == data['close']
        data['rsi_high'] = data['RSI'].rolling(window=lookback, center=True).max() == data['RSI']
        data['rsi_low'] = data['RSI'].rolling(window=lookback, center=True).min() == data['RSI']
        
        # 初始化背离信号
        data['bullish_divergence'] = False
        data['bearish_divergence'] = False
        
        # 检测背离(简化版本)
        for i in range(lookback, len(data) - lookback):
            # 寻找前面的高点/低点
            recent_price_highs = data['close'][data['price_high']][max(0, i-50):i]
            recent_rsi_highs = data['RSI'][data['rsi_high']][max(0, i-50):i]
            
            if len(recent_price_highs) >= 2 and len(recent_rsi_highs) >= 2:
                if (recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2] and 
                    recent_rsi_highs.iloc[-1] < recent_rsi_highs.iloc[-2]):
                    data.loc[data.index[i], 'bearish_divergence'] = True
            
            recent_price_lows = data['close'][data['price_low']][max(0, i-50):i]
            recent_rsi_lows = data['RSI'][data['rsi_low']][max(0, i-50):i]
            
            if len(recent_price_lows) >= 2 and len(recent_rsi_lows) >= 2:
                if (recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2] and 
                    recent_rsi_lows.iloc[-1] > recent_rsi_lows.iloc[-2]):
                    data.loc[data.index[i], 'bullish_divergence'] = True
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_rsi(data)
        data = self.detect_divergence(data)
        
        # 传统超买超卖信号
        data['oversold_signal'] = (data['RSI'] < self.oversold) & (data['RSI'].shift(1) >= self.oversold)
        data['overbought_signal'] = (data['RSI'] > self.overbought) & (data['RSI'].shift(1) <= self.overbought)
        
        # RSI中线突破信号
        data['rsi_bull_cross'] = (data['RSI'] > 50) & (data['RSI'].shift(1) <= 50)
        data['rsi_bear_cross'] = (data['RSI'] < 50) & (data['RSI'].shift(1) >= 50)
        
        # RSI与其移动平均线的关系
        data['rsi_above_ma'] = data['RSI'] > data['RSI_MA']
        data['rsi_below_ma'] = data['RSI'] < data['RSI_MA']
        
        # 组合信号
        data['signal'] = 0
        
        # 买入信号
        buy_conditions = (
            data['oversold_signal'] |
            data['bullish_divergence'] |
            (data['rsi_bull_cross'] & data['rsi_above_ma'])
        )
        data.loc[buy_conditions, 'signal'] = 1
        
        # 卖出信号
        sell_conditions = (
            data['overbought_signal'] |
            data['bearish_divergence'] |
            (data['rsi_bear_cross'] & data['rsi_below_ma'])
        )
        data.loc[sell_conditions, 'signal'] = -1
        
        return data

# RSI策略使用示例
def demo_rsi_strategy():
    """RSI策略演示"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 生成有趋势的价格数据
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
    
    print("RSI策略回测完成")
    return results
```

#### 1.3.2 MACD动量策略

**策略原理**：
MACD通过快慢EMA差值识别动量变化，MACD柱状图显示动量强弱变化趋势。

```python
class MACDStrategy:
    """MACD动量策略"""
    
    def __init__(self, fast=12, slow=26, signal=9):
        """
        初始化MACD策略
        
        Args:
            fast: 快速EMA周期
            slow: 慢速EMA周期
            signal: 信号线EMA周期
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        data = data.copy()
        
        # 计算快慢EMA
        data['EMA_fast'] = data['close'].ewm(span=self.fast).mean()
        data['EMA_slow'] = data['close'].ewm(span=self.slow).mean()
        
        # 计算MACD线
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        
        # 计算信号线
        data['MACD_signal'] = data['MACD'].ewm(span=self.signal).mean()
        
        # 计算MACD柱状图
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        data = self.calculate_macd(data)
        
        # MACD金叉死叉
        data['golden_cross'] = (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))
        data['death_cross'] = (data['MACD'] < data['MACD_signal']) & (data['MACD'].shift(1) >= data['MACD_signal'].shift(1))
        
        # MACD零轴突破
        data['zero_cross_up'] = (data['MACD'] > 0) & (data['MACD'].shift(1) <= 0)
        data['zero_cross_down'] = (data['MACD'] < 0) & (data['MACD'].shift(1) >= 0)
        
        # MACD柱状图信号
        data['histogram_increasing'] = data['MACD_histogram'] > data['MACD_histogram'].shift(1)
        data['histogram_decreasing'] = data['MACD_histogram'] < data['MACD_histogram'].shift(1)
        data['histogram_turning_up'] = (data['MACD_histogram'] > 0) & (data['MACD_histogram'].shift(1) <= 0)
        data['histogram_turning_down'] = (data['MACD_histogram'] < 0) & (data['MACD_histogram'].shift(1) >= 0)
        
        # 组合信号
        data['signal'] = 0
        
        # 买入信号
        buy_conditions = (
            data['golden_cross'] |
            data['zero_cross_up'] |
            data['histogram_turning_up']
        )
        data.loc[buy_conditions, 'signal'] = 1
        
        # 卖出信号
        sell_conditions = (
            data['death_cross'] |
            data['zero_cross_down'] |
            data['histogram_turning_down']
        )
        data.loc[sell_conditions, 'signal'] = -1
        
        return data
```
