## 二、均值回归策略体系

**策略体系概述**：
均值回归策略基于"价格会向长期均值回归"的理论假设，通过识别价格偏离均值的时机进行反向交易。这类策略在震荡市场中表现较好，但在趋势市场中可能面临较大风险。

**核心理论基础**：
- **均值回归理论**：价格围绕内在价值波动，偏离后会回归
- **统计套利理论**：利用价格关系的统计特性获取收益
- **协整理论**：寻找长期稳定的价格关系

### 2.1 统计套利策略族

#### 2.1.1 配对交易策略

**策略原理**：
选择历史上价格走势高度相关的两只股票，当价差偏离历史均值时进行反向操作，等待价差回归。配对交易是市场中性策略，理论上不受市场整体方向影响。

**核心要素**：
- **股票选择**：基本面相似、行业相关、历史相关性高
- **价差计算**：Stock1 - β × Stock2，其中β为对冲比率
- **信号生成**：价差的Z-score超过阈值时触发交易
- **风险控制**：设置止损、时间止损、相关性监控

```python
import statsmodels.api as sm
from scipy.stats import zscore
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

class PairsTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, lookback_period=60, entry_threshold=2.0, exit_threshold=0.5, 
                 stop_loss=3.0, max_holding_days=30):
        """
        初始化配对交易策略
        
        Args:
            lookback_period: 价差计算的回望期
            entry_threshold: 入场Z-score阈值
            exit_threshold: 出场Z-score阈值
            stop_loss: 止损Z-score阈值
            max_holding_days: 最大持仓天数
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        self.current_position = 0  # 当前仓位状态
        self.entry_date = None
    
    def find_cointegrated_pairs(self, price_data: pd.DataFrame, min_correlation=0.7) -> List[Tuple]:
        """寻找协整的股票对"""
        from statsmodels.tsa.stattools import coint
        
        n = price_data.shape[1]
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = price_data.iloc[:, i].dropna()
                stock2 = price_data.iloc[:, j].dropna()
                
                # 确保有足够的重叠数据
                common_index = stock1.index.intersection(stock2.index)
                if len(common_index) < 100:  # 至少100个交易日
                    continue
                
                stock1_aligned = stock1.loc[common_index]
                stock2_aligned = stock2.loc[common_index]
                
                # 检查相关性
                correlation = stock1_aligned.corr(stock2_aligned)
                if abs(correlation) < min_correlation:
                    continue
                
                # 协整检验
                try:
                    score, pvalue, _ = coint(stock1_aligned, stock2_aligned)
                    if pvalue < 0.05:  # 5%显著性水平
                        pairs.append({
                            'stock1': price_data.columns[i],
                            'stock2': price_data.columns[j],
                            'pvalue': pvalue,
                            'correlation': correlation,
                            'adf_stat': score
                        })
                except:
                    continue
        
        return sorted(pairs, key=lambda x: x['pvalue'])
    
    def calculate_spread(self, stock1: pd.Series, stock2: pd.Series) -> Tuple[pd.Series, float, Dict]:
        """计算价差和对冲比率"""
        # 对齐数据
        common_index = stock1.index.intersection(stock2.index)
        stock1_aligned = stock1.loc[common_index]
        stock2_aligned = stock2.loc[common_index]
        
        # 计算对冲比率(使用OLS回归)
        X = sm.add_constant(stock2_aligned)
        model = sm.OLS(stock1_aligned, X).fit()
        hedge_ratio = model.params[1]
        
        # 计算价差
        spread = stock1_aligned - hedge_ratio * stock2_aligned
        
        # 计算回归统计信息
        stats = {
            'hedge_ratio': hedge_ratio,
            'intercept': model.params[0],
            'r_squared': model.rsquared,
            'p_value': model.pvalues[1],
            'std_error': model.bse[1]
        }
        
        return spread, hedge_ratio, stats
    
    def calculate_rolling_zscore(self, spread: pd.Series) -> pd.Series:
        """计算滚动Z-score"""
        rolling_mean = spread.rolling(window=self.lookback_period).mean()
        rolling_std = spread.rolling(window=self.lookback_period).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self, stock1: pd.Series, stock2: pd.Series) -> Tuple[pd.DataFrame, Dict]:
        """生成交易信号"""
        spread, hedge_ratio, stats = self.calculate_spread(stock1, stock2)
        zscore = self.calculate_rolling_zscore(spread)
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['hedge_ratio'] = hedge_ratio
        
        # 基础交易信号
        signals['long_entry'] = zscore < -self.entry_threshold  # 价差过低，买入价差
        signals['short_entry'] = zscore > self.entry_threshold   # 价差过高，卖出价差
        signals['exit'] = abs(zscore) < self.exit_threshold      # 价差回归，平仓
        
        # 止损信号
        signals['stop_loss'] = abs(zscore) > self.stop_loss
        
        # 生成仓位信号
        signals['position'] = 0
        current_pos = 0
        entry_date = None
        
        for i, (date, row) in enumerate(signals.iterrows()):
            # 检查时间止损
            if entry_date and (date - entry_date).days > self.max_holding_days:
                current_pos = 0
                entry_date = None
            
            # 检查止损
            if current_pos != 0 and row['stop_loss']:
                current_pos = 0
                entry_date = None
            
            # 检查出场信号
            if current_pos != 0 and row['exit']:
                current_pos = 0
                entry_date = None
            
            # 检查入场信号
            if current_pos == 0:
                if row['long_entry']:
                    current_pos = 1
                    entry_date = date
                elif row['short_entry']:
                    current_pos = -1
                    entry_date = date
            
            signals.iloc[i, signals.columns.get_loc('position')] = current_pos
        
        return signals, stats
    
    def calculate_portfolio_returns(self, stock1: pd.Series, stock2: pd.Series, 
                                  signals: pd.DataFrame) -> pd.DataFrame:
        """计算投资组合收益"""
        # 股票收益率
        ret1 = stock1.pct_change()
        ret2 = stock2.pct_change()
        
        # 配对交易收益 = position * (ret1 - hedge_ratio * ret2)
        hedge_ratio = signals['hedge_ratio'].iloc[0]  # 使用固定对冲比率
        pair_returns = signals['position'].shift(1) * (ret1 - hedge_ratio * ret2)
        
        # 组合统计
        portfolio_stats = pd.DataFrame(index=signals.index)
        portfolio_stats['stock1_returns'] = ret1
        portfolio_stats['stock2_returns'] = ret2
        portfolio_stats['pair_returns'] = pair_returns
        portfolio_stats['cumulative_returns'] = (1 + pair_returns).cumprod()
        portfolio_stats['position'] = signals['position']
        portfolio_stats['spread'] = signals['spread']
        portfolio_stats['zscore'] = signals['zscore']
        
        return portfolio_stats
    
    def backtest_pairs(self, stock1: pd.Series, stock2: pd.Series) -> Dict:
        """配对交易回测"""
        signals, spread_stats = self.generate_signals(stock1, stock2)
        portfolio_stats = self.calculate_portfolio_returns(stock1, stock2, signals)
        
        # 计算业绩指标
        returns = portfolio_stats['pair_returns'].dropna()
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 交易统计
        positions = signals['position']
        trades = (positions != positions.shift(1)) & (positions != 0)
        num_trades = trades.sum()
        
        # 胜率统计
        trade_returns = []
        current_trade_start = None
        
        for i, pos in enumerate(positions):
            if pos != 0 and current_trade_start is None:
                current_trade_start = i
            elif pos == 0 and current_trade_start is not None:
                trade_ret = portfolio_stats['pair_returns'].iloc[current_trade_start:i].sum()
                trade_returns.append(trade_ret)
                current_trade_start = None
        
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

# 配对交易使用示例
def demo_pairs_trading():
    """配对交易策略演示"""
    # 创建模拟的两只相关股票数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 生成共同趋势
    common_trend = np.cumsum(np.random.normal(0.0005, 0.01, 500))
    
    # 股票1：基础价格 + 共同趋势 + 个股噪音
    stock1_specific = np.cumsum(np.random.normal(0, 0.008, 500))
    stock1_prices = 100 * np.exp(common_trend + stock1_specific * 0.3)
    
    # 股票2：基础价格 + 共同趋势 + 个股噪音
    stock2_specific = np.cumsum(np.random.normal(0, 0.008, 500))
    stock2_prices = 80 * np.exp(common_trend + stock2_specific * 0.3)
    
    stock1 = pd.Series(stock1_prices, index=dates)
    stock2 = pd.Series(stock2_prices, index=dates)
    
    # 运行配对交易策略
    strategy = PairsTradingStrategy(
        lookback_period=60, 
        entry_threshold=2.0, 
        exit_threshold=0.5,
        stop_loss=3.0,
        max_holding_days=30
    )
    
    results = strategy.backtest_pairs(stock1, stock2)
    
    # 打印结果
    print("配对交易回测结果:")
    for key, value in results['performance'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return results

# 运行演示
if __name__ == "__main__":
    pairs_results = demo_pairs_trading()
```

**实际应用案例**：

1. **A股经典配对**：
   - 中国平安(601318) vs 中国人寿(601628) - 保险板块
   - 茅台(600519) vs 五粮液(000858) - 白酒板块
   - 招商银行(600036) vs 平安银行(000001) - 银行板块

2. **美股科技股配对**：
   - Apple(AAPL) vs Microsoft(MSFT)
   - Google(GOOGL) vs Facebook(META)
   - Amazon(AMZN) vs Netflix(NFLX)

3. **ETF配对**：
   - SPY vs QQQ（大盘vs科技）
   - XLF vs XLE（金融vs能源）
   - IWM vs QQQ（小盘vs大盘科技）

**配对选择标准**：

```python
class PairSelectionCriteria:
    """配对选择标准"""
    
    @staticmethod
    def correlation_test(stock1: pd.Series, stock2: pd.Series, min_corr=0.7) -> bool:
        """相关性测试"""
        correlation = stock1.corr(stock2)
        return abs(correlation) >= min_corr
    
    @staticmethod
    def cointegration_test(stock1: pd.Series, stock2: pd.Series, significance=0.05) -> Tuple[bool, float]:
        """协整检验"""
        from statsmodels.tsa.stattools import coint
        try:
            _, pvalue, _ = coint(stock1, stock2)
            return pvalue < significance, pvalue
        except:
            return False, 1.0
    
    @staticmethod
    def fundamental_similarity(stock1_info: Dict, stock2_info: Dict) -> float:
        """基本面相似度评分"""
        # 行业相似度
        industry_score = 1.0 if stock1_info.get('industry') == stock2_info.get('industry') else 0.0
        
        # 市值相似度
        mc1, mc2 = stock1_info.get('market_cap', 0), stock2_info.get('market_cap', 0)
        if mc1 > 0 and mc2 > 0:
            size_ratio = min(mc1, mc2) / max(mc1, mc2)
            size_score = size_ratio
        else:
            size_score = 0.0
        
        # 综合评分
        total_score = (industry_score * 0.6 + size_score * 0.4)
        return total_score
```

#### 2.1.2 ETF套利策略

**策略原理**：
利用ETF二级市场价格与净值(NAV)之间的偏离进行套利。ETF套利包括一级市场申赎套利和二级市场价差套利两种主要形式。

**核心机制**：
- **一级市场申赎**：大额资金可直接用一篮子股票申购或赎回ETF份额
- **二级市场交易**：像股票一样在交易所买卖ETF份额
- **套利机会**：当ETF价格与NAV存在显著偏离时进行套利

```python
class ETFArbitrageStrategy:
    """ETF套利策略"""
    
    def __init__(self, threshold=0.5, transaction_cost=0.002, min_arbitrage_amount=1000000):
        """
        初始化ETF套利策略
        
        Args:
            threshold: 套利阈值(%)
            transaction_cost: 交易成本(%)
            min_arbitrage_amount: 最小套利金额
        """
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.min_arbitrage_amount = min_arbitrage_amount
    
    def calculate_theoretical_nav(self, constituent_data: pd.DataFrame, 
                                weights: pd.Series) -> pd.Series:
        """计算理论净值"""
        # 确保权重和股票数据对齐
        aligned_data = constituent_data.reindex(columns=weights.index, fill_value=0)
        theoretical_nav = (aligned_data * weights).sum(axis=1)
        return theoretical_nav
    
    def calculate_premium_discount(self, etf_price: pd.Series, nav: pd.Series) -> pd.Series:
        """计算ETF溢价/折价率"""
        premium = (etf_price - nav) / nav * 100
        return premium
    
    def calculate_arbitrage_profit(self, premium: float, trade_amount: float) -> float:
        """计算套利利润（考虑交易成本）"""
        gross_profit = abs(premium) / 100 * trade_amount
        transaction_costs = self.transaction_cost * trade_amount * 2  # 买卖两次
        net_profit = gross_profit - transaction_costs
        return max(0, net_profit)
    
    def generate_arbitrage_signals(self, etf_data: pd.DataFrame, 
                                 constituent_data: pd.DataFrame,
                                 weights: pd.Series) -> pd.DataFrame:
        """生成套利信号"""
        # 计算理论NAV
        theoretical_nav = self.calculate_theoretical_nav(constituent_data, weights)
        
        # 计算溢价率
        premium = self.calculate_premium_discount(etf_data['price'], theoretical_nav)
        
        # 计算流动性指标
        etf_volume = etf_data.get('volume', pd.Series(0, index=etf_data.index))
        constituent_liquidity = constituent_data.sum(axis=1)  # 简化的流动性指标
        
        # 生成交易信号
        signals = pd.DataFrame(index=etf_data.index)
        signals['etf_price'] = etf_data['price']
        signals['theoretical_nav'] = theoretical_nav
        signals['premium'] = premium
        signals['etf_volume'] = etf_volume
        
        # 套利条件
        sufficient_volume = etf_volume > self.min_arbitrage_amount / etf_data['price']
        large_enough_premium = abs(premium) > self.threshold
        
        # 套利信号
        signals['arbitrage_long'] = (
            (premium < -self.threshold) &  # ETF被低估
            sufficient_volume &
            large_enough_premium
        )
        signals['arbitrage_short'] = (
            (premium > self.threshold) &   # ETF被高估
            sufficient_volume &
            large_enough_premium
        )
        
        # 计算潜在利润
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
        """日内套利策略"""
        
        # 重采样到指定频率
        etf_resampled = etf_data.resample(time_window).last()
        constituent_resampled = constituent_data.resample(time_window).last()
        
        signals = self.generate_arbitrage_signals(etf_resampled, constituent_resampled, weights)
        
        # 日内特殊考虑
        # 1. 开盘和收盘前后的套利机会
        market_open = signals.index.time == pd.Timestamp('09:30').time()
        market_close = signals.index.time >= pd.Timestamp('14:50').time()
        
        signals['market_open'] = market_open
        signals['market_close'] = market_close
        
        # 2. 调整套利阈值（开盘收盘时波动更大）
        adjusted_threshold = np.where(
            market_open | market_close,
            self.threshold * 1.5,  # 提高阈值
            self.threshold
        )
        
        # 重新计算信号
        signals['arbitrage_long'] = (
            (signals['premium'] < -adjusted_threshold) &
            (signals['etf_volume'] > 0) &
            (abs(signals['premium']) > adjusted_threshold)
        )
        
        signals['arbitrage_short'] = (
            (signals['premium'] > adjusted_threshold) &
            (signals['etf_volume'] > 0) &
            (abs(signals['premium']) > adjusted_threshold)
        )
        
        return signals
    
    def risk_management(self, signals: pd.DataFrame) -> pd.DataFrame:
        """风险管理"""
        signals = signals.copy()
        
        # 1. 连续套利限制（避免过度交易）
        signals['consecutive_trades'] = (
            (signals['arbitrage_long'] | signals['arbitrage_short'])
            .rolling(window=10)
            .sum()
        )
        
        # 2. 波动率过滤（高波动期间暂停套利）
        price_volatility = signals['etf_price'].pct_change().rolling(20).std()
        high_volatility = price_volatility > price_volatility.quantile(0.9)
        
        # 3. 应用风险控制
        signals['risk_adjusted_long'] = (
            signals['arbitrage_long'] &
            (signals['consecutive_trades'] < 3) &
            ~high_volatility
        )
        
        signals['risk_adjusted_short'] = (
            signals['arbitrage_short'] &
            (signals['consecutive_trades'] < 3) &
            ~high_volatility
        )
        
        return signals

# ETF套利使用示例
def demo_etf_arbitrage():
    """ETF套利策略演示"""
    # 创建模拟ETF和成分股数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01 09:30:00', periods=1000, freq='1min')
    
    # 模拟5只成分股
    n_stocks = 5
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], 
                       index=[f'stock_{i}' for i in range(n_stocks)])
    
    # 生成成分股价格（带相关性）
    base_price = 100
    stock_prices = {}
    
    for i, stock in enumerate(weights.index):
        # 添加一些随机游走和共同因子
        common_factor = np.cumsum(np.random.normal(0, 0.001, len(dates)))
        idiosyncratic = np.cumsum(np.random.normal(0, 0.0005, len(dates)))
        
        prices = base_price * (1 + i * 0.2) * np.exp(common_factor * 0.7 + idiosyncratic * 0.3)
        stock_prices[stock] = prices
    
    constituent_data = pd.DataFrame(stock_prices, index=dates)
    
    # 计算理论ETF价格并添加噪音
    theoretical_nav = (constituent_data * weights).sum(axis=1)
    noise = np.random.normal(0, 0.002, len(dates))  # 2%的价格噪音
    etf_prices = theoretical_nav * (1 + noise)
    
    etf_data = pd.DataFrame({
        'price': etf_prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # 运行套利策略
    strategy = ETFArbitrageStrategy(threshold=0.3, transaction_cost=0.001)
    signals = strategy.intraday_arbitrage_strategy(etf_data, constituent_data, weights)
    signals = strategy.risk_management(signals)
    
    # 统计套利机会
    arbitrage_opportunities = (
        signals['risk_adjusted_long'] | signals['risk_adjusted_short']
    ).sum()
    
    avg_premium = signals['premium'].abs().mean()
    max_potential_profit = signals['potential_profit'].max()
    
    print(f"ETF套利分析结果:")
    print(f"套利机会次数: {arbitrage_opportunities}")
    print(f"平均溢价率: {avg_premium:.3f}%")
    print(f"最大潜在利润: ${max_potential_profit:,.2f}")
    
    return signals

# 运行演示
if __name__ == "__main__":
    etf_results = demo_etf_arbitrage()
```

**ETF套利类型**：

1. **现金套利**：
   - 适用于现金创设型ETF
   - 利用申购赎回价格差异
   - 风险较低，收益稳定

2. **实物套利**：
   - 需要准备一篮子成分股
   - 套利空间通常更大
   - 对资金量和操作能力要求高

3. **跨市场套利**：
   - 同一ETF在不同市场的价差
   - 如港股ETF vs A股ETF
   - 需考虑汇率风险

### 2.2 协整策略族

#### 2.2.1 向量误差修正模型(VECM)策略

**策略原理**：
VECM模型用于分析多个协整序列之间的短期动态调整和长期均衡关系。当价格偏离长期均衡时，误差修正机制会推动价格回归均衡。

**核心概念**：
- **协整关系**：多个时间序列之间的长期稳定关系
- **误差修正项**：当期价格对长期均衡的偏离程度
- **调整速度**：价格向均衡回归的速度参数

```python
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class VECMStrategy:
    """向量误差修正模型策略"""
    
    def __init__(self, lag_order=1, lookback_window=252):
        """
        初始化VECM策略
        
        Args:
            lag_order: 滞后阶数
            lookback_window: 回望窗口期
        """
        self.lag_order = lag_order
        self.lookback_window = lookback_window
        self.model = None
        
    def test_cointegration(self, price_data: pd.DataFrame, significance_level=0.05) -> Dict:
        """Johansen协整检验"""
        try:
            result = coint_johansen(price_data.dropna(), det_order=0, k_ar_diff=self.lag_order)
            
            # 提取检验结果
            trace_stats = result.lr1  # 迹统计量
            max_eigen_stats = result.lr2  # 最大特征值统计量
            crit_values_trace = result.cvt  # 迹统计量临界值
            crit_values_max_eigen = result.cvm  # 最大特征值临界值
            
            # 确定协整关系数量
            significance_idx = {'10%': 0, '5%': 1, '1%': 2}[f'{int(significance_level*100)}%']
            
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
            print(f"协整检验失败: {e}")
            return None
    
    def fit_vecm(self, price_data: pd.DataFrame, coint_rank=1) -> Tuple[object, Dict]:
        """拟合VECM模型"""
        try:
            # 首先进行协整检验
            coint_test = self.test_cointegration(price_data)
            if coint_test is None:
                return None, {}
            
            # 建议使用的协整关系数量
            recommended_rank = min(coint_test['n_coint_trace'], coint_test['n_coint_max_eigen'])
            actual_rank = min(coint_rank, recommended_rank, len(price_data.columns) - 1)
            
            if actual_rank <= 0:
                print("未发现协整关系")
                return None, {}
            
            # 拟合VECM模型
            self.model = VECM(
                price_data.dropna(), 
                k_ar_diff=self.lag_order,
                coint_rank=actual_rank,
                deterministic='ci'  # 包含常数项
            )
            vecm_result = self.model.fit()
            
            # 提取关键参数
            model_info = {
                'alpha': vecm_result.alpha,  # 调整系数
                'beta': vecm_result.beta,    # 协整向量
                'gamma': vecm_result.gamma,  # 短期调整系数
                'coint_rank': actual_rank,
                'log_likelihood': vecm_result.llf,
                'aic': vecm_result.aic,
                'bic': vecm_result.bic
            }
            
            return vecm_result, model_info
            
        except Exception as e:
            print(f"VECM模型拟合失败: {e}")
            return None, {}
    
    def calculate_error_correction_term(self, price_data: pd.DataFrame, 
                                      beta: np.ndarray) -> pd.Series:
        """计算误差修正项"""
        # 协整关系：beta' * price_data
        # 这里假设第一个协整向量
        coint_vector = beta[:, 0] if beta.ndim > 1 else beta
        error_correction = np.dot(price_data.values, coint_vector)
        
        return pd.Series(error_correction, index=price_data.index)
    
    def generate_trading_signals(self, price_data: pd.DataFrame, 
                               entry_threshold=1.5, exit_threshold=0.5) -> pd.DataFrame:
        """生成交易信号"""
        signals = pd.DataFrame(index=price_data.index)
        
        # 滚动拟合VECM模型
        for i in range(self.lookback_window, len(price_data)):
            window_data = price_data.iloc[i-self.lookback_window:i]
            
            # 拟合模型
            vecm_result, model_info = self.fit_vecm(window_data)
            
            if vecm_result is None:
                signals.iloc[i] = 0
                continue
            
            # 计算当前误差修正项
            current_data = price_data.iloc[i-1:i]  # 当前时点
            ect = self.calculate_error_correction_term(current_data, model_info['beta'])
            
            # 标准化误差修正项
            historical_ect = self.calculate_error_correction_term(window_data, model_info['beta'])
            ect_mean = historical_ect.mean()
            ect_std = historical_ect.std()
            
            if ect_std > 0:
                ect_zscore = (ect.iloc[0] - ect_mean) / ect_std
            else:
                ect_zscore = 0
            
            # 生成信号
            if ect_zscore > entry_threshold:
                # 协整关系高于均值，预期回归，做空组合
                signal = -1
            elif ect_zscore < -entry_threshold:
                # 协整关系低于均值，预期回归，做多组合
                signal = 1
            elif abs(ect_zscore) < exit_threshold:
                # 接近均值，平仓
                signal = 0
            else:
                # 保持当前仓位
                signal = signals.iloc[i-1] if i > 0 else 0
            
            # 存储信号和相关信息
            signals.loc[price_data.index[i], 'signal'] = signal
            signals.loc[price_data.index[i], 'ect'] = ect.iloc[0]
            signals.loc[price_data.index[i], 'ect_zscore'] = ect_zscore
            signals.loc[price_data.index[i], 'alpha_speed'] = model_info['alpha'][0, 0]  # 第一个变量的调整速度
        
        return signals.fillna(0)
    
    def portfolio_weights_from_cointegration(self, beta: np.ndarray, 
                                           normalize=True) -> np.ndarray:
        """从协整向量计算投资组合权重"""
        # 使用第一个协整向量
        weights = beta[:, 0] if beta.ndim > 1 else beta
        
        if normalize:
            # 归一化权重使其和为1
            weights = weights / np.sum(np.abs(weights))
        
        return weights
    
    def backtest_vecm_strategy(self, price_data: pd.DataFrame) -> Dict:
        """VECM策略回测"""
        signals = self.generate_trading_signals(price_data)
        
        # 计算收益率
        returns = price_data.pct_change()
        
        # 假设等权重投资组合（可以用协整向量优化）
        portfolio_returns = returns.mean(axis=1)
        
        # 应用交易信号
        strategy_returns = signals['signal'].shift(1) * portfolio_returns
        
        # 计算业绩指标
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = strategy_returns.mean() * 252
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + strategy_returns).cumprod()
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

# VECM策略使用示例
def demo_vecm_strategy():
    """VECM策略演示"""
    # 创建3只协整股票的模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 生成协整的价格序列
    # 共同趋势
    common_trend = np.cumsum(np.random.normal(0, 0.01, 500))
    
    # 三只股票，有长期协整关系但短期可能偏离
    stock1 = 100 * np.exp(common_trend + np.cumsum(np.random.normal(0, 0.005, 500)))
    stock2 = 80 * np.exp(common_trend * 1.2 + np.cumsum(np.random.normal(0, 0.005, 500)))
    stock3 = 120 * np.exp(common_trend * 0.8 + np.cumsum(np.random.normal(0, 0.005, 500)))
    
    price_data = pd.DataFrame({
        'stock1': stock1,
        'stock2': stock2,
        'stock3': stock3
    }, index=dates)
    
    # 运行VECM策略
    strategy = VECMStrategy(lag_order=1, lookback_window=120)
    results = strategy.backtest_vecm_strategy(price_data)
    
    # 打印结果
    print("VECM策略回测结果:")
    for key, value in results['performance'].items():
        print(f"{key}: {value:.4f}")
    
    return results

# 运行演示
if __name__ == "__main__":
    vecm_results = demo_vecm_strategy()
```
