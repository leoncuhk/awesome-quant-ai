
## 八、期权策略体系

### 8.1 Delta中性策略

```python
class DeltaNeutralStrategy:
    def __init__(self, rebalance_threshold=0.05):
        self.rebalance_threshold = rebalance_threshold
        self.positions = {}
        
    def calculate_greeks(self, stock_price, strike_price, time_to_expiry, 
                        risk_free_rate, volatility, option_type='call'):
        """计算期权希腊字母"""
        from scipy.stats import norm
        
        d1 = (np.log(stock_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (stock_price * volatility * np.sqrt(time_to_expiry))
        
        # Theta
        theta_call = (-stock_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                     risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        
        if option_type == 'call':
            theta = theta_call / 365
        else:
            theta = (theta_call + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry)) / 365
        
        # Vega
        vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    def create_delta_neutral_portfolio(self, option_positions, stock_price, market_data):
        """创建Delta中性投资组合"""
        total_delta = 0
        portfolio_value = 0
        
        # 计算期权组合的总Delta
        for position in option_positions:
            greeks = self.calculate_greeks(
                stock_price,
                position['strike'],
                position['time_to_expiry'],
                position['risk_free_rate'],
                position['volatility'],
                position['option_type']
            )
            
            position_delta = greeks['delta'] * position['quantity'] * position['multiplier']
            total_delta += position_delta
            portfolio_value += position['market_value']
        
        # 计算需要的股票数量来中性化Delta
        stock_hedge_quantity = -total_delta
        
        return {
            'stock_hedge_quantity': stock_hedge_quantity,
            'total_delta': total_delta,
            'portfolio_value': portfolio_value
        }
    
    def rebalance_portfolio(self, current_positions, new_stock_price, market_data):
        """重新平衡投资组合"""
        # 重新计算当前Delta
        current_delta = 0
        
        for position in current_positions['options']:
            # 更新期权参数
            position['time_to_expiry'] = max(position['time_to_expiry'] - 1/365, 0.001)
            
            greeks = self.calculate_greeks(
                new_stock_price,
                position['strike'],
                position['time_to_expiry'],
                position['risk_free_rate'],
                position['volatility'],
                position['option_type']
            )
            
            position_delta = greeks['delta'] * position['quantity'] * position['multiplier']
            current_delta += position_delta
        
        # 加上股票的Delta
        current_delta += current_positions['stock_quantity']
        
        # 检查是否需要重新平衡
        if abs(current_delta) > self.rebalance_threshold:
            adjustment = -current_delta
            
            return {
                'rebalance_needed': True,
                'stock_adjustment': adjustment,
                'new_stock_position': current_positions['stock_quantity'] + adjustment
            }
        
        return {'rebalance_needed': False}
    
    def gamma_scalping_strategy(self, option_positions, stock_price_path, transaction_cost=0.001):
        """Gamma Scalping策略"""
        pnl_series = []
        positions = option_positions.copy()
        stock_position = 0
        
        for i, stock_price in enumerate(stock_price_path):
            daily_pnl = 0
            
            # 计算当前组合Greeks
            total_gamma = 0
            total_delta = 0
            
            for position in positions:
                greeks = self.calculate_greeks(
                    stock_price,
                    position['strike'],
                    position['time_to_expiry'],
                    position['risk_free_rate'],
                    position['volatility'],
                    position['option_type']
                )
                
                total_delta += greeks['delta'] * position['quantity']
                total_gamma += greeks['gamma'] * position['quantity']
            
            # Gamma Scalping交易
            if i > 0:
                stock_move = stock_price - stock_price_path[i-1]
                
                # 基于Gamma的理论交易量
                theoretical_trade = -0.5 * total_gamma * stock_move
                
                # 实际交易（考虑交易成本）
                if abs(theoretical_trade) > 0.01:  # 最小交易门槛
                    actual_trade = theoretical_trade
                    stock_position += actual_trade
                    
                    # 计算交易成本
                    trade_cost = abs(actual_trade) * stock_price * transaction_cost
                    daily_pnl -= trade_cost
                    
                    # Gamma Scalping收益
                    scalping_pnl = 0.5 * total_gamma * stock_move**2
                    daily_pnl += scalping_pnl
            
            # 时间衰减
            theta_decay = sum([self.calculate_greeks(
                stock_price, pos['strike'], pos['time_to_expiry'],
                pos['risk_free_rate'], pos['volatility'], pos['option_type']
            )['theta'] * pos['quantity'] for pos in positions])
            
            daily_pnl += theta_decay
            
            pnl_series.append(daily_pnl)
            
            # 更新到期时间
            for position in positions:
                position['time_to_expiry'] = max(position['time_to_expiry'] - 1/365, 0.001)
        
        return pd.Series(pnl_series, index=range(len(stock_price_path)))

class VolatilityTradingStrategy:
    def __init__(self):
        self.positions = {}
    
    def long_volatility_strategy(self, stock_price, strike_prices, time_to_expiry, 
                               risk_free_rate, implied_vol, realized_vol):
        """做多波动率策略"""
        strategies = {}
        
        # Long Straddle
        atm_strike = min(strike_prices, key=lambda x: abs(x - stock_price))
        
        call_greeks = self.calculate_option_greeks(
            stock_price, atm_strike, time_to_expiry, risk_free_rate, implied_vol, 'call'
        )
        put_greeks = self.calculate_option_greeks(
            stock_price, atm_strike, time_to_expiry, risk_free_rate, implied_vol, 'put'
        )
        
        straddle_vega = call_greeks['vega'] + put_greeks['vega']
        straddle_gamma = call_greeks['gamma'] + put_greeks['gamma']
        
        strategies['long_straddle'] = {
            'positions': [
                {'type': 'call', 'strike': atm_strike, 'quantity': 1},
                {'type': 'put', 'strike': atm_strike, 'quantity': 1}
            ],
            'total_vega': straddle_vega,
            'total_gamma': straddle_gamma,
            'breakeven_up': atm_strike + (call_greeks['price'] + put_greeks['price']),
            'breakeven_down': atm_strike - (call_greeks['price'] + put_greeks['price'])
        }
        
        # Long Strangle
        otm_call_strike = min([s for s in strike_prices if s > stock_price], default=strike_prices[-1])
        otm_put_strike = max([s for s in strike_prices if s < stock_price], default=strike_prices[0])
        
        strategies['long_strangle'] = {
            'positions': [
                {'type': 'call', 'strike': otm_call_strike, 'quantity': 1},
                {'type': 'put', 'strike': otm_put_strike, 'quantity': 1}
            ]
        }
        
        return strategies
    
    def calendar_spread_strategy(self, stock_price, strike_price, short_expiry, long_expiry,
                               risk_free_rate, short_vol, long_vol):
        """日历价差策略"""
        # 卖出短期期权，买入长期期权
        short_option = self.calculate_option_greeks(
            stock_price, strike_price, short_expiry, risk_free_rate, short_vol, 'call'
        )
        long_option = self.calculate_option_greeks(
            stock_price, strike_price, long_expiry, risk_free_rate, long_vol, 'call'
        )
        
        net_premium = long_option['price'] - short_option['price']
        net_theta = long_option['theta'] - short_option['theta']
        net_vega = long_option['vega'] - short_option['vega']
        
        return {
            'net_premium': net_premium,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'max_profit_price': strike_price,  # 最大收益在行权价处
            'time_decay_advantage': -net_theta > 0  # 时间衰减是否有利
        }
    
    def calculate_option_greeks(self, S, K, T, r, sigma, option_type):
        """计算期权价格和希腊字母"""
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
        
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                r*K*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2))
        vega = S*norm.pdf(d1)*np.sqrt(T)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta/365,  # 日Theta
            'vega': vega/100     # 1%波动率变化的Vega
        }

## 九、宏观策略体系

### 9.1 利率策略

class InterestRateStrategy:
    def __init__(self):
        self.yield_curve_data = {}
    
    def calculate_yield_curve_indicators(self, yield_data):
        """计算收益率曲线指标"""
        indicators = pd.DataFrame(index=yield_data.index)
        
        # 收益率曲线斜率
        indicators['slope_2y10y'] = yield_data['10Y'] - yield_data['2Y']
        indicators['slope_3m10y'] = yield_data['10Y'] - yield_data['3M']
        indicators['slope_5y30y'] = yield_data['30Y'] - yield_data['5Y']
        
        # 收益率曲线曲率
        indicators['curvature'] = (yield_data['2Y'] + yield_data['10Y']) - 2 * yield_data['5Y']
        
        # 收益率变化率
        for tenor in ['3M', '2Y', '5Y', '10Y', '30Y']:
            if tenor in yield_data.columns:
                indicators[f'{tenor}_change'] = yield_data[tenor].diff()
                indicators[f'{tenor}_roc'] = yield_data[tenor].pct_change()
        
        # Level (水平)
        indicators['level'] = yield_data[['2Y', '5Y', '10Y']].mean(axis=1)
        
        return indicators
    
    def yield_curve_trading_signals(self, yield_indicators):
        """基于收益率曲线的交易信号"""
        signals = pd.DataFrame(index=yield_indicators.index)
        
        # 斜率交易信号
        slope_mean = yield_indicators['slope_2y10y'].rolling(252).mean()
        slope_std = yield_indicators['slope_2y10y'].rolling(252).std()
        
        # 斜率陡化/平坦化信号
        signals['steepening'] = (yield_indicators['slope_2y10y'] > slope_mean + slope_std)
        signals['flattening'] = (yield_indicators['slope_2y10y'] < slope_mean - slope_std)
        
        # 收益率水平信号
        level_percentile = yield_indicators['level'].rolling(252).rank(pct=True)
        signals['rates_low'] = level_percentile < 0.2
        signals['rates_high'] = level_percentile > 0.8
        
        # 曲率交易信号
        curvature_zscore = (yield_indicators['curvature'] - yield_indicators['curvature'].rolling(252).mean()) / yield_indicators['curvature'].rolling(252).std()
        signals['butterfly_long'] = curvature_zscore < -2  # 收益率曲线凸向上
        signals['butterfly_short'] = curvature_zscore > 2  # 收益率曲线凹向下
        
        return signals
    
    def duration_hedged_strategy(self, bond_prices, duration_data):
        """久期对冲策略"""
        # 计算修正久期
        modified_duration = duration_data['modified_duration']
        
        # 构建久期中性组合
        portfolio_weights = pd.DataFrame(index=bond_prices.index)
        
        # 假设我们有短期债券(2Y)和长期债券(10Y)
        short_duration = duration_data['2Y_duration']
        long_duration = duration_data['10Y_duration']
        target_duration = 5  # 目标久期
        
        # 计算权重使组合久期等于目标久期
        w_long = (target_duration - short_duration) / (long_duration - short_duration)
        w_short = 1 - w_long
        
        portfolio_weights['long_bond'] = w_long
        portfolio_weights['short_bond'] = w_short
        
        return portfolio_weights

### 9.2 汇率策略

class CurrencyStrategy:
    def __init__(self):
        self.currency_pairs = {}
    
    def carry_trade_strategy(self, fx_data, interest_rate_data):
        """利差交易策略"""
        signals = pd.DataFrame(index=fx_data.index)
        
        # 计算利差
        for pair in fx_data.columns:
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            if base_currency in interest_rate_data.columns and quote_currency in interest_rate_data.columns:
                interest_diff = interest_rate_data[base_currency] - interest_rate_data[quote_currency]
                
                # 利差交易信号
                signals[f'{pair}_carry'] = np.where(interest_diff > 0.01, 1, 
                                                  np.where(interest_diff < -0.01, -1, 0))
                
                # 考虑汇率动量
                fx_momentum = fx_data[pair].pct_change(20)  # 20日动量
                
                # 组合信号：利差+动量
                signals[f'{pair}_combined'] = np.where(
                    (signals[f'{pair}_carry'] == 1) & (fx_momentum > 0), 1,
                    np.where((signals[f'{pair}_carry'] == -1) & (fx_momentum < 0), -1, 0)
                )
        
        return signals
    
    def purchasing_power_parity_strategy(self, fx_data, inflation_data):
        """购买力平价策略"""
        signals = pd.DataFrame(index=fx_data.index)
        
        for pair in fx_data.columns:
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            if base_currency in inflation_data.columns and quote_currency in inflation_data.columns:
                # 计算相对通胀率
                inflation_diff = inflation_data[base_currency] - inflation_data[quote_currency]
                
                # 计算理论汇率变化
                theoretical_fx_change = inflation_diff.cumsum()
                
                # 计算实际汇率变化
                actual_fx_change = np.log(fx_data[pair] / fx_data[pair].iloc[0])
                
                # PPP偏离度
                ppp_deviation = actual_fx_change - theoretical_fx_change
                
                # 均值回归信号
                ppp_zscore = (ppp_deviation - ppp_deviation.rolling(252).mean()) / ppp_deviation.rolling(252).std()
                
                signals[f'{pair}_ppp_signal'] = np.where(ppp_zscore > 2, -1,  # 高估，做空
                                                        np.where(ppp_zscore < -2, 1, 0))  # 低估，做多
        
        return signals
    
    def volatility_regime_strategy(self, fx_data, vol_threshold=0.01):
        """波动率制度策略"""
        signals = pd.DataFrame(index=fx_data.index)
        
        for pair in fx_data.columns:
            # 计算滚动波动率
            returns = fx_data[pair].pct_change()
            rolling_vol = returns.rolling(20).std()
            
            # 波动率制度识别
            high_vol_regime = rolling_vol > rolling_vol.rolling(252).quantile(0.75)
            low_vol_regime = rolling_vol < rolling_vol.rolling(252).quantile(0.25)
            
            # 不同制度下的策略
            # 高波动率制度：均值回归
            mean_revert_signal = np.where(returns.rolling(5).mean() > 0, -1, 1)
            
            # 低波动率制度：动量跟随
            momentum_signal = np.where(returns.rolling(10).mean() > 0, 1, -1)
            
            # 组合信号
            signals[f'{pair}_regime'] = np.where(high_vol_regime, mean_revert_signal,
                                               np.where(low_vol_regime, momentum_signal, 0))
        
        return signals

## 十、策略组合与风险管理

### 10.1 多策略组合框架

class MultiStrategyPortfolio:
    def __init__(self, max_allocation_per_strategy=0.3):
        self.strategies = {}
        self.allocations = {}
        self.max_allocation = max_allocation_per_strategy
        
    def add_strategy(self, name, strategy_returns, sharpe_ratio, max_drawdown):
        """添加策略到组合"""
        self.strategies[name] = {
            'returns': strategy_returns,
            'sharpe': sharpe_ratio,
            'max_dd': max_drawdown,
            'volatility': strategy_returns.std() * np.sqrt(252)
        }
    
    def optimize_allocations(self, method='equal_risk_contribution'):
        """优化策略配置"""
        strategy_returns = pd.DataFrame({name: data['returns'] 
                                       for name, data in self.strategies.items()})
        
        if method == 'equal_weight':
            n_strategies = len(self.strategies)
            allocations = {name: 1/n_strategies for name in self.strategies.keys()}
            
        elif method == 'risk_parity':
            allocations = self.risk_parity_optimization(strategy_returns)
            
        elif method == 'mean_variance':
            allocations = self.mean_variance_optimization(strategy_returns)
            
        elif method == 'equal_risk_contribution':
            allocations = self.equal_risk_contribution(strategy_returns)
        
        # 应用最大配置约束
        total_allocation = sum(allocations.values())
        for name in allocations:
            allocations[name] = min(allocations[name], self.max_allocation)
        
        # 重新标准化
        total_allocation = sum(allocations.values())
        allocations = {name: weight/total_allocation for name, weight in allocations.items()}
        
        self.allocations = allocations
        return allocations
    
    def risk_parity_optimization(self, returns):
        """风险平价优化"""
        cov_matrix = returns.cov() * 252  # 年化协方差矩阵
        
        def risk_budget_objective(weights, cov_matrix):
            """风险预算目标函数"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # 目标：所有策略的风险贡献相等
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
    
    def calculate_portfolio_metrics(self):
        """计算组合指标"""
        if not self.allocations:
            return None
        
        # 计算组合收益
        portfolio_returns = sum(self.allocations[name] * self.strategies[name]['returns'] 
                              for name in self.allocations)
        
        # 计算指标
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        
        # 最大回撤
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

### 10.2 动态风险管理

class DynamicRiskManager:
    def __init__(self, target_volatility=0.15, lookback_window=60):
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window
        
    def calculate_portfolio_volatility(self, returns):
        """计算投资组合波动率"""
        return returns.rolling(self.lookback_window).std() * np.sqrt(252)
    
    def volatility_targeting(self, strategy_returns):
        """波动率目标调整"""
        realized_vol = self.calculate_portfolio_volatility(strategy_returns)
        
        # 计算杠杆调整因子
        leverage_factor = self.target_volatility / realized_vol
        leverage_factor = leverage_factor.fillna(1).clip(0.5, 2.0)  # 限制杠杆范围
        
        # 调整后的收益
        adjusted_returns = strategy_returns * leverage_factor.shift(1)
        
        return adjusted_returns, leverage_factor
    
    def regime_based_risk_adjustment(self, returns, market_indicators):
        """基于市场制度的风险调整"""
        risk_multipliers = pd.Series(1.0, index=returns.index)
        
        # VIX制度
        if 'vix' in market_indicators.columns:
            vix_high = market_indicators['vix'] > market_indicators['vix'].rolling(252).quantile(0.8)
            risk_multipliers[vix_high] *= 0.5  # 高VIX时降低风险
        
        # 趋势制度
        if 'market_trend' in market_indicators.columns:
            downtrend = market_indicators['market_trend'] < 0
            risk_multipliers[downtrend] *= 0.7  # 下跌趋势时降低风险
        
        # 流动性制度
        if 'liquidity_stress' in market_indicators.columns:
            liquidity_stress = market_indicators['liquidity_stress'] > market_indicators['liquidity_stress'].rolling(252).quantile(0.9)
            risk_multipliers[liquidity_stress] *= 0.3  # 流动性紧张时大幅降低风险
        
        adjusted_returns = returns * risk_multipliers.shift(1)
        return adjusted_returns, risk_multipliers
    
    def drawdown_control(self, returns, max_drawdown_threshold=0.1):
        """回撤控制"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        # 当回撤超过阈值时减少仓位
        risk_off = drawdown < -max_drawdown_threshold
        
        position_multiplier = pd.Series(1.0, index=returns.index)
        position_multiplier[risk_off] = 0.5  # 回撤过大时减半仓位
        
        controlled_returns = returns * position_multiplier.shift(1)
        
        return controlled_returns, position_multiplier

## 十一、业绩评估与归因

### 11.1 策略评估框架

class StrategyEvaluator:
    def __init__(self):
        self.benchmark_return = 0.03  # 无风险收益率
        
    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """计算全面的业绩指标"""
        metrics = {}
        
        # 基础指标
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = returns.mean() * 252
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (metrics['annual_return'] - self.benchmark_return) / metrics['annual_volatility']
        
        # 偏度和峰度
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # 回撤分析
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        
        # 回撤持续时间
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
        
        metrics['avg_drawdown_duration'] = np.mean(drawdown_periods) if drawdown_periods else 0
        metrics['max_drawdown_duration'] = max(drawdown_periods) if drawdown_periods else 0
        
        # 胜率统计
        metrics['win_rate'] = (returns > 0).mean()
        metrics['profit_factor'] = returns[returns > 0].sum() / abs(returns[returns < 0].sum())
        
        # 相对基准指标
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            metrics['alpha'] = excess_returns.mean() * 252
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = metrics['alpha'] / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
            
            # Beta计算
            covariance = np.cov(returns.dropna(), benchmark_returns.dropna())[0, 1]
            benchmark_variance = benchmark_returns.var()
            metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        return metrics
    
    def rolling_performance_analysis(self, returns, window=252):
        """滚动业绩分析"""
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics['rolling_sharpe'] = (rolling_metrics['rolling_return'] * 252 - self.benchmark_return) / rolling_metrics['rolling_volatility']
        
        # 滚动最大回撤
        for i in range(window, len(returns)):
            period_returns = returns.iloc[i-window:i]
            cumulative = (1 + period_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            rolling_metrics.iloc[i]['rolling_max_drawdown'] = drawdown.min()
        
        return rolling_metrics
    