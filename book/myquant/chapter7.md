
## 七、波动率策略体系

### 7.1 GARCH模型波动率预测策略

```python
from arch import arch_model
import scipy.optimize as optimize

class GARCHVolatilityStrategy:
    def __init__(self, model_type='GARCH', p=1, q=1):
        self.model_type = model_type
        self.p = p
        self.q = q
        self.model = None
        
    def fit_garch_model(self, returns):
        """拟合GARCH模型"""
        # 移除缺失值和异常值
        returns_clean = returns.dropna()
        returns_clean = returns_clean[np.abs(returns_clean) < returns_clean.std() * 5]
        
        # 拟合GARCH模型
        model = arch_model(returns_clean * 100, vol='Garch', p=self.p, q=self.q)
        self.model = model.fit(disp='off')
        
        return self.model
    
    def forecast_volatility(self, horizon=1):
        """预测未来波动率"""
        forecast = self.model.forecast(horizon=horizon)
        predicted_variance = forecast.variance.iloc[-1, :] / 10000  # 转换回原始尺度
        predicted_volatility = np.sqrt(predicted_variance)
        
        return predicted_volatility
    
    def calculate_volatility_signals(self, returns, lookback=252):
        """基于波动率预测生成交易信号"""
        signals = pd.DataFrame(index=returns.index)
        volatility_forecasts = []
        realized_volatilities = []
        
        for i in range(lookback, len(returns)):
            # 滚动窗口拟合GARCH模型
            window_returns = returns.iloc[i-lookback:i]
            
            try:
                self.fit_garch_model(window_returns)
                vol_forecast = self.forecast_volatility(horizon=1)[0]
                volatility_forecasts.append(vol_forecast)
                
                # 计算已实现波动率
                realized_vol = window_returns.rolling(20).std().iloc[-1]
                realized_volatilities.append(realized_vol)
                
            except:
                volatility_forecasts.append(np.nan)
                realized_volatilities.append(np.nan)
        
        # 添加NaN值以对齐索引
        volatility_forecasts = [np.nan] * lookback + volatility_forecasts
        realized_volatilities = [np.nan] * lookback + realized_volatilities
        
        signals['vol_forecast'] = volatility_forecasts
        signals['vol_realized'] = realized_volatilities
        signals['vol_ratio'] = signals['vol_forecast'] / signals['vol_realized']
        
        # 生成交易信号
        signals['vol_breakout'] = (signals['vol_forecast'] > signals['vol_forecast'].rolling(60).quantile(0.8))
        signals['vol_mean_revert'] = (signals['vol_forecast'] < signals['vol_forecast'].rolling(60).quantile(0.2))
        
        return signals

class VolatilitySurfaceStrategy:
    def __init__(self):
        self.vol_surface = {}
    
    def calculate_implied_volatility(self, option_prices, stock_price, strike_prices, 
                                   time_to_expiry, risk_free_rate, option_type='call'):
        """计算隐含波动率"""
        from scipy.optimize import brentq
        
        def black_scholes_price(vol):
            """Black-Scholes期权定价公式"""
            from scipy.stats import norm
            
            d1 = (np.log(stock_price / strike_prices) + 
                  (risk_free_rate + 0.5 * vol**2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
            d2 = d1 - vol * np.sqrt(time_to_expiry)
            
            if option_type == 'call':
                price = (stock_price * norm.cdf(d1) - 
                        strike_prices * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:
                price = (strike_prices * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                        stock_price * norm.cdf(-d1))
            
            return price
        
        implied_vols = []
        
        for i, market_price in enumerate(option_prices):
            try:
                strike = strike_prices[i]
                
                def objective(vol):
                    return black_scholes_price(vol) - market_price
                
                iv = brentq(objective, 0.01, 5.0)
                implied_vols.append(iv)
            except:
                implied_vols.append(np.nan)
        
        return np.array(implied_vols)
    
    def build_volatility_surface(self, option_data):
        """构建波动率曲面"""
        vol_surface = {}
        
        for expiry in option_data['expiry'].unique():
            expiry_data = option_data[option_data['expiry'] == expiry]
            
            strikes = expiry_data['strike'].values
            call_prices = expiry_data['call_price'].values
            put_prices = expiry_data['put_price'].values
            
            stock_price = expiry_data['underlying_price'].iloc[0]
            time_to_expiry = expiry_data['time_to_expiry'].iloc[0]
            risk_free_rate = expiry_data['risk_free_rate'].iloc[0]
            
            # 计算看涨和看跌期权隐含波动率
            call_ivs = self.calculate_implied_volatility(
                call_prices, stock_price, strikes, time_to_expiry, risk_free_rate, 'call'
            )
            put_ivs = self.calculate_implied_volatility(
                put_prices, stock_price, strikes, time_to_expiry, risk_free_rate, 'put'
            )
            
            vol_surface[expiry] = {
                'strikes': strikes,
                'call_ivs': call_ivs,
                'put_ivs': put_ivs,
                'atm_iv': np.interp(stock_price, strikes, call_ivs)
            }
        
        return vol_surface
    
    def detect_volatility_arbitrage(self, vol_surface):
        """检测波动率套利机会"""
        arbitrage_signals = []
        
        for expiry, surface_data in vol_surface.items():
            strikes = surface_data['strikes']
            call_ivs = surface_data['call_ivs']
            put_ivs = surface_data['put_ivs']
            
            # Put-Call Parity检查
            iv_spread = call_ivs - put_ivs
            abnormal_spread = np.abs(iv_spread) > 0.02  # 2%阈值
            
            # 波动率偏斜异常
            atm_strike_idx = np.argmin(np.abs(strikes - surface_data.get('atm_strike', strikes[len(strikes)//2])))
            
            if atm_strike_idx > 0 and atm_strike_idx < len(call_ivs) - 1:
                left_skew = call_ivs[atm_strike_idx] - call_ivs[atm_strike_idx - 1]
                right_skew = call_ivs[atm_strike_idx + 1] - call_ivs[atm_strike_idx]
                
                # 异常偏斜
                abnormal_skew = abs(left_skew - right_skew) > 0.05
                
                arbitrage_signals.append({
                    'expiry': expiry,
                    'abnormal_spread': abnormal_spread.any(),
                    'abnormal_skew': abnormal_skew,
                    'max_spread': np.max(np.abs(iv_spread))
                })
        
        return arbitrage_signals
```
