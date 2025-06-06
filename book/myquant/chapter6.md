
## 六、基本面量化策略体系

### 6.1 多因子模型策略

#### 6.1.1 Fama-French五因子模型

**策略原理**：
基于Fama-French五因子模型，通过市场因子(MKT)、规模因子(SMB)、价值因子(HML)、盈利能力因子(RMW)和投资因子(CMA)来解释股票收益率。

```python
class FamaFrenchStrategy:
    def __init__(self):
        self.factor_loadings = {}
        self.factor_returns = {}
    
    def calculate_market_factor(self, returns, market_returns):
        """计算市场因子(MKT)"""
        risk_free_rate = 0.03 / 252  # 假设年化无风险利率3%
        market_excess = market_returns - risk_free_rate
        return market_excess
    
    def calculate_size_factor(self, stock_data):
        """计算规模因子(SMB) - Small Minus Big"""
        # 按市值分组
        stock_data['market_cap_rank'] = stock_data['market_cap'].rank(pct=True)
        
        small_cap = stock_data[stock_data['market_cap_rank'] <= 0.5]
        big_cap = stock_data[stock_data['market_cap_rank'] > 0.5]
        
        smb = small_cap['returns'].mean() - big_cap['returns'].mean()
        return smb
    
    def calculate_value_factor(self, stock_data):
        """计算价值因子(HML) - High Minus Low"""
        # 按账面市值比分组
        stock_data['bm_rank'] = stock_data['book_to_market'].rank(pct=True)
        
        high_bm = stock_data[stock_data['bm_rank'] >= 0.7]
        low_bm = stock_data[stock_data['bm_rank'] <= 0.3]
        
        hml = high_bm['returns'].mean() - low_bm['returns'].mean()
        return hml
    
    def calculate_profitability_factor(self, stock_data):
        """计算盈利能力因子(RMW) - Robust Minus Weak"""
        # 按ROE分组
        stock_data['roe_rank'] = stock_data['roe'].rank(pct=True)
        
        robust_prof = stock_data[stock_data['roe_rank'] >= 0.7]
        weak_prof = stock_data[stock_data['roe_rank'] <= 0.3]
        
        rmw = robust_prof['returns'].mean() - weak_prof['returns'].mean()
        return rmw
    
    def calculate_investment_factor(self, stock_data):
        """计算投资因子(CMA) - Conservative Minus Aggressive"""
        # 按资产增长率分组
        stock_data['asset_growth_rank'] = stock_data['asset_growth'].rank(pct=True)
        
        conservative = stock_data[stock_data['asset_growth_rank'] <= 0.3]
        aggressive = stock_data[stock_data['asset_growth_rank'] >= 0.7]
        
        cma = conservative['returns'].mean() - aggressive['returns'].mean()
        return cma
    
    def build_factor_model(self, stock_data, market_data):
        """构建五因子模型"""
        import statsmodels.api as sm
        
        # 计算各因子
        mkt = self.calculate_market_factor(stock_data['returns'], market_data['returns'])
        smb = self.calculate_size_factor(stock_data)
        hml = self.calculate_value_factor(stock_data)
        rmw = self.calculate_profitability_factor(stock_data)
        cma = self.calculate_investment_factor(stock_data)
        
        # 对每只股票回归
        results = {}
        
        for stock in stock_data['stock_id'].unique():
            stock_returns = stock_data[stock_data['stock_id'] == stock]['returns']
            
            # 构建回归模型
            X = pd.DataFrame({
                'MKT': mkt,
                'SMB': smb,
                'HML': hml,
                'RMW': rmw,
                'CMA': cma
            })
            X = sm.add_constant(X)
            
            model = sm.OLS(stock_returns, X).fit()
            results[stock] = {
                'alpha': model.params[0],
                'beta_mkt': model.params[1],
                'beta_smb': model.params[2],
                'beta_hml': model.params[3],
                'beta_rmw': model.params[4],
                'beta_cma': model.params[5],
                'r_squared': model.rsquared
            }
        
        return results
    
    def generate_signals(self, stock_data, factor_results):
        """基于因子载荷生成信号"""
        signals = pd.DataFrame()
        
        for stock, factors in factor_results.items():
            # 基于alpha选股
            if factors['alpha'] > 0 and factors['r_squared'] > 0.3:
                signals.loc[stock, 'alpha_signal'] = 1
            else:
                signals.loc[stock, 'alpha_signal'] = 0
            
            # 基于因子载荷选股
            score = 0
            
            # 偏好小市值股票
            if factors['beta_smb'] > 0:
                score += 1
            
            # 偏好价值股
            if factors['beta_hml'] > 0:
                score += 1
            
            # 偏好高盈利能力股票
            if factors['beta_rmw'] > 0:
                score += 1
            
            # 偏好保守投资的公司
            if factors['beta_cma'] > 0:
                score += 1
            
            signals.loc[stock, 'factor_score'] = score
            signals.loc[stock, 'final_signal'] = 1 if score >= 3 else 0
        
        return signals
```

#### 6.1.2 自定义多因子模型

```python
class CustomMultiFactorStrategy:
    def __init__(self):
        self.factor_calculator = FactorCalculator()
        self.factor_weights = {}
    
    def calculate_fundamental_factors(self, financial_data):
        """计算基本面因子"""
        factors = pd.DataFrame(index=financial_data.index)
        
        # 估值因子
        factors['PE'] = financial_data['market_cap'] / financial_data['net_income']
        factors['PB'] = financial_data['market_cap'] / financial_data['book_value']
        factors['PS'] = financial_data['market_cap'] / financial_data['revenue']
        factors['PCF'] = financial_data['market_cap'] / financial_data['operating_cashflow']
        factors['EV_EBITDA'] = financial_data['enterprise_value'] / financial_data['ebitda']
        
        # 成长因子
        factors['revenue_growth'] = financial_data['revenue'].pct_change(periods=4)  # 年同比
        factors['earnings_growth'] = financial_data['net_income'].pct_change(periods=4)
        factors['book_value_growth'] = financial_data['book_value'].pct_change(periods=4)
        factors['roa_change'] = financial_data['roa'].diff(periods=4)
        factors['roe_change'] = financial_data['roe'].diff(periods=4)
        
        # 质量因子
        factors['roa'] = financial_data['net_income'] / financial_data['total_assets']
        factors['roe'] = financial_data['net_income'] / financial_data['shareholders_equity']
        factors['gross_margin'] = financial_data['gross_profit'] / financial_data['revenue']
        factors['operating_margin'] = financial_data['operating_income'] / financial_data['revenue']
        factors['debt_to_equity'] = financial_data['total_debt'] / financial_data['shareholders_equity']
        factors['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']
        factors['asset_turnover'] = financial_data['revenue'] / financial_data['total_assets']
        
        # 现金流因子
        factors['fcf_yield'] = financial_data['free_cashflow'] / financial_data['market_cap']
        factors['ocf_to_sales'] = financial_data['operating_cashflow'] / financial_data['revenue']
        factors['capex_to_sales'] = financial_data['capex'] / financial_data['revenue']
        
        return factors
    
    def calculate_technical_factors(self, price_data):
        """计算技术面因子"""
        factors = pd.DataFrame(index=price_data.index)
        
        # 动量因子
        for period in [1, 3, 6, 12]:
            factors[f'momentum_{period}m'] = price_data['close'].pct_change(periods=period*21)
        
        # 反转因子
        factors['reversal_1m'] = -price_data['close'].pct_change(periods=21)
        
        # 波动率因子
        for period in [1, 3, 6]:
            returns = price_data['close'].pct_change()
            factors[f'volatility_{period}m'] = returns.rolling(period*21).std()
        
        # 流动性因子
        factors['turnover'] = price_data['volume'] / price_data['shares_outstanding']
        factors['amihud_illiq'] = abs(price_data['close'].pct_change()) / (price_data['volume'] * price_data['close'])
        
        # 技术指标因子
        factors['rsi'] = self.calculate_rsi(price_data['close'])
        factors['macd_signal'] = self.calculate_macd_signal(price_data['close'])
        
        return factors
    
    def factor_neutralization(self, factors, industry_codes, market_cap):
        """因子中性化处理"""
        neutralized_factors = factors.copy()
        
        for factor in factors.columns:
            if factors[factor].isna().all():
                continue
                
            # 行业中性化
            for industry in industry_codes.unique():
                industry_mask = industry_codes == industry
                if industry_mask.sum() > 1:
                    industry_mean = factors.loc[industry_mask, factor].mean()
                    neutralized_factors.loc[industry_mask, factor] -= industry_mean
            
            # 市值中性化
            market_cap_log = np.log(market_cap)
            correlation = neutralized_factors[factor].corr(market_cap_log)
            if not np.isnan(correlation):
                beta = correlation * neutralized_factors[factor].std() / market_cap_log.std()
                neutralized_factors[factor] -= beta * (market_cap_log - market_cap_log.mean())
        
        return neutralized_factors
    
    def factor_scoring(self, factors):
        """因子打分"""
        scores = pd.DataFrame(index=factors.index)
        
        # 估值因子得分（低估值高分）
        valuation_factors = ['PE', 'PB', 'PS', 'PCF', 'EV_EBITDA']
        valuation_score = 0
        for factor in valuation_factors:
            if factor in factors.columns:
                score = -factors[factor].rank(pct=True) + 1  # 反向排序
                valuation_score += score
        scores['valuation_score'] = valuation_score / len(valuation_factors)
        
        # 成长因子得分（高成长高分）
        growth_factors = ['revenue_growth', 'earnings_growth', 'book_value_growth', 'roa_change', 'roe_change']
        growth_score = 0
        for factor in growth_factors:
            if factor in factors.columns:
                score = factors[factor].rank(pct=True)
                growth_score += score
        scores['growth_score'] = growth_score / len(growth_factors)
        
        # 质量因子得分
        quality_factors = ['roa', 'roe', 'gross_margin', 'operating_margin', 'current_ratio', 'asset_turnover']
        quality_score = 0
        for factor in quality_factors:
            if factor in factors.columns:
                score = factors[factor].rank(pct=True)
                quality_score += score
        
        # 负向质量因子
        negative_quality_factors = ['debt_to_equity']
        for factor in negative_quality_factors:
            if factor in factors.columns:
                score = -factors[factor].rank(pct=True) + 1
                quality_score += score
        
        scores['quality_score'] = quality_score / (len(quality_factors) + len(negative_quality_factors))
        
        # 综合得分
        scores['composite_score'] = (scores['valuation_score'] * 0.4 + 
                                   scores['growth_score'] * 0.3 + 
                                   scores['quality_score'] * 0.3)
        
        return scores
    
    def generate_portfolio_weights(self, scores, max_weight=0.05):
        """生成投资组合权重"""
        # 选择得分前30%的股票
        threshold = scores['composite_score'].quantile(0.7)
        selected_stocks = scores[scores['composite_score'] >= threshold]
        
        # 基于得分分配权重
        raw_weights = selected_stocks['composite_score'] / selected_stocks['composite_score'].sum()
        
        # 限制单只股票最大权重
        weights = np.minimum(raw_weights, max_weight)
        weights = weights / weights.sum()  # 重新标准化
        
        return weights
    
    def calculate_rsi(self, prices, window=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd_signal(self, prices):
        """计算MACD信号"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd > signal).astype(int)
```
