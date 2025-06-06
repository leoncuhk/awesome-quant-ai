## 三、套利策略体系

本章将深入探讨广义上的套利策略体系。与第二章中重点介绍的、主要依赖历史数据和均值回归原理的统计套利策略不同，本章所涵盖的套利方法范围更广。我们将讨论基于确定性金融理论的无风险套利（如期现套利、转换套利），这类策略在理想条件下风险较低；同时也会介绍涉及跨资产、跨市场操作的套利策略（如可转债套利），这些策略可能依赖更复杂的定价模型和风险管理技术。本章旨在全面展现套利交易的多样性及其在不同市场环境下的应用。

### 3.1 无风险套利策略

#### 3.1.1 期现套利策略

**策略原理**：
利用期货价格与现货价格之间的不合理价差进行套利，基于期货定价理论。当期货的市场价格偏离其理论价格（由现货价格、无风险利率、持有成本、到期时间等因素决定）到一定程度，足以覆盖交易成本并产生利润时，套利机会出现。

**核心公式**：
期货理论价格 \( F_0 = S_0 \cdot e^{(r-q)T} \)
其中：
- \( F_0 \) = 期货理论价格
- \( S_0 \) = 当前现货价格
- \( r \) = 无风险利率 (连续复利)
- \( q \) = 连续股息率或持有期收益率
- \( T \) = 到期时间 (年)

**套利机制**：
1.  **正向套利 (Cash-Carry Arbitrage)**：当 \( F_{market} > F_0 + \text{Costs} \)
    - 操作：借款买入现货，同时卖出期货合约。
    - 结果：锁定无风险利润，利润来源于期货高估部分。
2.  **反向套利 (Reverse Cash-Carry Arbitrage)**：当 \( F_{market} < F_0 - \text{Costs} \)
    - 操作：融券卖出现货（或卖出已持有的现货），同时买入期货合约。
    - 结果：锁定无风险利润，利润来源于期货低估部分。

**实现细节**：
- **成本考量**：交易佣金、冲击成本、资金成本、融券成本等。
- **交割风险**：实物交割的期货需要考虑交割的便利性和成本。
- **流动性风险**：确保现货和期货市场都有足够的流动性执行套利。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class FuturesCashArbitrageStrategy:
    """期现套利策略"""
    
    def __init__(self, transaction_cost_pct: float = 0.001, interest_rate: float = 0.03, 
                 dividend_yield: float = 0.01):
        """
        初始化期现套利策略
        
        Args:
            transaction_cost_pct: 单边交易成本百分比 (例如 0.001 表示 0.1%)
            interest_rate: 无风险利率 (年化)
            dividend_yield: 标的资产年化股息率
        """
        self.transaction_cost_pct = transaction_cost_pct
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield
    
    def calculate_theoretical_futures_price(self, spot_price: float, time_to_maturity_years: float) -> float:
        """
        计算期货理论价格 (连续复利)
        
        Args:
            spot_price: 现货价格
            time_to_maturity_years: 期货到期时间 (年)
            
        Returns:
            期货理论价格
        """
        theoretical_price = spot_price * np.exp(
            (self.interest_rate - self.dividend_yield) * time_to_maturity_years
        )
        return theoretical_price
    
    def identify_arbitrage_opportunities(self, futures_price: float, spot_price: float, 
                                         time_to_maturity_years: float) -> Dict:
        """
        识别期现套利机会
        
        Args:
            futures_price: 期货市场价格
            spot_price: 现货市场价格
            time_to_maturity_years: 期货到期时间 (年)
            
        Returns:
            包含套利机会信息的字典
        """
        theoretical_price = self.calculate_theoretical_futures_price(spot_price, time_to_maturity_years)
        
        cost_adj_spot_buy = spot_price * (1 + self.transaction_cost_pct)
        cost_adj_futures_sell = futures_price * (1 - self.transaction_cost_pct)
        
        cost_adj_spot_sell = spot_price * (1 - self.transaction_cost_pct)
        cost_adj_futures_buy = futures_price * (1 + self.transaction_cost_pct)

        opportunity = {
            'futures_price': futures_price,
            'spot_price': spot_price,
            'theoretical_futures_price': theoretical_price,
            'time_to_maturity_years': time_to_maturity_years,
            'arbitrage_type': None, 
            'profit_margin_pct': 0.0,
            'description': "No arbitrage opportunity"
        }

        # 正向套利: 买入现货，卖出期货 (当期货价格相对理论价格过高)
        # 实际期货卖价 - 现货买入成本及持有成本 > 0
        profit_positive = cost_adj_futures_sell - cost_adj_spot_buy * np.exp((self.interest_rate - self.dividend_yield) * time_to_maturity_years)
        
        if profit_positive > 0:
            opportunity['arbitrage_type'] = 'positive' # 买现货，卖期货
            initial_investment_positive = spot_price * (1 + self.transaction_cost_pct)
            opportunity['profit_margin_pct'] = (profit_positive / initial_investment_positive) * 100
            opportunity['description'] = (
                f"Positive arbitrage: Buy spot at {spot_price:.2f}, sell futures at {futures_price:.2f}. "
                f"Theoretical futures price is {theoretical_price:.2f}. "
                f"Expected profit margin: {opportunity['profit_margin_pct']:.4f}%."
            )
            return opportunity

        # 反向套利: 卖出现货，买入期货 (当期货价格相对理论价格过低)
        # 现货卖出所得及持有期收益 - 期货买入成本 > 0
        profit_negative = cost_adj_spot_sell * np.exp((self.interest_rate - self.dividend_yield) * time_to_maturity_years) - cost_adj_futures_buy
        
        if profit_negative > 0:
            opportunity['arbitrage_type'] = 'negative' # 卖现货，买期货
            initial_investment_negative = futures_price * (1 + self.transaction_cost_pct) 
            opportunity['profit_margin_pct'] = (profit_negative / initial_investment_negative) * 100
            opportunity['description'] = (
                f"Negative arbitrage: Sell spot at {spot_price:.2f}, buy futures at {futures_price:.2f}. "
                f"Theoretical futures price is {theoretical_price:.2f}. "
                f"Expected profit margin: {opportunity['profit_margin_pct']:.4f}%."
            )
            return opportunity
            
        return opportunity

# 期现套利策略使用示例
def demo_futures_cash_arbitrage():
    """期现套利策略演示"""
    strategy = FuturesCashArbitrageStrategy(
        transaction_cost_pct=0.001,  # 0.1%
        interest_rate=0.03,          # 3%
        dividend_yield=0.01          # 1%
    )
    
    # 案例 1: 正向套利机会
    print("\\n--- Case 1: Positive Arbitrage ---")
    spot_price_1 = 100.0
    futures_price_1 = 103.0
    time_to_maturity_1 = 0.5  # 半年
    
    opportunity_1 = strategy.identify_arbitrage_opportunities(futures_price_1, spot_price_1, time_to_maturity_1)
    print(f"Spot Price: {spot_price_1}, Futures Price: {futures_price_1}, Maturity: {time_to_maturity_1} years")
    print(f"Theoretical Futures Price: {opportunity_1['theoretical_futures_price']:.2f}")
    print(f"Arbitrage Type: {opportunity_1['arbitrage_type']}")
    print(f"Profit Margin: {opportunity_1['profit_margin_pct']:.4f}%")
    print(f"Description: {opportunity_1['description']}")

    # 案例 2: 反向套利机会
    print("\\n--- Case 2: Negative Arbitrage ---")
    spot_price_2 = 100.0
    futures_price_2 = 98.0
    time_to_maturity_2 = 0.25  # 3个月
    
    opportunity_2 = strategy.identify_arbitrage_opportunities(futures_price_2, spot_price_2, time_to_maturity_2)
    print(f"Spot Price: {spot_price_2}, Futures Price: {futures_price_2}, Maturity: {time_to_maturity_2} years")
    print(f"Theoretical Futures Price: {opportunity_2['theoretical_futures_price']:.2f}")
    print(f"Arbitrage Type: {opportunity_2['arbitrage_type']}")
    print(f"Profit Margin: {opportunity_2['profit_margin_pct']:.4f}%")
    print(f"Description: {opportunity_2['description']}")
    
    # 案例 3: 无套利机会
    print("\\n--- Case 3: No Arbitrage ---")
    spot_price_3 = 100.0
    theoretical_3 = strategy.calculate_theoretical_futures_price(spot_price_3, 0.5)
    futures_price_3 = theoretical_3 * 1.0005 
    time_to_maturity_3 = 0.5

    opportunity_3 = strategy.identify_arbitrage_opportunities(futures_price_3, spot_price_3, time_to_maturity_3)
    print(f"Spot Price: {spot_price_3}, Futures Price: {futures_price_3:.2f}, Maturity: {time_to_maturity_3} years")
    print(f"Theoretical Futures Price: {opportunity_3['theoretical_futures_price']:.2f}")
    print(f"Arbitrage Type: {opportunity_3['arbitrage_type']}")
    print(f"Profit Margin: {opportunity_3['profit_margin_pct']:.4f}%")
    print(f"Description: {opportunity_3['description']}")

    return opportunity_1, opportunity_2, opportunity_3

if __name__ == "__main__":
    # ... (其他策略的 demo 调用)
    demo_futures_cash_arbitrage()
```

#### 3.1.2 转换套利策略

**策略原理**：
转换套利（Conversion Arbitrage）与反向转换套利（Reverse Conversion Arbitrage）是基于期权的买卖权平价理论 (Put-Call Parity) 的无风险套利策略。该理论指出，具有相同标的资产、相同行权价和相同到期日的欧式看涨期权和看跌期权的价格之间存在一个确定的关系。

**买卖权平价公式**：
\[ S_0 + P_0 = C_0 + K \cdot e^{-rT} \]
其中：
- \( S_0 \) = 标的资产当前价格
- \( P_0 \) = 看跌期权当前价格
- \( C_0 \) = 看涨期权当前价格
- \( K \) = 行权价
- \( r \) = 无风险利率 (连续复利)
- \( T \) = 期权到期时间 (年)
- \( e^{-rT} \) = 贴现因子

当市场价格偏离这一平价关系并且足以覆盖交易成本时，套利机会出现。

**1. 转换套利 (Conversion Arbitrage)**
- **套利条件**：当 \( S_0 + P_0 - C_0 < K \cdot e^{-rT} - \text{Costs} \) （即组合的实际成本低于其理论价值/未来保证收益的现值）
- **操作**（构建一个合成空头债券头寸，其成本低于直接借入资金的成本）：
    1. 买入标的股票 (支出 \( S_0 \))
    2. 买入看跌期权 (支出 \( P_0 \))
    3. 卖出看涨期权 (收入 \( C_0 \))
- **组合净成本** = \( S_0 + P_0 - C_0 \)
- **到期日组合价值**：无论股价如何变动，该组合在到期日都价值 \( K \)。
    - 若 \( S_T > K \)：看涨期权被行权，股票以 \( K \) 卖出；看跌期权作废。
    - 若 \( S_T \le K \)：看跌期权被行权，股票以 \( K \) 卖出；看涨期权作废。
- **套利利润** = \( K \cdot e^{-rT} - (S_0 + P_0 - C_0) - \text{Transaction Costs} \)

**2. 反向转换套利 (Reverse Conversion Arbitrage)**
- **套利条件**：当 \( C_0 - P_0 - S_0 > -K \cdot e^{-rT} + \text{Costs} \) （即组合的实际收入高于其未来保证支付的现值）
- **操作**（构建一个合成多头债券头寸，其收益高于直接贷出资金的收益）：
    1. 卖空标的股票 (收入 \( S_0 \))
    2. 卖出看跌期权 (收入 \( P_0 \))
    3. 买入看涨期权 (支出 \( C_0 \))
- **组合净收入** = \( S_0 + P_0 - C_0 \) (注：这里指初始现金流，通常卖空股票等会产生正现金流)
    更准确地说是：初始现金流为 \( C_0 - P_0 - S_0 \)
- **到期日组合价值**：无论股价如何变动，该组合在到期日都价值 \( -K \) (即需要支付 \( K \))。
- **套利利润** = \( (C_0 - P_0 - S_0) - (-K \cdot e^{-rT}) - \text{Transaction Costs} = C_0 - P_0 - S_0 + K \cdot e^{-rT} - \text{Transaction Costs} \)

**实现细节**：
- **交易成本**：股票和期权的佣金、买卖价差、借券费用（用于卖空）等。
- **期权类型**：该平价关系严格适用于欧式期权。美式期权由于可以提前行权，可能存在微小偏差。
- **同步执行**：套利成功依赖于所有交易腿的同步执行以锁定价格。
- **资金占用**：卖出期权（尤其是裸卖）需要保证金。

```python
import numpy as np
from typing import Dict

class ConversionArbitrageStrategy:
    """
    转换套利与反向转换套利策略
    基于欧式期权的买卖权平价关系 (Put-Call Parity): S + P = C + K * e^(-rT)
    """

    def __init__(self, stock_tx_cost_pct: float = 0.001, option_tx_cost_per_share: float = 0.001):
        """
        初始化转换套利策略
        Args:
            stock_tx_cost_pct: 股票交易成本百分比 (例如 0.001 表示 0.1%)
            option_tx_cost_per_share: 单份期权合约对应每股的平均交易成本 (例如合约手续费平摊到每股)
        """
        self.stock_tx_cost_pct = stock_tx_cost_pct 
        self.option_tx_cost_per_share = option_tx_cost_per_share

    def identify_arbitrage_opportunity(self, stock_price: float, call_price: float, put_price: float,
                                       strike_price: float, risk_free_rate: float,
                                       time_to_expiration_years: float) -> Dict:
        """
        识别转换套利或反向转换套利机会 (计算均基于单股)

        Args:
            stock_price: 标的股票当前价格
            call_price: 看涨期权市场价格 (每股)
            put_price: 看跌期权市场价格 (每股)
            strike_price: 期权行权价 (call和put相同)
            risk_free_rate: 无风险年利率
            time_to_expiration_years: 期权到期时间 (年)

        Returns:
            包含套利机会信息的字典
        """
        present_value_strike = strike_price * np.exp(-risk_free_rate * time_to_expiration_years)

        opportunity = {
            'stock_price': stock_price, 'call_price': call_price, 'put_price': put_price,
            'strike_price': strike_price, 'pv_strike': present_value_strike,
            'arbitrage_type': None,
            'profit_per_share': 0.0,
            'description': "No arbitrage opportunity based on Put-Call Parity"
        }

        # 1. 转换套利 (Conversion Arbitrage)
        # 策略: 买股票 (S), 买看跌 (P), 卖看涨 (C)
        # 组合成本 (每股): S_buy + P_buy - C_sell
        cost_stock_buy_eff = stock_price * (1 + self.stock_tx_cost_pct)
        cost_put_buy_eff = put_price + self.option_tx_cost_per_share
        proceeds_call_sell_eff = call_price - self.option_tx_cost_per_share
        
        net_cost_conversion = cost_stock_buy_eff + cost_put_buy_eff - proceeds_call_sell_eff
        # 到期价值 K
        # 套利条件: net_cost_conversion < PV(K)
        profit_conversion = present_value_strike - net_cost_conversion

        if profit_conversion > 0:
            opportunity['arbitrage_type'] = "Conversion"
            opportunity['profit_per_share'] = profit_conversion
            opportunity['description'] = (
                f"Conversion Arbitrage: Buy stock, buy put, sell call. "
                f"Net cost per share ({net_cost_conversion:.4f}) < PV of Strike ({present_value_strike:.4f}). "
                f"Profit per share: {profit_conversion:.4f}"
            )
            return opportunity

        # 2. 反向转换套利 (Reverse Conversion Arbitrage)
        # 策略: 卖空股票 (-S), 卖看跌 (-P), 买看涨 (C)
        # 组合初始现金流 (每股): S_sell + P_sell - C_buy
        proceeds_stock_sell_eff = stock_price * (1 - self.stock_tx_cost_pct)
        proceeds_put_sell_eff = put_price - self.option_tx_cost_per_share
        cost_call_buy_eff = call_price + self.option_tx_cost_per_share

        net_proceeds_reverse_conversion = proceeds_stock_sell_eff + proceeds_put_sell_eff - cost_call_buy_eff
        # 到期需支付 K, 即到期价值 -K
        # 套利条件: net_proceeds_reverse_conversion > -PV(K)  (即 net_proceeds_reverse_conversion + PV(K) > 0)
        profit_reverse_conversion = net_proceeds_reverse_conversion + present_value_strike
        
        if profit_reverse_conversion > 0:
            opportunity['arbitrage_type'] = "Reverse Conversion"
            opportunity['profit_per_share'] = profit_reverse_conversion
            opportunity['description'] = (
                f"Reverse Conversion Arbitrage: Sell stock, sell put, buy call. "
                f"Net proceeds per share ({net_proceeds_reverse_conversion:.4f}) > -PV of Strike ({-present_value_strike:.4f}). "
                f"Profit per share: {profit_reverse_conversion:.4f}"
            )
            return opportunity

        return opportunity

# 转换套利策略使用示例
def demo_conversion_arbitrage():
    """转换套利与反向转换套利策略演示"""
    # 假设期权价格和交易成本都是针对每股计算的
    strategy = ConversionArbitrageStrategy(stock_tx_cost_pct=0.0005, option_tx_cost_per_share=0.005) 
    risk_free_rate = 0.02
    time_to_expiration = 0.25 # 3 个月
    strike_price = 100.0

    print("\n--- Case 1: Conversion Arbitrage Opportunity ---")
    # 条件: S_buy_eff + P_buy_eff - C_sell_eff < PV(K)
    # PV(K) = 100 * exp(-0.02 * 0.25) = 99.5012
    # S_buy_eff = 100 * (1+0.0005) = 100.05
    # P_buy_eff = 1.50 + 0.005 = 1.505
    # C_sell_eff = 2.50 - 0.005 = 2.495
    # Net cost = 100.05 + 1.505 - 2.495 = 101.555 - 2.495 = 99.06
    # 99.06 < 99.5012. Profit = 99.5012 - 99.06 = 0.4412
    stock_price_1 = 100.0
    call_price_1 = 2.50 
    put_price_1 = 1.50  
    
    opportunity_1 = strategy.identify_arbitrage_opportunity(
        stock_price_1, call_price_1, put_price_1, strike_price,
        risk_free_rate, time_to_expiration
    )
    print(opportunity_1['description'])
    # 预期的 S+P-C = 100+1.5-2.5 = 99. PV(K) = 99.50. 99 < 99.50.


    print("\n--- Case 2: Reverse Conversion Arbitrage Opportunity ---")
    # 条件: S_sell_eff + P_sell_eff - C_buy_eff > -PV(K)
    # PV(K) = 99.5012. -PV(K) = -99.5012
    # S_sell_eff = 100 * (1-0.0005) = 99.95
    # P_sell_eff = 1.00 - 0.005 = 0.995
    # C_buy_eff = 2.00 + 0.005 = 2.005
    # Net proceeds = 99.95 + 0.995 - 2.005 = 100.945 - 2.005 = 98.94
    # 98.94 > -99.5012. Profit = 98.94 + (-99.5012) (mistake here, profit is NetProceeds - (-PV(K)) = NetProceeds + PV(K) )
    # Profit = NetProceeds - (-PV(K)) = 98.94 + 99.5012 = 198.44 (this is wrong)
    # Profit = net_proceeds_reverse_conversion - (-present_value_strike) = net_proceeds_reverse_conversion + present_value_strike
    stock_price_2 = 100.0
    call_price_2 = 2.00 # Call is cheap
    put_price_2 = 1.00  # Put is cheap
    # Example for C-P-S > -PV(K): C=3, P=1, S=100. PV(K)=99.5. C-P-S=3-1-100 = -98. -98 > -99.5. Profit = -98 - (-99.5) = 1.5
    call_price_2_rev = 3.0
    put_price_2_rev = 1.0

    opportunity_2 = strategy.identify_arbitrage_opportunity(
        stock_price_2, call_price_2_rev, put_price_2_rev, strike_price,
        risk_free_rate, time_to_expiration
    )
    print(opportunity_2['description'])

    print("\n--- Case 3: No Arbitrage Opportunity ---")
    # PV(K) = 99.5012
    # For Conversion: S+P-C. E.g. S=100, P=2, C=2.5 => 99.5. Costs make it non-profitable.
    # S_buy_eff = 100.05, P_buy_eff = 2.005, C_sell_eff = 2.495. Net Cost = 100.05+2.005-2.495 = 99.56. 99.56 is not < 99.5012
    stock_price_3 = 100.0
    call_price_3 = 2.50
    put_price_3 = 2.00 
    
    opportunity_3 = strategy.identify_arbitrage_opportunity(
        stock_price_3, call_price_3, put_price_3, strike_price,
        risk_free_rate, time_to_expiration
    )
    print(opportunity_3['description'])
    
    return opportunity_1, opportunity_2, opportunity_3

# (如果直接运行此文件)
# if __name__ == "__main__":
#     # ... (其他策略的 demo 调用)
#     demo_conversion_arbitrage()
```

### 3.2 跨资产套利策略

#### 3.2.1 可转债套利策略

**策略原理**：
可转换债券 (Convertible Bond, CB) 是一种混合型证券，它赋予债券持有人在特定时期内按约定价格（转股价）将其转换为发行公司普通股的权利。可转债的价值主要由两部分构成：纯粹的债券价值（债底价值）和内嵌的股票看涨期权价值。

可转债套利策略旨在利用市场价格与其理论估值之间的偏差。当可转债的市场价格低于其理论价值时，存在买入套利机会；反之，则存在卖出（或卖空）套利机会。

**核心概念**：
- **债底价值 (Bond Floor)**：如果可转债不被转换为股票，其作为普通债券的价值，取决于票面利率、到期时间、市场利率以及发行人的信用评级（体现为信用利差）。计算方法为对未来所有债券现金流（利息和本金）按适当的折现率（市场利率+信用利差）进行贴现。
- **转换价值 (Conversion Value)**：如果立即将可转债转换为股票，其等值的股票市场价值。转换价值 = 正股市场价格 × 转换比率。
- **转换比率 (Conversion Ratio)**：每张可转债可转换的股票数量。通常定义为：可转债面值 / 初始转股价。
- **转股价 (Conversion Price)**：债券持有人转换股票时，为获得一股股票所需支付的等效价格。通常在发行时设定，并可能随特定条款（如分红、送股）调整。
- **内嵌期权价值 (Option Value)**：可转债赋予的转换权利的价值，可以看作是一个以转股价为行权价，正股为标的，可转债剩余期限为到期时间的看涨期权。通常用期权定价模型（如Black-Scholes）估算。
- **理论价值** = 债底价值 + 内嵌期权价值。
- **转换溢价率 (Conversion Premium Percentage)**：\\( \\frac{\\text{可转债市场价格} - \\text{转换价值}}{\\text{转换价值}} \\times 100\\% \\)。反映了市场为转换权利以及债底保护所支付的额外价格。

**套利机制**：
1.  **低估套利**：当可转债市场价格显著低于其理论价值时，买入可转债。此策略隐含做多波动率（因为期权价值随波动率上升而上升）。
2.  **高估套利**：当可转债市场价格显著高于其理论价值时，卖空可转债（如果市场允许且成本合理）。此策略隐含做空波动率。

**风险管理与对冲**：
- **Delta中性对冲**：由于可转债价格受其标的股票价格变动的影响（Delta风险），许多套利者会通过买入或卖空相应数量的标的股票来对冲这一风险，旨在剥离股价方向性风险，从而更专注于交易可转债的错误定价或波动率。对冲比例（Delta）需要根据可转债的Delta值动态调整。
- **其他希腊字母风险**：Gamma（Delta变动率）、Vega（波动率敏感度）、Theta（时间价值衰减）、Rho（利率敏感度）等也需要关注和管理。
- **信用风险**：发行人信用状况恶化会导致债底价值下降，进而影响可转债价格。
- **流动性风险**：部分可转债市场流动性可能不足，导致难以按理想价格执行交易。

```python
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional

class ConvertibleBondArbitrageStrategy:
    """可转债套利策略"""

    def __init__(self, credit_spread: float = 0.02):
        """
        初始化可转债套利策略
        Args:
            credit_spread: 发行人的信用利差 (年化), 用于折现债券现金流得到债底价值。
                           例如, 0.02 表示 2%。
        """
        self.credit_spread = credit_spread

    def calculate_straight_bond_value(self, face_value: float, coupon_rate: float, 
                                      time_to_maturity_years: float, market_interest_rate: float, 
                                      payments_per_year: int = 2) -> float:
        """
        计算纯债价值 (债底)
        """
        total_pv = 0
        num_payments = int(np.round(time_to_maturity_years * payments_per_year))
        coupon_payment = (face_value * coupon_rate) / payments_per_year
        # 使用发行人的要求回报率折现，即无风险利率 + 信用利差
        discount_rate_per_period = (market_interest_rate + self.credit_spread) / payments_per_year

        if num_payments <= 0 or time_to_maturity_years <= 1/(365*2): # 如果非常接近到期或已过
             return face_value / (1 + discount_rate_per_period * payments_per_year * time_to_maturity_years) if time_to_maturity_years > 0 else face_value

        for i in range(1, num_payments + 1):
            total_pv += coupon_payment / ((1 + discount_rate_per_period) ** i)
        
        total_pv += face_value / ((1 + discount_rate_per_period) ** num_payments)
        return total_pv

    def calculate_option_component_value(self, stock_price: float, conversion_price: float, 
                                         volatility: float, risk_free_rate: float, 
                                         time_to_maturity_years: float, conversion_ratio: float) -> float:
        """
        使用Black-Scholes模型计算嵌入期权的价值
        Args:
            stock_price: 正股当前价格
            conversion_price: 转股价 (每股转换价格 K_option)
            volatility: 正股年化波动率
            risk_free_rate: 无风险年利率
            time_to_maturity_years: 转债剩余到期时间 (年)
            conversion_ratio: 转换比例 (每张债券可换多少股股票)
        Returns:
            每张债券的期权总价值
        """
        if time_to_maturity_years <= 1/(365*2) or volatility <= 0.001 or conversion_price <=0: # 避免数学错误及处理临期
            # 如果非常临近到期，期权价值趋向于内在价值
            intrinsic_value = max(0, stock_price - conversion_price) * conversion_ratio
            return intrinsic_value

        d1 = (np.log(stock_price / conversion_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_maturity_years) / \
             (volatility * np.sqrt(time_to_maturity_years))
        d2 = d1 - volatility * np.sqrt(time_to_maturity_years)
        
        call_option_value_per_share = (stock_price * norm.cdf(d1) - 
                                       conversion_price * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(d2))
        
        total_option_value = call_option_value_per_share * conversion_ratio
        return max(0, total_option_value) 

    def calculate_theoretical_cb_value(self, face_value: float, coupon_rate: float, market_interest_rate: float,
                                       stock_price: float, conversion_price: float, conversion_ratio: float,
                                       volatility: float, risk_free_rate: float, 
                                       time_to_maturity_years: float, payments_per_year: int = 2) -> Dict:
        """计算可转债的理论价值"""
        straight_bond_val = self.calculate_straight_bond_value(
            face_value, coupon_rate, time_to_maturity_years, market_interest_rate, payments_per_year
        )
        
        option_val = self.calculate_option_component_value(
            stock_price, conversion_price, volatility, risk_free_rate, time_to_maturity_years, conversion_ratio
        )
        
        theoretical_value = straight_bond_val + option_val
        
        return {
            "straight_bond_value": straight_bond_val,
            "option_component_value": option_val,
            "theoretical_cb_value": theoretical_value
        }

    def generate_arbitrage_signals(self, cb_market_price: float, face_value: float, coupon_rate: float, 
                                 market_interest_rate: float, stock_price: float, 
                                 conversion_ratio: float, 
                                 volatility: float, risk_free_rate: float, 
                                 time_to_maturity_years: float,
                                 arbitrage_threshold_pct: float = 0.05, 
                                 payments_per_year: int = 2) -> Dict:
        """
        基于可转债市场价格与理论价值的比较生成套利信号
        """
        if conversion_ratio <= 0:
            conversion_price = float('inf') 
        else:
            conversion_price = face_value / conversion_ratio

        theoretical_values = self.calculate_theoretical_cb_value(
            face_value, coupon_rate, market_interest_rate, stock_price, conversion_price, conversion_ratio,
            volatility, risk_free_rate, time_to_maturity_years, payments_per_year
        )
        theoretical_cb_val = theoretical_values["theoretical_cb_value"]

        current_conversion_value = stock_price * conversion_ratio
        
        if current_conversion_value <= 0:
            conversion_premium_pct = float('inf') if cb_market_price > 0 else 0.0
        else:
            conversion_premium_pct = (cb_market_price - current_conversion_value) / current_conversion_value * 100

        signal_info = {
            'cb_market_price': cb_market_price,
            'theoretical_cb_value': theoretical_cb_val,
            'straight_bond_value': theoretical_values["straight_bond_value"],
            'option_component_value': theoretical_values["option_component_value"],
            'current_conversion_value': current_conversion_value,
            'conversion_premium_pct': conversion_premium_pct,
            'stock_price': stock_price,
            'conversion_price_effective': conversion_price,
            'arbitrage_type': None, 
            'deviation_pct': (cb_market_price - theoretical_cb_val) / theoretical_cb_val * 100 if theoretical_cb_val != 0 else float('inf'),
            'description': "No significant arbitrage opportunity"
        }

        if theoretical_cb_val == 0: # Avoid division by zero if theoretical value is zero
            return signal_info

        if cb_market_price < theoretical_cb_val * (1 - arbitrage_threshold_pct):
            signal_info['arbitrage_type'] = 'Buy_CB' 
            signal_info['description'] = (
                f"Buy CB: Market price ({cb_market_price:.2f}) is significantly below "
                f"theoretical value ({theoretical_cb_val:.2f}). "
                f"Deviation: {signal_info['deviation_pct']:.2f}%"
            )
        elif cb_market_price > theoretical_cb_val * (1 + arbitrage_threshold_pct):
            signal_info['arbitrage_type'] = 'Sell_CB' 
            signal_info['description'] = (
                f"Sell CB: Market price ({cb_market_price:.2f}) is significantly above "
                f"theoretical value ({theoretical_cb_val:.2f}). "
                f"Deviation: {signal_info['deviation_pct']:.2f}%"
            )
        
        return signal_info

# 可转债套利策略使用示例
def demo_convertible_bond_arbitrage():
    """可转债套利策略演示"""
    strategy = ConvertibleBondArbitrageStrategy(credit_spread=0.03) # 3%信用利差

    face_value = 100.0
    coupon_rate = 0.02  
    time_to_maturity = 3.0  
    conversion_ratio = 5.0 # 面值100, 转股价20 (100/5=20)
    
    stock_price = 22.0  
    volatility = 0.30   
    risk_free_rate = 0.015 
    market_interest_rate = 0.02 
    
    print("\n--- Case 1: Undervalued Convertible Bond ---")
    cb_market_price_1 = 105.0 
    
    signal_1 = strategy.generate_arbitrage_signals(
        cb_market_price_1, face_value, coupon_rate, market_interest_rate,
        stock_price, conversion_ratio, volatility, risk_free_rate, time_to_maturity,
        arbitrage_threshold_pct=0.02 
    )
    print(f"CB Market Price: {signal_1['cb_market_price']:.2f}")
    print(f"Theoretical CB Value: {signal_1['theoretical_cb_value']:.2f}")
    print(f"  Straight Bond Value: {signal_1['straight_bond_value']:.2f}")
    print(f"  Option Component Value: {signal_1['option_component_value']:.2f}")
    print(f"Conversion Value (Market): {signal_1['current_conversion_value']:.2f} (Stock {signal_1['stock_price']:.2f} * Ratio {conversion_ratio})")
    print(f"Conversion Premium: {signal_1['conversion_premium_pct']:.2f}%")
    print(f"Effective Conversion Price: {signal_1['conversion_price_effective']:.2f}")
    print(f"Arbitrage Type: {signal_1['arbitrage_type']}")
    print(f"Deviation from Theoretical: {signal_1['deviation_pct']:.2f}%")
    print(f"Description: {signal_1['description']}")

    print("\n--- Case 2: Overvalued Convertible Bond ---")
    cb_market_price_2 = 125.0 
    stock_price_2 = 24.0 
    
    signal_2 = strategy.generate_arbitrage_signals(
        cb_market_price_2, face_value, coupon_rate, market_interest_rate,
        stock_price_2, conversion_ratio, volatility, risk_free_rate, time_to_maturity,
        arbitrage_threshold_pct=0.02
    )
    print(f"CB Market Price: {signal_2['cb_market_price']:.2f}")
    print(f"Theoretical CB Value: {signal_2['theoretical_cb_value']:.2f}")
    print(f"  Straight Bond Value: {signal_2['straight_bond_value']:.2f}")
    print(f"  Option Component Value: {signal_2['option_component_value']:.2f}")
    print(f"Conversion Value (Market): {signal_2['current_conversion_value']:.2f} (Stock {signal_2['stock_price']:.2f} * Ratio {conversion_ratio})")
    print(f"Conversion Premium: {signal_2['conversion_premium_pct']:.2f}%")
    print(f"Effective Conversion Price: {signal_2['conversion_price_effective']:.2f}")
    print(f"Arbitrage Type: {signal_2['arbitrage_type']}")
    print(f"Deviation from Theoretical: {signal_2['deviation_pct']:.2f}%")
    print(f"Description: {signal_2['description']}")

    print("\n--- Case 3: Fairly Valued Convertible Bond ---")
    # Based on Case 1 params, theoretical value is around 111-113. Let's use a price near that.
    cb_market_price_3 = signal_1['theoretical_cb_value'] * 1.005 # Slightly above theoretical but within threshold
    
    signal_3 = strategy.generate_arbitrage_signals(
        cb_market_price_3, face_value, coupon_rate, market_interest_rate,
        stock_price, conversion_ratio, volatility, risk_free_rate, time_to_maturity,
        arbitrage_threshold_pct=0.02
    )
    print(f"CB Market Price: {signal_3['cb_market_price']:.2f}")
    print(f"Theoretical CB Value: {signal_3['theoretical_cb_value']:.2f}")
    print(f"Arbitrage Type: {signal_3['arbitrage_type']}")
    print(f"Deviation from Theoretical: {signal_3['deviation_pct']:.2f}%")
    print(f"Description: {signal_3['description']}")
    
    return signal_1, signal_2, signal_3

if __name__ == "__main__":
    # ... (其他策略的 demo 调用)
    # demo_futures_cash_arbitrage() # Called in its own section
    # demo_conversion_arbitrage()   # Called in its own section
    demo_convertible_bond_arbitrage()
```
