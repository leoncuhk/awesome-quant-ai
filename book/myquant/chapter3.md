# Chapter 3: Arbitrage Strategies

This chapter surveys arbitrage in the broad sense. Unlike the statistical arbitrage strategies of Chapter 2, which rely on historical data and mean reversion, the methods here span a wider range: near-risk-free arbitrage grounded in deterministic pricing relationships (cash-futures arbitrage, options conversion arbitrage), and cross-asset arbitrage such as convertible bond arbitrage, which depends on richer pricing models and active risk management. The goal is to show the diversity of arbitrage trading and where each style applies. All numerical examples in this chapter use synthetic data for illustration.

## 3.1 Risk-Free Arbitrage Strategies

### 3.1.1 Cash-Futures Arbitrage

**Strategy rationale**:
Cash-futures arbitrage exploits mispricing between a futures contract and its underlying spot asset, based on the cost-of-carry model of futures pricing. When the market futures price deviates from its theoretical price (determined by the spot price, the risk-free rate, carry costs, and time to maturity) by more than total transaction costs, an arbitrage opportunity exists.

**Core formula**:
Theoretical futures price \( F_0 = S_0 \cdot e^{(r-q)T} \)
where:
- \( F_0 \) = theoretical futures price
- \( S_0 \) = current spot price
- \( r \) = risk-free rate (continuously compounded)
- \( q \) = continuous dividend yield (or carry yield) of the underlying
- \( T \) = time to maturity (years)

**Arbitrage mechanics**:
1.  **Cash-and-carry arbitrage** (futures overpriced): when \( F_{market} > F_0 + \text{Costs} \)
    - Trades: borrow to buy the spot asset and simultaneously sell the futures contract.
    - Result: a locked-in, riskless profit equal to the overpricing of the futures.
2.  **Reverse cash-and-carry arbitrage** (futures underpriced): when \( F_{market} < F_0 - \text{Costs} \)
    - Trades: short the spot asset (or sell existing holdings) and simultaneously buy the futures contract.
    - Result: a locked-in, riskless profit equal to the underpricing of the futures.

**Implementation notes**:
- **Costs**: commissions, market impact, funding costs, and stock-borrow fees for the short leg.
- **Delivery risk**: physically delivered futures require attention to delivery logistics and costs.
- **Liquidity risk**: both the spot and futures markets must be liquid enough to execute the full package.

For pedagogy, the model below compares single price snapshots and ignores margin requirements, financing spreads, and execution slippage.

```python
import numpy as np
from typing import Dict

class FuturesCashArbitrageStrategy:
    """Cash-futures arbitrage strategy."""

    def __init__(self, transaction_cost_pct: float = 0.001, interest_rate: float = 0.03,
                 dividend_yield: float = 0.01):
        """
        Initialize the strategy.

        Args:
            transaction_cost_pct: one-way transaction cost as a fraction (0.001 = 0.1%)
            interest_rate: risk-free rate (annualized, continuous compounding)
            dividend_yield: annualized dividend yield of the underlying
        """
        self.transaction_cost_pct = transaction_cost_pct
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield

    def calculate_theoretical_futures_price(self, spot_price: float, time_to_maturity_years: float) -> float:
        """
        Theoretical futures price under the cost-of-carry model (continuous compounding).

        Args:
            spot_price: current spot price
            time_to_maturity_years: time to futures expiration (years)

        Returns:
            Theoretical futures price.
        """
        theoretical_price = spot_price * np.exp(
            (self.interest_rate - self.dividend_yield) * time_to_maturity_years
        )
        return theoretical_price

    def identify_arbitrage_opportunities(self, futures_price: float, spot_price: float,
                                         time_to_maturity_years: float) -> Dict:
        """
        Identify cash-futures arbitrage opportunities.

        Args:
            futures_price: market futures price
            spot_price: market spot price
            time_to_maturity_years: time to futures expiration (years)

        Returns:
            Dictionary describing the opportunity (if any).
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

        # Cash-and-carry (positive) arbitrage: buy spot, sell futures
        # (futures overpriced relative to the carry-adjusted spot).
        # Profit at maturity = effective futures sale price
        #                      - carried cost of the effective spot purchase.
        profit_positive = cost_adj_futures_sell - cost_adj_spot_buy * np.exp((self.interest_rate - self.dividend_yield) * time_to_maturity_years)

        if profit_positive > 0:
            opportunity['arbitrage_type'] = 'positive'  # buy spot, sell futures
            # Margin is quoted against the notional spot outlay; real futures
            # positions post margin, so this is a conservative proxy.
            initial_investment_positive = spot_price * (1 + self.transaction_cost_pct)
            opportunity['profit_margin_pct'] = (profit_positive / initial_investment_positive) * 100
            opportunity['description'] = (
                f"Positive arbitrage: Buy spot at {spot_price:.2f}, sell futures at {futures_price:.2f}. "
                f"Theoretical futures price is {theoretical_price:.2f}. "
                f"Expected profit margin: {opportunity['profit_margin_pct']:.4f}%."
            )
            return opportunity

        # Reverse cash-and-carry (negative) arbitrage: sell spot, buy futures
        # (futures underpriced relative to the carry-adjusted spot).
        # Profit at maturity = carried proceeds of the effective spot sale
        #                      - effective futures purchase price.
        profit_negative = cost_adj_spot_sell * np.exp((self.interest_rate - self.dividend_yield) * time_to_maturity_years) - cost_adj_futures_buy

        if profit_negative > 0:
            opportunity['arbitrage_type'] = 'negative'  # sell spot, buy futures
            # Margin is quoted against the notional futures purchase (the spot
            # sale brings in cash up front); real futures positions post
            # margin, so this is a conservative proxy.
            initial_investment_negative = futures_price * (1 + self.transaction_cost_pct)
            opportunity['profit_margin_pct'] = (profit_negative / initial_investment_negative) * 100
            opportunity['description'] = (
                f"Negative arbitrage: Sell spot at {spot_price:.2f}, buy futures at {futures_price:.2f}. "
                f"Theoretical futures price is {theoretical_price:.2f}. "
                f"Expected profit margin: {opportunity['profit_margin_pct']:.4f}%."
            )
            return opportunity

        return opportunity

# Usage example
def demo_futures_cash_arbitrage():
    """Cash-futures arbitrage demo (synthetic data for illustration)."""
    strategy = FuturesCashArbitrageStrategy(
        transaction_cost_pct=0.001,  # 0.1%
        interest_rate=0.03,          # 3%
        dividend_yield=0.01          # 1%
    )

    print("\n--- Case 1: Positive Arbitrage ---")
    # Futures at 103 vs a theoretical price of ~101.01: rich enough to cover
    # costs, so cash-and-carry (buy spot, sell futures) is profitable.
    spot_price_1 = 100.0
    futures_price_1 = 103.0
    time_to_maturity_1 = 0.5  # six months

    opportunity_1 = strategy.identify_arbitrage_opportunities(futures_price_1, spot_price_1, time_to_maturity_1)
    print(f"Spot Price: {spot_price_1}, Futures Price: {futures_price_1}, Maturity: {time_to_maturity_1} years")
    print(f"Theoretical Futures Price: {opportunity_1['theoretical_futures_price']:.2f}")
    print(f"Arbitrage Type: {opportunity_1['arbitrage_type']}")
    print(f"Profit Margin: {opportunity_1['profit_margin_pct']:.4f}%")
    print(f"Description: {opportunity_1['description']}")

    print("\n--- Case 2: Negative Arbitrage ---")
    # Futures at 98 vs a theoretical price of ~100.50: cheap enough that
    # reverse cash-and-carry (sell spot, buy futures) is profitable.
    spot_price_2 = 100.0
    futures_price_2 = 98.0
    time_to_maturity_2 = 0.25  # three months

    opportunity_2 = strategy.identify_arbitrage_opportunities(futures_price_2, spot_price_2, time_to_maturity_2)
    print(f"Spot Price: {spot_price_2}, Futures Price: {futures_price_2}, Maturity: {time_to_maturity_2} years")
    print(f"Theoretical Futures Price: {opportunity_2['theoretical_futures_price']:.2f}")
    print(f"Arbitrage Type: {opportunity_2['arbitrage_type']}")
    print(f"Profit Margin: {opportunity_2['profit_margin_pct']:.4f}%")
    print(f"Description: {opportunity_2['description']}")

    print("\n--- Case 3: No Arbitrage ---")
    # Futures only 0.05% above theoretical: the gap is inside the
    # transaction-cost band, so neither direction is profitable.
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
    demo_futures_cash_arbitrage()
```

### 3.1.2 Conversion Arbitrage

**Strategy rationale**:
Conversion arbitrage and reverse conversion arbitrage are risk-free option strategies based on put-call parity. Parity states that European call and put options on the same underlying, with the same strike and expiration, must satisfy a fixed price relationship.

**Put-call parity**:
\[ S_0 + P_0 = C_0 + K \cdot e^{-rT} \]
where:
- \( S_0 \) = current price of the underlying
- \( P_0 \) = current put price
- \( C_0 \) = current call price
- \( K \) = strike price
- \( r \) = risk-free rate (continuously compounded)
- \( T \) = time to expiration (years)
- \( e^{-rT} \) = discount factor

When market prices deviate from this relationship by more than transaction costs, an arbitrage opportunity exists.

**1. Conversion Arbitrage**
- **Trigger**: \( S_0 + P_0 - C_0 < K \cdot e^{-rT} - \text{Costs} \) — the call is rich relative to the put, so the package costs less than the present value of the \( K \) it is guaranteed to pay.
- **Trades** (a synthetic lending position: pay a known amount today, receive \( K \) at expiration):
    1. Buy the underlying stock (pay \( S_0 \))
    2. Buy the put (pay \( P_0 \))
    3. Sell the call (receive \( C_0 \))
- **Net cost** = \( S_0 + P_0 - C_0 \)
- **Value at expiration**: the package is worth exactly \( K \) regardless of the stock price.
    - If \( S_T > K \): the call is assigned and the stock is delivered at \( K \); the put expires worthless.
    - If \( S_T \le K \): the put is exercised and the stock is sold at \( K \); the call expires worthless.
- **Arbitrage profit** = \( K \cdot e^{-rT} - (S_0 + P_0 - C_0) - \text{Transaction Costs} \)

**2. Reverse Conversion Arbitrage**
- **Trigger**: \( S_0 + P_0 - C_0 > K \cdot e^{-rT} + \text{Costs} \) — the call is cheap relative to the put, so the short package raises more cash than the present value of the \( K \) it must pay at expiration.
- **Trades** (a synthetic borrowing position: receive cash today, repay a known \( K \) at expiration):
    1. Short the underlying stock (receive \( S_0 \))
    2. Sell the put (receive \( P_0 \))
    3. Buy the call (pay \( C_0 \))
- **Net initial proceeds** = \( S_0 + P_0 - C_0 \)
- **Value at expiration**: the package is worth exactly \( -K \) (i.e., \( K \) must be paid) regardless of the stock price.
    - If \( S_T \ge K \): exercise the call to buy the stock back at \( K \); the put expires worthless.
    - If \( S_T < K \): the put is assigned, forcing a stock purchase at \( K \); the call expires worthless.
- **Arbitrage profit** = \( (S_0 + P_0 - C_0) - K \cdot e^{-rT} - \text{Transaction Costs} \)

**Implementation notes**:
- **Transaction costs**: commissions and bid-ask spreads on stock and options, plus stock-borrow fees for the short leg.
- **Option style**: parity holds exactly for European options; American options can deviate slightly due to early exercise.
- **Simultaneous execution**: the arbitrage only locks in if all legs are executed at the quoted prices.
- **Capital usage**: written options (especially naked shorts) require margin.

```python
import numpy as np
from typing import Dict

class ConversionArbitrageStrategy:
    """
    Conversion and reverse conversion arbitrage.
    Based on European put-call parity: S + P = C + K * e^(-rT).
    """

    def __init__(self, stock_tx_cost_pct: float = 0.001, option_tx_cost_per_share: float = 0.001):
        """
        Initialize the strategy.

        Args:
            stock_tx_cost_pct: stock transaction cost as a fraction (0.001 = 0.1%)
            option_tx_cost_per_share: average option transaction cost per share
                (e.g., contract fees spread over the shares per contract)
        """
        self.stock_tx_cost_pct = stock_tx_cost_pct
        self.option_tx_cost_per_share = option_tx_cost_per_share

    def identify_arbitrage_opportunity(self, stock_price: float, call_price: float, put_price: float,
                                       strike_price: float, risk_free_rate: float,
                                       time_to_expiration_years: float) -> Dict:
        """
        Identify a conversion or reverse conversion opportunity (all figures per share).

        Args:
            stock_price: current price of the underlying stock
            call_price: market call price (per share)
            put_price: market put price (per share)
            strike_price: common strike of the call and put
            risk_free_rate: annualized risk-free rate
            time_to_expiration_years: time to option expiration (years)

        Returns:
            Dictionary describing the opportunity (if any).
        """
        present_value_strike = strike_price * np.exp(-risk_free_rate * time_to_expiration_years)

        opportunity = {
            'stock_price': stock_price, 'call_price': call_price, 'put_price': put_price,
            'strike_price': strike_price, 'pv_strike': present_value_strike,
            'arbitrage_type': None,
            'profit_per_share': 0.0,
            'description': "No arbitrage opportunity based on Put-Call Parity"
        }

        # 1. Conversion arbitrage: buy stock (S), buy put (P), sell call (C).
        # Effective package cost per share: S_buy + P_buy - C_sell.
        cost_stock_buy_eff = stock_price * (1 + self.stock_tx_cost_pct)
        cost_put_buy_eff = put_price + self.option_tx_cost_per_share
        proceeds_call_sell_eff = call_price - self.option_tx_cost_per_share

        net_cost_conversion = cost_stock_buy_eff + cost_put_buy_eff - proceeds_call_sell_eff
        # The package is worth K at expiration, so the trade is profitable
        # when net_cost_conversion < PV(K).
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

        # 2. Reverse conversion arbitrage: short stock (-S), sell put (-P), buy call (C).
        # Effective initial proceeds per share: S_sell + P_sell - C_buy.
        proceeds_stock_sell_eff = stock_price * (1 - self.stock_tx_cost_pct)
        proceeds_put_sell_eff = put_price - self.option_tx_cost_per_share
        cost_call_buy_eff = call_price + self.option_tx_cost_per_share

        net_proceeds_reverse_conversion = proceeds_stock_sell_eff + proceeds_put_sell_eff - cost_call_buy_eff
        # The package requires paying K at expiration, so the trade is
        # profitable when net_proceeds_reverse_conversion > PV(K).
        profit_reverse_conversion = net_proceeds_reverse_conversion - present_value_strike

        if profit_reverse_conversion > 0:
            opportunity['arbitrage_type'] = "Reverse Conversion"
            opportunity['profit_per_share'] = profit_reverse_conversion
            opportunity['description'] = (
                f"Reverse Conversion Arbitrage: Sell stock, sell put, buy call. "
                f"Net proceeds per share ({net_proceeds_reverse_conversion:.4f}) > PV of Strike ({present_value_strike:.4f}). "
                f"Profit per share: {profit_reverse_conversion:.4f}"
            )
            return opportunity

        return opportunity

# Usage example
def demo_conversion_arbitrage():
    """Conversion / reverse conversion arbitrage demo (synthetic data for illustration)."""
    # Option prices and transaction costs are all quoted per share.
    strategy = ConversionArbitrageStrategy(stock_tx_cost_pct=0.0005, option_tx_cost_per_share=0.005)
    risk_free_rate = 0.02
    time_to_expiration = 0.25  # three months
    strike_price = 100.0
    # In every case below: PV(K) = 100 * exp(-0.02 * 0.25) = 99.5012.

    print("\n--- Case 1: Conversion Arbitrage Opportunity ---")
    # The call is rich relative to the put, so the conversion package is cheap:
    #   net cost = 100 * 1.0005 + (1.50 + 0.005) - (2.50 - 0.005)
    #            = 100.05 + 1.505 - 2.495 = 99.06 < PV(K) = 99.5012
    #   locked-in profit = 99.5012 - 99.06 = 0.4412 per share
    stock_price_1 = 100.0
    call_price_1 = 2.50
    put_price_1 = 1.50

    opportunity_1 = strategy.identify_arbitrage_opportunity(
        stock_price_1, call_price_1, put_price_1, strike_price,
        risk_free_rate, time_to_expiration
    )
    print(opportunity_1['description'])

    print("\n--- Case 2: Reverse Conversion Arbitrage Opportunity ---")
    # The call is cheap relative to the put, so the short package raises more
    # cash than the present value of the K owed at expiration:
    #   net proceeds = 100 * 0.9995 + (2.00 - 0.005) - (1.00 + 0.005)
    #                = 99.95 + 1.995 - 1.005 = 100.94 > PV(K) = 99.5012
    #   locked-in profit = 100.94 - 99.5012 = 1.4388 per share
    stock_price_2 = 100.0
    call_price_2 = 1.00
    put_price_2 = 2.00

    opportunity_2 = strategy.identify_arbitrage_opportunity(
        stock_price_2, call_price_2, put_price_2, strike_price,
        risk_free_rate, time_to_expiration
    )
    print(opportunity_2['description'])

    print("\n--- Case 3: No Arbitrage Opportunity ---")
    # Prices sit close enough to parity that transaction costs absorb the gap:
    #   conversion net cost   = 100.05 + 2.005 - 2.495 = 99.56 (not < 99.5012)
    #   reversal net proceeds = 99.95 + 1.995 - 2.505 = 99.44 (not > 99.5012)
    stock_price_3 = 100.0
    call_price_3 = 2.50
    put_price_3 = 2.00

    opportunity_3 = strategy.identify_arbitrage_opportunity(
        stock_price_3, call_price_3, put_price_3, strike_price,
        risk_free_rate, time_to_expiration
    )
    print(opportunity_3['description'])

    return opportunity_1, opportunity_2, opportunity_3

if __name__ == "__main__":
    demo_conversion_arbitrage()
```

## 3.2 Cross-Asset Arbitrage Strategies

### 3.2.1 Convertible Bond Arbitrage

**Strategy rationale**:
A convertible bond (CB) is a hybrid security that gives the holder the right to convert the bond into a fixed number of the issuer's common shares at a preset conversion price during a specified period. Its value therefore has two components: the value of the straight bond (the bond floor) and the value of the embedded call option on the stock.

Convertible bond arbitrage exploits deviations between the market price of the CB and its theoretical valuation. When the CB trades below its theoretical value, the arbitrageur buys it; when it trades above, the arbitrageur sells (or shorts) it.

**Core concepts**:
- **Bond floor**: the value of the CB as a straight bond if it is never converted, driven by the coupon rate, time to maturity, market interest rates, and the issuer's credit quality (expressed as a credit spread). It is computed by discounting all bond cash flows (coupons and principal) at an appropriate rate (market rate + credit spread).
- **Conversion value**: the market value of the shares received if the bond were converted immediately. Conversion value = stock price × conversion ratio.
- **Conversion ratio**: the number of shares each bond converts into, typically face value / initial conversion price.
- **Conversion price**: the effective price paid per share upon conversion. Set at issuance and adjusted under specified terms (e.g., dividends, stock splits).
- **Embedded option value**: the value of the conversion right, viewed as a call option on the stock with strike equal to the conversion price and maturity equal to the bond's remaining life. Usually estimated with an option pricing model such as Black-Scholes.
- **Theoretical value** = bond floor + embedded option value.
- **Conversion premium (%)**: \( \frac{\text{CB market price} - \text{conversion value}}{\text{conversion value}} \times 100\% \) — the extra price the market pays for the conversion right plus the bond-floor protection.

**Arbitrage mechanics**:
1.  **Undervaluation trade**: when the CB market price is significantly below its theoretical value, buy the CB. This position is implicitly long volatility (the option value rises with volatility).
2.  **Overvaluation trade**: when the CB market price is significantly above its theoretical value, short the CB (where shorting is available at reasonable cost). This position is implicitly short volatility.

**Risk management and hedging**:
- **Delta-neutral hedging**: because the CB price moves with the underlying stock (delta risk), arbitrageurs typically short (or buy) the appropriate amount of stock to strip out directional exposure, isolating the mispricing / volatility trade. The hedge ratio must be rebalanced as the CB's delta changes.
- **Other Greeks**: gamma (rate of change of delta), vega (volatility sensitivity), theta (time decay), and rho (rate sensitivity) all need monitoring.
- **Credit risk**: deterioration in the issuer's credit lowers the bond floor and hence the CB price.
- **Liquidity risk**: some CB issues trade thinly, making it hard to execute at model prices.

For pedagogy, the model below treats the embedded option as a plain European call and ignores real-world CB features such as call/put provisions, conversion-price resets, and dilution.

```python
import numpy as np
from scipy.stats import norm
from typing import Dict

class ConvertibleBondArbitrageStrategy:
    """Convertible bond arbitrage strategy."""

    def __init__(self, credit_spread: float = 0.02):
        """
        Initialize the strategy.

        Args:
            credit_spread: issuer's annualized credit spread, used to discount
                bond cash flows for the bond floor (0.02 = 2%).
        """
        self.credit_spread = credit_spread

    def calculate_straight_bond_value(self, face_value: float, coupon_rate: float,
                                      time_to_maturity_years: float, market_interest_rate: float,
                                      payments_per_year: int = 2) -> float:
        """
        Straight-bond value (the bond floor).
        """
        total_pv = 0
        num_payments = int(np.round(time_to_maturity_years * payments_per_year))
        coupon_payment = (face_value * coupon_rate) / payments_per_year
        # Discount at the issuer's required return: risk-free market rate + credit spread.
        discount_rate_per_period = (market_interest_rate + self.credit_spread) / payments_per_year

        # Very close to (or past) maturity: discount the principal over the stub period.
        if num_payments <= 0 or time_to_maturity_years <= 1/(365*2):
            return face_value / (1 + discount_rate_per_period * payments_per_year * time_to_maturity_years) if time_to_maturity_years > 0 else face_value

        for i in range(1, num_payments + 1):
            total_pv += coupon_payment / ((1 + discount_rate_per_period) ** i)

        total_pv += face_value / ((1 + discount_rate_per_period) ** num_payments)
        return total_pv

    def calculate_option_component_value(self, stock_price: float, conversion_price: float,
                                         volatility: float, risk_free_rate: float,
                                         time_to_maturity_years: float, conversion_ratio: float) -> float:
        """
        Value of the embedded option using the Black-Scholes model.

        Args:
            stock_price: current price of the underlying stock
            conversion_price: conversion price per share (the option strike)
            volatility: annualized volatility of the stock
            risk_free_rate: annualized risk-free rate
            time_to_maturity_years: remaining life of the CB (years)
            conversion_ratio: shares received per bond upon conversion

        Returns:
            Total option value per bond.
        """
        # Guard against degenerate inputs; near expiration the option value
        # collapses to its intrinsic value.
        if time_to_maturity_years <= 1/(365*2) or volatility <= 0.001 or conversion_price <= 0:
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
        """Theoretical CB value = bond floor + embedded option value."""
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
        Generate arbitrage signals by comparing the CB market price with its theoretical value.
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

        if theoretical_cb_val == 0:  # avoid division by zero below
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

# Usage example
def demo_convertible_bond_arbitrage():
    """Convertible bond arbitrage demo (synthetic data for illustration)."""
    strategy = ConvertibleBondArbitrageStrategy(credit_spread=0.03)  # 3% credit spread

    face_value = 100.0
    coupon_rate = 0.02
    time_to_maturity = 3.0
    conversion_ratio = 5.0  # face value 100, conversion price 100 / 5 = 20

    stock_price = 22.0
    volatility = 0.30
    risk_free_rate = 0.015
    market_interest_rate = 0.02

    print("\n--- Case 1: Undervalued Convertible Bond ---")
    # With these inputs the bond floor is ~91.7 and the option component ~29.0,
    # so the theoretical value is ~120.8. A market price of 105 is far below
    # the 2% threshold band and triggers a Buy signal.
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
    # With the stock at 24 the theoretical value rises to ~128.1; a market
    # price of 135 sits more than 2% above it and triggers a Sell signal.
    cb_market_price_2 = 135.0
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
    # Price the bond 0.5% above the Case 1 theoretical value: inside the
    # 2% no-trade band, so no signal is generated.
    cb_market_price_3 = signal_1['theoretical_cb_value'] * 1.005

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
    demo_convertible_bond_arbitrage()
```
