# Chapter 8: Options Strategies

Options let a quantitative trader take positions not only on price direction, but on volatility, time decay, and the shape of the return distribution itself. This chapter builds the core options toolkit: computing Black-Scholes Greeks, constructing and rebalancing delta-neutral portfolios, monetizing convexity through gamma scalping, and expressing volatility views with straddles, strangles, and calendar spreads. The unifying theme is that a delta-hedged book is not riskless — it is a deliberate position in gamma, theta, and vega, and profitable trading comes from managing the interaction between those Greeks.

## 8.1 Delta-Neutral Strategies and the Greeks

A delta-neutral portfolio pairs options with an offsetting stock position so that the net first-order exposure to the underlying is approximately zero. What remains is exposure to the higher-order Greeks: long gamma positions profit from realized movement (paid for via theta decay), while short gamma positions collect theta but lose on large moves. The class below implements the Black-Scholes Greeks, initial hedge construction, threshold-based rebalancing, and a gamma-scalping simulation whose per-step P&L is approximately `0.5 * gamma * move^2` from scalping, minus transaction costs, plus (negative) theta.

```python
import numpy as np
import pandas as pd
from scipy.stats import norm


class DeltaNeutralStrategy:
    def __init__(self, rebalance_threshold=50):
        # Threshold on the aggregate portfolio delta, measured in
        # SHARE-EQUIVALENTS (option delta x quantity x multiplier). With
        # 100-share contracts even a small book carries deltas in the
        # hundreds, so a fractional threshold like 0.05 would fire on
        # every bar; 50 share-equivalents is a sensible default. Scale it
        # with position size, or normalize delta before comparing.
        self.rebalance_threshold = rebalance_threshold
        self.positions = {}

    def calculate_greeks(self, stock_price, strike_price, time_to_expiry,
                         risk_free_rate, volatility, option_type='call'):
        """Compute Black-Scholes Greeks for a European option."""
        d1 = (np.log(stock_price / strike_price) +
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (identical for calls and puts)
        gamma = norm.pdf(d1) / (stock_price * volatility * np.sqrt(time_to_expiry))

        # Theta: shared time-decay term, plus a carry term whose sign
        # differs between calls and puts. Converted to per calendar day.
        decay_term = (-stock_price * norm.pdf(d1) * volatility /
                      (2 * np.sqrt(time_to_expiry)))
        carry_term = (risk_free_rate * strike_price *
                      np.exp(-risk_free_rate * time_to_expiry))
        if option_type == 'call':
            theta = (decay_term - carry_term * norm.cdf(d2)) / 365
        else:
            theta = (decay_term + carry_term * norm.cdf(-d2)) / 365

        # Vega, per 1 percentage point change in volatility
        vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

    def create_delta_neutral_portfolio(self, option_positions, stock_price):
        """Build a delta-neutral portfolio by hedging the options book with stock."""
        total_delta = 0
        portfolio_value = 0

        # Aggregate delta of the option book
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

        # Stock quantity required to neutralize the aggregate delta
        stock_hedge_quantity = -total_delta

        return {
            'stock_hedge_quantity': stock_hedge_quantity,
            'total_delta': total_delta,
            'portfolio_value': portfolio_value
        }

    def rebalance_portfolio(self, current_positions, new_stock_price):
        """Re-hedge when the portfolio delta drifts beyond the threshold.

        Works on copies of the position dicts, so the caller's data is
        never mutated (same contract as gamma_scalping_strategy).
        """
        current_delta = 0
        options = [dict(p) for p in current_positions['options']]

        for position in options:
            # One day has passed; roll down the time to expiry
            position['time_to_expiry'] = max(position['time_to_expiry'] - 1 / 365, 0.001)

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

        # Add the delta of the stock hedge (delta of stock is 1 per share)
        current_delta += current_positions['stock_quantity']

        # Rebalance only when the drift exceeds the threshold
        if abs(current_delta) > self.rebalance_threshold:
            adjustment = -current_delta

            return {
                'rebalance_needed': True,
                'stock_adjustment': adjustment,
                'new_stock_position': current_positions['stock_quantity'] + adjustment
            }

        return {'rebalance_needed': False}

    def gamma_scalping_strategy(self, option_positions, stock_price_path,
                                transaction_cost=0.001):
        """Simulate a gamma-scalping P&L series along a given price path.

        The stock hedge is tracked in shares (initial hedge plus re-hedge
        trades), but daily P&L uses the standard gamma-theta decomposition
        (0.5 * gamma * move^2 minus theta decay minus trading costs) rather
        than marking the stock leg to market.
        """
        pnl_series = []
        # Copy each position dict so we do not mutate the caller's data
        positions = [dict(p) for p in option_positions]
        stock_position = 0.0

        for i, stock_price in enumerate(stock_price_path):
            daily_pnl = 0

            # Aggregate Greeks of the current option book
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

            if i == 0:
                # Establish the initial delta hedge against the option book
                initial_hedge = -total_delta
                stock_position += initial_hedge
                daily_pnl -= abs(initial_hedge) * stock_price * transaction_cost
            else:
                stock_move = stock_price - stock_price_path[i - 1]

                # Gamma P&L: convexity accrues on EVERY move, whether or
                # not a re-hedge trade fires below
                daily_pnl += 0.5 * total_gamma * stock_move**2

                # Re-hedge: the delta change is approximately gamma * price move
                theoretical_trade = -total_gamma * stock_move

                # Execute only above a minimum trade threshold
                if abs(theoretical_trade) > 0.01:
                    stock_position += theoretical_trade

                    # Transaction costs
                    trade_cost = abs(theoretical_trade) * stock_price * transaction_cost
                    daily_pnl -= trade_cost

            # Time decay (theta is already per day)
            theta_decay = sum(
                self.calculate_greeks(
                    stock_price, pos['strike'], pos['time_to_expiry'],
                    pos['risk_free_rate'], pos['volatility'], pos['option_type']
                )['theta'] * pos['quantity']
                for pos in positions
            )

            daily_pnl += theta_decay

            pnl_series.append(daily_pnl)

            # Roll down time to expiry for the next step
            for position in positions:
                position['time_to_expiry'] = max(position['time_to_expiry'] - 1 / 365, 0.001)

        return pd.Series(pnl_series, index=range(len(stock_price_path)))
```

## 8.2 Volatility Trading Strategies

Volatility strategies express a view on how much the underlying will move rather than in which direction. A long straddle (buy an at-the-money call and put) or a long strangle (buy out-of-the-money wings) is long vega and gamma, profiting when realized volatility exceeds what was implied at entry. A calendar spread sells a short-dated option and buys a longer-dated one at the same strike, harvesting the faster time decay of the near contract while staying long vega. The class below sizes these structures and reports their aggregate Greeks and break-even points.

```python
import numpy as np
import pandas as pd
from scipy.stats import norm


class VolatilityTradingStrategy:
    def __init__(self):
        self.positions = {}

    def long_volatility_strategy(self, stock_price, strike_prices, time_to_expiry,
                                 risk_free_rate, implied_vol, realized_vol):
        """Construct long-volatility structures (straddle and strangle).

        `realized_vol` must be TRAILING realized volatility (or a model
        forecast) known at trade time — passing volatility realized over
        the subsequent holding period would be look-ahead bias.
        """
        strategies = {}

        # Long straddle: buy the ATM call and ATM put at the same strike
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
            'breakeven_down': atm_strike - (call_greeks['price'] + put_greeks['price']),
            # Positive edge when realized volatility exceeds implied volatility
            'vol_edge': realized_vol - implied_vol
        }

        # Long strangle: buy the nearest OTM call and OTM put (cheaper, wider break-evens)
        otm_call_strike = min([s for s in strike_prices if s > stock_price],
                              default=strike_prices[-1])
        otm_put_strike = max([s for s in strike_prices if s < stock_price],
                             default=strike_prices[0])

        strategies['long_strangle'] = {
            'positions': [
                {'type': 'call', 'strike': otm_call_strike, 'quantity': 1},
                {'type': 'put', 'strike': otm_put_strike, 'quantity': 1}
            ]
        }

        return strategies

    def calendar_spread_strategy(self, stock_price, strike_price, short_expiry,
                                 long_expiry, risk_free_rate, short_vol, long_vol):
        """Calendar spread: sell the near-dated option, buy the far-dated one."""
        short_option = self.calculate_option_greeks(
            stock_price, strike_price, short_expiry, risk_free_rate, short_vol, 'call'
        )
        long_option = self.calculate_option_greeks(
            stock_price, strike_price, long_expiry, risk_free_rate, long_vol, 'call'
        )

        # Net Greeks of the position: long the far option, short the near one
        net_premium = long_option['price'] - short_option['price']
        net_theta = long_option['theta'] - short_option['theta']
        net_vega = long_option['vega'] - short_option['vega']

        return {
            'net_premium': net_premium,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'max_profit_price': strike_price,  # maximum profit near the strike
            # Time decay favors the position when its net theta is positive,
            # i.e. the short near-dated leg decays faster than the long leg
            'time_decay_advantage': net_theta > 0
        }

    def calculate_option_greeks(self, S, K, T, r, sigma, option_type):
        """Black-Scholes price and Greeks for a European option."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta: the carry term enters with opposite signs for calls and puts
        decay_term = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            theta = decay_term - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = decay_term + r * K * np.exp(-r * T) * norm.cdf(-d2)

        vega = S * norm.pdf(d1) * np.sqrt(T)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # per calendar day
            'vega': vega / 100     # per 1 percentage point of volatility
        }
```
