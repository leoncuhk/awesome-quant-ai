# Chapter 4: High-Frequency Trading Strategies

High-frequency trading (HFT) strategies operate on the microstructure of markets: the order book, the flow of orders hitting it, and the tiny, short-lived price discrepancies that appear across venues. This chapter covers the three pillars of the HFT toolkit — market making (Section 4.1), order flow prediction (Section 4.2), and latency arbitrage (Section 4.3) — plus event-driven high-frequency strategies (Section 4.4). All simulations in this chapter use **synthetic data for illustration**; real implementations require exchange-grade market data, colocation, and far more sophisticated execution logic.

## 4.1 Market Microstructure Strategies

### 4.1.1 Market Making (Bid-Ask Spread Capture)

**Strategy rationale**:
Market making is one of the core strategies in high-frequency trading. The market maker acts as a liquidity provider, continuously posting a competitive buy price (bid) and sell price (ask), and earning the difference between them (the spread). The goal is to capture this tiny per-trade edge repeatedly across a large number of trades, while managing two structural risks: the inventory the strategy accumulates, and the adverse selection it suffers from better-informed counterparties.

- **Liquidity provision**: By quoting continuously, the market maker gives other participants the ability to trade immediately, narrows the market's natural bid-ask spread, and improves market efficiency.
- **Spread capture**: When a buy order lifts the market maker's ask, or a sell order hits its bid, the market maker earns part of the spread. Ideally both sides fill — a "round trip" — locking in one full spread of profit.
- **Risk management**: The main risks a market maker faces are:
    - **Inventory risk**: As orders fill, the market maker accumulates a long or short position (inventory). If the market moves against that position, the inventory loses money.
    - **Adverse selection risk**: The market maker's quotes may be picked off by traders with superior information (e.g., traders who know the price is about to move sharply), leaving the market maker holding an unfavorable position just before the move.

A successful market-making strategy must balance quote attractiveness (a tight spread wins more fills) against risk control (a wide spread reduces toxic fills and limits inventory buildup).

**Core concepts**:
- **Bid price**: The price at which the market maker is willing to buy.
- **Ask price**: The price at which the market maker is willing to sell.
- **Bid-ask spread**: Ask price minus bid price — the market maker's potential per-unit profit.
- **Mid-price**: Usually (bid + ask) / 2, treated as an estimate of the asset's current fair value. The market maker's quotes are typically centered on the mid-price, symmetrically or asymmetrically.
- **Order book**: The set of all outstanding buy and sell orders, organized by price level. The book's depth and shape are key inputs to the market maker's decisions.
- **Inventory**: The net position (long or short) the market maker holds as a result of fills. Managing inventory is one of the central challenges of market making.
- **Target inventory**: The ideal inventory level the market maker wants to maintain — usually zero or a small absolute value.
- **Liquidity**: The market's ability to absorb large trades at stable prices. The market maker supplies liquidity through its quotes.
- **Quote symmetry**: Whether the bid and ask are placed symmetrically around the perceived fair value (e.g., the mid-price). Asymmetric quoting can be used to actively manage inventory or express a mild short-term directional view.
- **Order flow imbalance (OFI)**: The imbalance between buyer-initiated and seller-initiated order activity over a given interval. OFI is a useful predictor of short-term price moves — Section 4.2 develops it in detail.

**Key parameters and models**:
- **Base spread**: The minimum target spread the market maker quotes, usually expressed as a fraction of the mid-price (in basis points).
- **Maximum inventory limit**: A cap on the net position (long or short) the strategy may hold, for risk control.
- **Risk aversion parameter**: In models such as Avellaneda-Stoikov, this parameter quantifies how much the market maker dislikes holding inventory. Higher risk aversion makes the strategy skew its quotes more aggressively to shed inventory faster.
- **Volatility**: The magnitude of price fluctuations. Higher volatility means higher risk, so market makers typically widen the spread to compensate.
- **Order flow information**: Signals derived from the top of book (best bid/ask) and deeper levels — resting quantities, order arrival rates, cancellation rates — used to adjust quotes.
- **Avellaneda-Stoikov model**: A classic optimal market-making model that computes optimal bid and ask quotes dynamically from the market maker's inventory, target inventory, market volatility, and risk aversion. It assumes the asset price follows Brownian motion and derives the optimal quotes by solving a Hamilton-Jacobi-Bellman equation.

**Risk management**:
- **Inventory management**:
    - **Quote skewing**: When inventory deviates from target, shift both quotes asymmetrically. For example, with a long position, lower both the bid and the ask: the cheaper ask attracts buyers (fills that reduce the position), while the lower bid makes it less likely that sellers hit it (fills that would grow the position).
    - **Mean reversion to target inventory**: The strategy continuously steers inventory back toward its target (usually zero).
- **Mitigating adverse selection**:
    - **Dynamic spread adjustment**: Widen the spread when volatility spikes or order flow shows strong one-sided pressure.
    - **Cool-down periods**: A market maker may pause quoting or pull quotes briefly after a fill to avoid being run over by a sequence of informed orders.
- **Stop-loss mechanisms**:
    - **Per-trade / short-horizon loss limits**: If a single trade or a short burst of trades loses more than a preset threshold, pause the strategy or widen the spread substantially.
    - **Gross inventory value limits**: Cap the market value of the total inventory exposure.
- **Volatility monitoring**: Track realized volatility in real time; when it jumps, widen the spread or stop quoting.
- **Event risk handling**: Around known announcements (earnings, macro data releases), pause quoting or widen spreads dramatically — volatility and adverse selection risk are extreme in these windows.
- **Technology and connectivity**: HFT places extreme demands on system speed and reliability; any technical failure can produce serious losses.

The simulation below is **synthetic data for illustration**: prices follow a random walk, and the order flow imbalance input is drawn at random rather than computed from real order book updates (see Section 4.2 for how OFI is actually computed).

```python
import time
import random
from collections import deque
from typing import Tuple

import numpy as np


class MarketMakingStrategy:
    """
    Illustrative high-frequency market-making strategy.

    The strategy provides liquidity by quoting a bid and an ask around the
    market mid-price and profits from the spread. Quotes are adjusted
    dynamically based on current inventory, short-term volatility, and a
    (simplified) order flow imbalance signal.
    """

    def __init__(self,
                 symbol: str,
                 base_spread_bps: float = 10.0,          # base spread in basis points
                 target_inventory: int = 0,               # target inventory
                 max_inventory: int = 1000,               # maximum inventory limit
                 risk_aversion_factor: float = 0.01,      # inventory risk-aversion factor
                 volatility_adjustment_factor: float = 0.5,  # volatility -> spread widening
                 order_flow_adjustment_factor: float = 2.0,  # OFI -> quote skew, in bps of mid
                 tick_size: float = 0.01,                 # minimum price increment
                 logging: bool = True):
        """
        Initialize the market-making parameters.

        Args:
            symbol: Name of the traded instrument.
            base_spread_bps: Base bid-ask spread in basis points (1 bps = 0.01%).
            target_inventory: Desired inventory level.
            max_inventory: Maximum allowed inventory deviation (long or short).
            risk_aversion_factor: Inventory risk aversion. Larger values make
                inventory skew the quotes more aggressively.
            volatility_adjustment_factor: How much volatility widens the spread.
            order_flow_adjustment_factor: Maximum quote skew from order flow
                imbalance, in basis points of the mid-price (OFI is in [-1, 1]).
            tick_size: Minimum price increment allowed by the market.
            logging: Whether to print log messages.
        """
        self.symbol = symbol
        self.base_spread_bps = base_spread_bps
        self.target_inventory = target_inventory
        self.max_inventory = abs(max_inventory)
        self.risk_aversion_factor = risk_aversion_factor
        self.volatility_adjustment_factor = volatility_adjustment_factor
        self.order_flow_adjustment_factor = order_flow_adjustment_factor
        self.tick_size = tick_size
        self.logging = logging

        self.current_inventory: int = 0
        self.pnl: float = 0.0
        self.total_traded_volume: int = 0
        self.bid_price: float = 0.0
        self.ask_price: float = 0.0
        self.mid_price_history = deque(maxlen=100)  # for short-term volatility estimation

        if self.logging:
            print(f"MarketMakingStrategy for {self.symbol} initialized.")
            print(f"  Base Spread: {self.base_spread_bps} bps")
            print(f"  Target Inventory: {self.target_inventory}")
            print(f"  Max Inventory: {self.max_inventory}")
            print(f"  Tick Size: {self.tick_size}")

    def _round_to_tick(self, price: float, direction: str = "nearest") -> float:
        """Snap a price to the nearest valid tick."""
        if direction == "up":       # round up to tick
            return np.ceil(price / self.tick_size) * self.tick_size
        elif direction == "down":   # round down to tick
            return np.floor(price / self.tick_size) * self.tick_size
        else:                       # round to nearest tick
            return np.round(price / self.tick_size) * self.tick_size

    def update_market_data(self, market_mid_price: float, market_best_bid: float, market_best_ask: float):
        """
        Receive a market data update and recompute the optimal quotes.

        Args:
            market_mid_price: Current market mid-price.
            market_best_bid: Current market best bid.
            market_best_ask: Current market best ask.
        """
        self.mid_price_history.append(market_mid_price)

        # 1. Estimate short-term volatility (simplified: std dev of recent mid-prices).
        volatility = np.std(self.mid_price_history) if len(self.mid_price_history) > 10 else self.tick_size * 5
        volatility = max(volatility, self.tick_size)  # floor volatility at one tick

        # 2. Order flow imbalance -- SIMULATED here as a random draw purely for
        #    illustration. In a real system, OFI is computed from order book
        #    updates (see Section 4.2). OFI > 0 = buy pressure, OFI < 0 = sell pressure.
        order_flow_imbalance = random.uniform(-1, 1)

        # 3. Base spread in price units.
        base_spread_amount = market_mid_price * (self.base_spread_bps / 10000.0)

        # 4. Inventory skew: when inventory is above target, shift quotes down
        #    to attract sellers of our inventory; when below target, shift up.
        inventory_delta = self.current_inventory - self.target_inventory
        inventory_skew = -inventory_delta * self.risk_aversion_factor * volatility
        # Scaling by volatility makes inventory control more aggressive in
        # turbulent markets.

        # 5. Volatility adjustment: higher volatility -> wider spread.
        volatility_spread_adjustment = volatility * self.volatility_adjustment_factor

        # 6. Order flow skew: with buy pressure (OFI > 0) shift quotes up;
        #    with sell pressure (OFI < 0) shift them down. Like the base
        #    spread, the skew is expressed in price units as basis points of
        #    the mid-price, so OFI = +/-1 moves the reference price by up to
        #    order_flow_adjustment_factor bps.
        order_flow_skew = order_flow_imbalance * market_mid_price * (self.order_flow_adjustment_factor / 10000.0)

        # 7. Raw bid/ask around the skewed reference price.
        reference_price = market_mid_price + inventory_skew + order_flow_skew

        half_spread = (base_spread_amount / 2) + (volatility_spread_adjustment / 2)

        raw_bid = reference_price - half_spread
        raw_ask = reference_price + half_spread

        # Ensure the spread is non-negative and at least one tick wide.
        if raw_ask - raw_bid < self.tick_size:
            adjustment = (self.tick_size - (raw_ask - raw_bid)) / 2
            raw_bid -= adjustment
            raw_ask += adjustment

        # 8. Snap quotes to valid ticks: bid rounds down, ask rounds up, so we
        #    never accidentally tighten past our intended spread.
        self.bid_price = self._round_to_tick(raw_bid, "down")
        self.ask_price = self._round_to_tick(raw_ask, "up")

        # 9. Risk control: never quote through the market's best prices
        #    (a simplified guard against crossing the market and getting
        #    filled instantly; real market makers use far richer logic).
        self.bid_price = min(self.bid_price, market_best_bid)
        self.ask_price = max(self.ask_price, market_best_ask)

        # Re-check that the spread is still at least one tick.
        if self.ask_price - self.bid_price < self.tick_size:
            self.ask_price = self.bid_price + self.tick_size

        if self.logging:
            print(f"[{time.strftime('%H:%M:%S')}] Mid: {market_mid_price:.2f}, Vol: {volatility:.4f}, OFI: {order_flow_imbalance:.2f}")
            print(f"  Inv: {self.current_inventory}, InvSkew: {inventory_skew:.4f}, OFSkew: {order_flow_skew:.4f}")
            print(f"  Quotes: Bid={self.bid_price:.2f}, Ask={self.ask_price:.2f} (Spread: {self.ask_price - self.bid_price:.2f})")

    def handle_trade_filled(self, trade_price: float, trade_quantity: int, side: str):
        """
        Process a fill notification. `side` is the AGGRESSOR's side:

        Args:
            trade_price: Fill price.
            trade_quantity: Fill quantity.
            side: 'buy'  -> a market buy order lifted our ask (we SOLD);
                  'sell' -> a market sell order hit our bid (we BOUGHT).
        """
        if side == 'buy':    # our ask was lifted -> we sold
            self.pnl += trade_quantity * trade_price
            self.current_inventory -= trade_quantity
            self.total_traded_volume += trade_quantity
            if self.logging:
                print(f"  >>> SOLD {trade_quantity} @ {trade_price:.2f}. New Inv: {self.current_inventory}. PnL: {self.pnl:.2f}")
        elif side == 'sell':  # our bid was hit -> we bought
            self.pnl -= trade_quantity * trade_price
            self.current_inventory += trade_quantity
            self.total_traded_volume += trade_quantity
            if self.logging:
                print(f"  <<< BOUGHT {trade_quantity} @ {trade_price:.2f}. New Inv: {self.current_inventory}. PnL: {self.pnl:.2f}")
        else:
            if self.logging:
                print(f"  Warning: Unknown trade side '{side}'")

        self.risk_management_check()

    def risk_management_check(self):
        """Run risk checks, e.g., the inventory limit."""
        if abs(self.current_inventory) > self.max_inventory:
            if self.logging:
                print(f"  !!! RISK: Inventory {self.current_inventory} exceeds max {self.max_inventory}. Need to reduce risk.")
            # A real system would take corrective action here, for example:
            # - widen the spread
            # - quote one-sided to shed inventory
            # - send aggressive orders to flatten part of the position
            # This example only prints a warning.

    def get_current_pnl_and_inventory_value(self, current_market_mid_price: float) -> Tuple[float, float]:
        """Return total PnL (realized + unrealized) and the mark-to-market inventory value."""
        inventory_value = self.current_inventory * current_market_mid_price
        total_pnl = self.pnl + inventory_value  # realized PnL + unrealized PnL
        return total_pnl, inventory_value


def simulate_market_tick(last_price: float, tick_size: float) -> Tuple[float, float, float, float]:
    """
    Simulate one market tick (synthetic data for illustration).
    Returns: (new mid-price, best bid, best ask, trade price).
    """
    # Random-walk price move.
    price_change = random.choice([-2, -1, 0, 1, 2]) * tick_size
    new_mid_price = round((last_price + price_change) / tick_size) * tick_size
    new_mid_price = max(new_mid_price, tick_size * 10)  # keep the price from collapsing

    # Random market spread.
    market_spread = random.randint(1, 5) * tick_size
    best_bid = new_mid_price - market_spread / 2
    best_ask = new_mid_price + market_spread / 2

    best_bid = round(best_bid / tick_size) * tick_size
    best_ask = round(best_ask / tick_size) * tick_size

    if best_ask - best_bid < tick_size:  # ensure at least one tick of spread
        best_ask = best_bid + tick_size

    # Simulated trade price. Most prints occur at or near the mid, but an
    # occasional aggressive order sweeps several ticks through the book --
    # these are the trades that can reach a market maker's quotes.
    sweep_ticks = random.choice([0] * 6 + [1, 2, 4, 6])
    sweep_side = random.choice([-1, 1])
    trade_price = new_mid_price + sweep_side * sweep_ticks * tick_size

    return new_mid_price, best_bid, best_ask, trade_price


def demo_market_making_strategy():
    """Run the market-making strategy on synthetic data for illustration."""
    print("\n--- Market Making Strategy Demo (synthetic data) ---")
    symbol_demo = "XYZ_STOCK"
    initial_price = 100.0
    tick_size_demo = 0.01

    strategy = MarketMakingStrategy(
        symbol=symbol_demo,
        base_spread_bps=10.0,        # 10 bps = 0.1%
        max_inventory=50,            # max inventory: 50 shares
        risk_aversion_factor=0.005,  # low risk aversion -> gentle inventory skew
        volatility_adjustment_factor=0.6,
        order_flow_adjustment_factor=3.0,  # up to 3 bps (~3 ticks at 100) of quote skew from OFI
        tick_size=tick_size_demo,
        logging=True
    )

    current_market_mid = initial_price
    current_market_bid = initial_price - tick_size_demo * 2  # assumed initial market spread
    current_market_ask = initial_price + tick_size_demo * 2

    num_ticks = 200            # number of simulated ticks
    trade_qty_per_fill = 10    # quantity per fill

    for i in range(num_ticks):
        print(f"\nTick {i+1}/{num_ticks}")

        # 1. The strategy updates its quotes from the current market state.
        strategy.update_market_data(current_market_mid, current_market_bid, current_market_ask)

        # 2. Simulate the next market tick.
        next_market_mid, next_market_bid, next_market_ask, market_trade_price = simulate_market_tick(current_market_mid, tick_size_demo)
        print(f"  Market moves: Mid={next_market_mid:.2f}, Bid={next_market_bid:.2f}, Ask={next_market_ask:.2f}, LastTrade={market_trade_price:.2f}")

        # 3. Check whether the strategy's quotes were filled (simplified logic):
        #    if the simulated trade price crosses one of our quotes, count a fill.
        #    A market BUY lifts our ask...
        if market_trade_price >= strategy.ask_price and strategy.ask_price > 0:
            # Simplification: fixed fill quantity at our quoted price.
            fill_price = strategy.ask_price
            strategy.handle_trade_filled(fill_price, trade_qty_per_fill, 'buy')  # aggressor bought -> we sold

        # ...a market SELL hits our bid.
        elif market_trade_price <= strategy.bid_price and strategy.bid_price > 0:
            fill_price = strategy.bid_price
            strategy.handle_trade_filled(fill_price, trade_qty_per_fill, 'sell')  # aggressor sold -> we bought
        else:
            if strategy.bid_price > 0 and strategy.ask_price > 0:  # only log when we have live quotes
                print(f"  No trade for strategy this tick. Our quotes: Bid {strategy.bid_price:.2f}, Ask {strategy.ask_price:.2f}")

        # Advance the market state for the next iteration.
        current_market_mid = next_market_mid
        current_market_bid = next_market_bid
        current_market_ask = next_market_ask

        time.sleep(0.05)  # simulate inter-tick latency

    # End of demo: compute final PnL.
    final_pnl, final_inventory_value = strategy.get_current_pnl_and_inventory_value(current_market_mid)
    print("\n--- Demo Finished ---")
    print(f"Final PnL (Realized + Unrealized based on last mid price {current_market_mid:.2f}): {final_pnl:.2f}")
    print(f"Final Inventory: {strategy.current_inventory} {symbol_demo} (Value: {final_inventory_value:.2f})")
    print(f"Total Volume Traded by Strategy: {strategy.total_traded_volume}")
    print(f"Final Strategy Quotes: Bid={strategy.bid_price:.2f}, Ask={strategy.ask_price:.2f}")


if __name__ == "__main__":
    demo_market_making_strategy()
```

## 4.2 Order Flow Prediction

### 4.2.1 Order Flow Imbalance (OFI)

**Strategy rationale**:
Order flow is the sequence of orders — submissions, cancellations, and executions — arriving at the order book. Because prices move when incoming demand and supply are imbalanced, order flow contains information about the *next* price move before it happens. Order flow prediction strategies extract that information into a signal and trade (or, for a market maker, skew quotes) in its direction over very short horizons.

The workhorse signal is **order flow imbalance (OFI)**, introduced by Cont, Kukanov, and Stoikov (2014). Instead of trying to classify individual trades as buyer- or seller-initiated, OFI measures net pressure at the top of the book directly from quote updates:

- An **increase** in the bid price, or an increase in the size resting at the bid, adds buying pressure (+).
- A **decrease** in the bid price, or a decrease in bid size, removes buying pressure (−).
- Symmetrically, ask-side changes contribute selling pressure with the opposite sign.

Summed over a short window, OFI has a strong, approximately linear, contemporaneous relationship with mid-price changes, and its persistence gives it modest predictive power for the next interval. In practice OFI is used as: a directional alpha over seconds-scale horizons, a quote-skew input for market making (Section 4.1), and an execution-timing signal for large parent orders.

**Implementation notes**:
- Compute OFI from level-1 (best bid/ask) updates; multi-level generalizations use deeper book levels with decaying weights.
- Always align the signal so that OFI measured over bar *t* predicts the price change over bar *t+1* — evaluating it against the same bar's price change measures contemporaneous correlation, not tradeable alpha.
- Raw OFI scales with typical order sizes; normalize by average depth or recent OFI volatility when comparing across instruments.

```python
import numpy as np
import pandas as pd


def compute_ofi(quotes: pd.DataFrame) -> pd.Series:
    """
    Level-1 order flow imbalance per quote update (Cont-Kukanov-Stoikov, 2014).

    Bid-side contribution per update:
        bid price up            -> +new bid size   (demand added)
        bid price down          -> -old bid size   (demand removed)
        bid price unchanged     -> change in bid size
    Ask-side contribution is symmetric with opposite sign.

    Args:
        quotes: DataFrame with columns ['bid_px', 'ask_px', 'bid_sz', 'ask_sz'],
                one row per quote update, in time order.

    Returns:
        Series of per-update OFI values (positive = net buying pressure).
    """
    bid_px, ask_px = quotes["bid_px"], quotes["ask_px"]
    bid_sz, ask_sz = quotes["bid_sz"], quotes["ask_sz"]
    prev_bid_px, prev_ask_px = bid_px.shift(1), ask_px.shift(1)
    prev_bid_sz, prev_ask_sz = bid_sz.shift(1), ask_sz.shift(1)

    bid_contrib = np.where(bid_px > prev_bid_px, bid_sz,
                  np.where(bid_px < prev_bid_px, -prev_bid_sz,
                           bid_sz - prev_bid_sz))
    ask_contrib = np.where(ask_px < prev_ask_px, ask_sz,
                  np.where(ask_px > prev_ask_px, -prev_ask_sz,
                           ask_sz - prev_ask_sz))

    ofi = pd.Series(bid_contrib - ask_contrib, index=quotes.index, name="ofi")
    return ofi.fillna(0.0)  # first update has no predecessor


def generate_synthetic_quotes(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic level-1 quote data for illustration.

    A persistent latent 'pressure' process drives both the mid-price drift and
    the bid/ask size imbalance, so the demo exhibits the empirically observed
    link between order flow imbalance and price moves.
    """
    rng = np.random.default_rng(seed)
    tick = 0.01
    mid = 100.0
    pressure = 0.0
    rows = []
    for _ in range(n):
        pressure = 0.95 * pressure + rng.normal(0.0, 1.0)          # persistent buy/sell pressure
        mid += 0.0005 * pressure + rng.normal(0.0, 0.002)          # pressure moves the price
        spread = tick * int(rng.integers(1, 3))
        bid_px = round((mid - spread / 2) / tick) * tick
        ask_px = round(bid_px + spread, 2)
        tilt = 0.5 * np.tanh(pressure)                              # pressure tilts resting sizes
        bid_sz = max(1, int(rng.poisson(100 * (1 + tilt))))
        ask_sz = max(1, int(rng.poisson(100 * (1 - tilt))))
        rows.append((bid_px, ask_px, bid_sz, ask_sz))

    quotes = pd.DataFrame(rows, columns=["bid_px", "ask_px", "bid_sz", "ask_sz"])
    quotes["mid"] = (quotes["bid_px"] + quotes["ask_px"]) / 2
    return quotes


def demo_ofi_signal():
    """Show that OFI over bar t predicts the mid-price move over bar t+1."""
    print("\n--- Order Flow Imbalance Signal Demo (synthetic data) ---")
    quotes = generate_synthetic_quotes()
    ofi = compute_ofi(quotes)

    # Aggregate into bars of 50 quote updates.
    bar_size = 50
    bar_id = np.arange(len(quotes)) // bar_size
    ofi_bar = ofi.groupby(bar_id).sum()
    mid_bar = quotes["mid"].groupby(bar_id).last()

    # Predictive alignment (no look-ahead): the OFI of bar t is compared with
    # the mid-price return realized over bar t+1.
    next_bar_ret = mid_bar.pct_change().shift(-1)
    contemporaneous_ret = mid_bar.pct_change()

    print(f"Corr(OFI_t, return over bar t)   [contemporaneous]: {ofi_bar.corr(contemporaneous_ret):.3f}")
    print(f"Corr(OFI_t, return over bar t+1) [predictive]     : {ofi_bar.corr(next_bar_ret):.3f}")

    # A minimal signal rule: go long next bar when OFI exceeds +1 std,
    # short when below -1 std. The std threshold is computed on an EXPANDING
    # window, so the threshold at bar t uses only OFI values up to bar t --
    # a full-sample std would leak future information into the rule.
    threshold = ofi_bar.expanding(min_periods=20).std()
    position = pd.Series(0, index=ofi_bar.index)
    position[ofi_bar > threshold] = 1
    position[ofi_bar < -threshold] = -1
    signal_ret = (position * next_bar_ret).dropna()
    print(f"Bars traded: {(position != 0).sum()}, "
          f"mean next-bar return when positioned: {signal_ret[position != 0].mean():.6f}")


if __name__ == "__main__":
    demo_ofi_signal()
```

The random OFI placeholder inside the Section 4.1 simulator stands in for exactly the `compute_ofi` output above: a production market maker feeds real, book-derived OFI into its quote-skew term instead of a random draw.

## 4.3 Latency Arbitrage

### 4.3.1 Cross-Exchange Latency Arbitrage

**Strategy rationale**:
When the same instrument trades on multiple venues, information reaches those venues at slightly different times. A latency arbitrageur with faster connectivity can observe a price change on one venue and trade against not-yet-updated quotes on another before they adjust. The edge is measured in microseconds to a few milliseconds, and the economics depend entirely on infrastructure: colocation, direct exchange feeds, and microwave/laser links between data centers.

**Why naive price comparison is not arbitrage**: A common beginner mistake is to compare *last-trade* prices across exchanges and flag any difference as an opportunity. That flags noise, not profit, because:

1. **Executable prices are the ask and the bid, not the last trade.** To capture the discrepancy you must *buy at the ask* on the cheap venue and *sell at the bid* on the expensive venue. Last-trade prices routinely differ across venues by roughly a spread without any crossable opportunity existing.
2. **Fees and costs eat the gross edge.** Taker fees on both legs (plus clearing/settlement and, in crypto, withdrawal costs) must be subtracted. Apparent edges smaller than round-trip costs are losses.
3. **Quote staleness creates phantom opportunities.** Feeds from different venues arrive with different delays. Comparing a fresh quote on one venue with a quote that is even a few hundred milliseconds old on another manufactures "arbitrage" that no longer exists. Every quote needs a timestamp, and stale quotes must be excluded.
4. **Size and the race matter.** The opportunity is limited to the displayed size, and other, faster participants are racing for the same quote — fill probability is well below one. A realistic model discounts expected profit by fill probability and adverse re-pricing.

The scanner below incorporates points 1-3 explicitly (executable bid/ask, fees on both legs, and a freshness check) and caps size at the displayed quantities; modeling the race (point 4) is beyond a book example.

```python
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Quote:
    """Top-of-book quote from one venue."""
    bid: float        # best bid price (what we can SELL at)
    ask: float        # best ask price (what we can BUY at)
    bid_size: float   # size displayed at the bid
    ask_size: float   # size displayed at the ask
    timestamp: float  # receive time in seconds (e.g., time.time())


class LatencyArbitrageStrategy:
    """
    Cross-exchange arbitrage scanner over executable (bid/ask) quotes.

    An opportunity is flagged only when, using FRESH quotes:
        net edge = (bid on expensive venue) - (ask on cheap venue) - fees
    exceeds the minimum threshold. Buying happens at the ask, selling at the
    bid -- never at last-trade prices.
    """

    def __init__(self,
                 taker_fee_bps: float = 2.0,      # taker fee per leg, in bps of notional
                 min_net_edge_bps: float = 1.0,   # minimum net edge to act, in bps
                 max_quote_age: float = 0.050):   # maximum quote age in seconds
        self.taker_fee_bps = taker_fee_bps
        self.min_net_edge_bps = min_net_edge_bps
        self.max_quote_age = max_quote_age
        self.quotes: Dict[str, Dict[str, Quote]] = {}  # symbol -> exchange -> Quote

    def update_quote(self, symbol: str, exchange: str, quote: Quote):
        """Store the latest top-of-book quote for (symbol, exchange)."""
        self.quotes.setdefault(symbol, {})[exchange] = quote

    def identify_arbitrage_opportunities(self, symbol: str, now: float = None) -> List[dict]:
        """
        Scan all venue pairs for executable, fee-adjusted, fresh opportunities.

        Args:
            symbol: Instrument to scan.
            now: Current time in seconds; defaults to time.time().

        Returns:
            List of opportunity dicts, possibly empty.
        """
        now = time.time() if now is None else now
        book = self.quotes.get(symbol, {})

        # Freshness check: discard quotes older than max_quote_age. Comparing
        # a live quote with a stale one manufactures phantom arbitrage.
        fresh = {ex: q for ex, q in book.items() if now - q.timestamp <= self.max_quote_age}
        if len(fresh) < 2:
            return []

        opportunities = []
        venues = list(fresh)
        for buy_ex in venues:
            for sell_ex in venues:
                if buy_ex == sell_ex:
                    continue
                buy_q = fresh[buy_ex]    # we BUY at this venue's ASK
                sell_q = fresh[sell_ex]  # we SELL at this venue's BID

                gross_edge = sell_q.bid - buy_q.ask
                if gross_edge <= 0:
                    continue  # books do not cross -- no opportunity

                # Taker fees on both legs.
                fee_cost = (buy_q.ask + sell_q.bid) * self.taker_fee_bps / 1e4
                net_edge = gross_edge - fee_cost
                net_edge_bps = net_edge / buy_q.ask * 1e4
                if net_edge_bps <= self.min_net_edge_bps:
                    continue

                # Executable size is capped by displayed size on both legs.
                size = min(buy_q.ask_size, sell_q.bid_size)
                opportunities.append({
                    "buy_exchange": buy_ex,
                    "sell_exchange": sell_ex,
                    "buy_price": buy_q.ask,
                    "sell_price": sell_q.bid,
                    "net_edge_bps": net_edge_bps,
                    "max_size": size,
                    "expected_profit": net_edge * size,
                })

        return opportunities


def demo_latency_arbitrage():
    """Exercise the scanner on synthetic quotes for illustration."""
    print("\n--- Latency Arbitrage Scanner Demo (synthetic data) ---")
    strat = LatencyArbitrageStrategy(taker_fee_bps=2.0, min_net_edge_bps=1.0, max_quote_age=0.050)
    now = time.time()

    # Case 1: books genuinely cross after fees -> opportunity.
    strat.update_quote("BTC-USD", "VenueA", Quote(bid=50000.0, ask=50001.0, bid_size=2.0, ask_size=1.5, timestamp=now))
    strat.update_quote("BTC-USD", "VenueB", Quote(bid=50040.0, ask=50041.0, bid_size=1.0, ask_size=2.0, timestamp=now))
    print("Crossed fresh quotes:", strat.identify_arbitrage_opportunities("BTC-USD", now=now))

    # Case 2: same prices, but VenueB's quote is 500 ms old -> filtered out.
    strat.update_quote("BTC-USD", "VenueB", Quote(bid=50040.0, ask=50041.0, bid_size=1.0, ask_size=2.0, timestamp=now - 0.5))
    print("Stale expensive-venue quote:", strat.identify_arbitrage_opportunities("BTC-USD", now=now))

    # Case 3: last trades differ but books do not cross -> no opportunity.
    strat.update_quote("BTC-USD", "VenueB", Quote(bid=50000.5, ask=50001.5, bid_size=1.0, ask_size=2.0, timestamp=now))
    print("Uncrossed books:", strat.identify_arbitrage_opportunities("BTC-USD", now=now))


if __name__ == "__main__":
    demo_latency_arbitrage()
```

## 4.4 Event-Driven High-Frequency Strategies

### 4.4.1 News Event Strategy

**Strategy rationale**:
News moves prices, and the first minutes after a release are when the information is incorporated. An event-driven HFT strategy scores incoming news for sentiment, confirms that the market is actually reacting (e.g., via an abnormal volume spike), and takes a short-horizon directional position in the direction of the sentiment. Requiring *both* a strong sentiment score and a volume confirmation filters out news the market ignores.

The implementation below combines two sentiment sources: a general-purpose model (TextBlob polarity) and a domain-specific keyword lexicon that catches finance terms generic models often miss. For non-English news feeds, swap in a local-language lexicon and sentiment model. Note that production systems use low-latency machine-readable news feeds and far more sophisticated NLP; this example illustrates the signal structure.

```python
import numpy as np
import pandas as pd
from textblob import TextBlob  # pip install textblob


class NewsEventStrategy:
    def __init__(self, sentiment_threshold: float = 0.1, volume_multiplier: float = 2.0):
        self.sentiment_threshold = sentiment_threshold
        self.volume_multiplier = volume_multiplier
        # Domain-specific lexicon: finance terms a generic sentiment model may
        # miss. For non-English news, replace with a local-language lexicon.
        self.keywords = {
            'positive': ['beats expectations', 'breakout', 'upgrade', 'growth',
                         'acquisition', 'partnership', 'record revenue'],
            'negative': ['misses expectations', 'downgrade', 'loss', 'risk',
                         'investigation', 'lawsuit', 'recall'],
        }

    def analyze_sentiment(self, news_text: str) -> float:
        """Score news sentiment in [-1, 1] by combining a general model with a keyword lexicon."""
        # General-purpose sentiment (TextBlob polarity, [-1, 1]).
        blob = TextBlob(news_text)
        polarity = blob.sentiment.polarity

        # Domain keyword score.
        text_lower = news_text.lower()
        positive_count = sum(1 for word in self.keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.keywords['negative'] if word in text_lower)
        keyword_sentiment = (positive_count - negative_count) / max(positive_count + negative_count, 1)

        # Combined sentiment score.
        combined_sentiment = (polarity + keyword_sentiment) / 2
        return combined_sentiment

    def generate_signals(self, news_data: dict, price_data: pd.Series, volume_data: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals from news events, confirmed by volume spikes.

        Args:
            news_data: {timestamp: {'content': str}} news items.
            price_data: Price series (defines the signal index).
            volume_data: Volume series aligned with price_data.

        Returns:
            DataFrame with columns ['news_signal', 'news_sentiment'].
        """
        signals = pd.DataFrame(0.0, index=price_data.index,
                               columns=['news_signal', 'news_sentiment'])

        # Volume baseline: rolling mean of the PRIOR 20 bars. shift(1) keeps
        # the current bar's (possibly spiking) volume out of its own baseline
        # and guarantees no look-ahead.
        avg_volume = volume_data.rolling(window=20).mean().shift(1)

        for timestamp, news in news_data.items():
            if timestamp not in signals.index:
                continue  # news outside the trading index

            sentiment = self.analyze_sentiment(news['content'])

            # Volume confirmation: is the market reacting?
            current_volume = volume_data.loc[timestamp]
            baseline = avg_volume.loc[timestamp]
            if pd.isna(baseline):
                continue  # not enough history for a baseline
            volume_spike = current_volume > baseline * self.volume_multiplier

            # Signal: strong sentiment AND abnormal volume.
            if abs(sentiment) > self.sentiment_threshold and volume_spike:
                signal_strength = min(abs(sentiment) * 2, 1.0)  # cap signal strength
                signals.loc[timestamp, 'news_signal'] = np.sign(sentiment) * signal_strength
                signals.loc[timestamp, 'news_sentiment'] = sentiment

        return signals


def demo_news_event_strategy():
    """Run the news strategy on synthetic data for illustration."""
    print("\n--- News Event Strategy Demo (synthetic data) ---")
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-02 09:30", periods=120, freq="min")
    prices = pd.Series(100 + rng.normal(0, 0.05, len(idx)).cumsum(), index=idx)
    volume = pd.Series(rng.integers(800, 1200, len(idx)).astype(float), index=idx)

    # Inject a news event with an accompanying volume spike.
    news_time = idx[60]
    volume.loc[news_time] = 5000.0
    news_data = {
        news_time: {"content": "Company beats expectations and announces "
                               "acquisition; record revenue drives strong growth"}
    }

    strategy = NewsEventStrategy(sentiment_threshold=0.1, volume_multiplier=2.0)
    signals = strategy.generate_signals(news_data, prices, volume)

    fired = signals[signals['news_signal'] != 0]
    print(f"Signals fired: {len(fired)}")
    print(fired)


if __name__ == "__main__":
    demo_news_event_strategy()
```
