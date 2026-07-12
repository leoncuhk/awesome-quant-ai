# Systematic Trading: A unique new method for designing trading and investing systems

Reading notes on Robert Carver's *Systematic Trading* (2015).

> **Scope of these notes:** They cover the front portion of the book — the Introduction, Part One (Theory: psychology, simple rules, sticking to the plan, trading styles), the Part Two toolbox chapters on fitting and portfolio allocation, and the beginning of Part Three (the framework overview) up through the Forecasts chapter. The remaining framework chapters (volatility targeting, position sizing, trading speed and costs, portfolio construction) and Part Four (Practice) are not covered.

A trading system is a complete decision logic for dealing with uncertainty in price action: where to enter, what to do when a position moves against you, what to do when it moves in your favor, where to place the stop, where to take profit, and when to add to a position. It is a consistent lens for observing otherwise chaotic charts.

## Learning Strategy

1. **Understanding the Basics**
   - **Preface** and **Introduction**: Start with these sections to understand the motivation behind systematic trading and the scope of the book.

2. **Theory**
   - **Part One**: Focus on understanding why systematic trading is necessary and the psychological biases that affect human traders.

3. **Toolbox**
   - **Part Two**: Learn about the tools and techniques used in systematic trading, such as fitting models and portfolio allocation.

4. **Framework**
   - **Part Three**: Study the detailed framework for creating systematic strategies, including instrument selection, forecasting, and risk management.

5. **Practice**
   - **Part Four**: Apply the learned concepts to real-world scenarios tailored to different types of traders (semi-automatic trader, asset allocating investor, staunch systems trader).

6. **Additional Resources**
   - Use the **Glossary** and **Appendices** for definitions and further reading to deepen your understanding of specific concepts and technical details.

By following this structure, you can systematically approach the book and build a strong foundation in designing and implementing trading and investing systems.

## Three Archetypal Systematic Traders and Investors

**Key points:**

- **Asset allocating investor**: Spreads capital across asset classes, avoids chasing whatever is currently fashionable, and typically uses no leverage. Carver uses an ETF portfolio to show how the systematic framework applies to asset allocation.
- **Semi-automatic trader**: Trades on discretionary market opportunities, uses leverage and derivatives, but manages positions and risk through the systematic framework. Carver illustrates this with a UK equity trading example.
- **Staunch systems trader**: Relies entirely on systematic trading rules to forecast price moves, and uses the framework to manage positions and risk. Systems traders typically develop and validate rules with backtesting software; Carver shows how to use such tools safely.

**Structure of the book:**

- **Theory**: Why you should run a systematic strategy, and an overview of available trading styles.
- **Toolbox**: The two key techniques for building systematic strategies: backtesting/fitting and portfolio optimization.
- **Framework**: A complete, extensible framework for constructing systematic strategies.
- **Practice**: How to apply the framework to the three archetypal traders and investors.

**Limits of human psychology:**

- Humans still outperform computers at complex intellectual tasks, but emotion prevents us from exploiting that advantage.
- Fear and greed routinely produce bad decisions — even experienced investors are not immune.

**Advantages of systematic trading:**

- By removing emotional interference, systematic trading makes decisions more rational and consistent.
- In extreme market conditions, systems keep running and can stay profitable while humans tend to panic.

A trading system should be modular, like a car — components can be adjusted and upgraded independently. Successful systems trading does not depend on a magical system, but on avoiding common mistakes: over-complex systems, over-optimism, taking excessive risk, and trading too frequently. Behavioral research shows that humans are systematically prone to error in financial decisions. This is why simple trading rules often outperform clever humans — and why a system only works if you actually stick to it.

**Human cognitive flaws:**

- Despite the brain's sophistication, cognitive biases produce systematically irrational behavior.
- These instinctive reactions usually overwhelm our natural decision-making advantages, which is why simple decision rules tend to work better.

**Prospect Theory:**

- Explains why investors make predictable mistakes in decisions such as whether to exit a losing position.
- Most people are reluctant to close a loser, because doing so means admitting a mistake — and a realized loss hurts more than a paper loss.
- The instinct is to deny that a loss is real until it is crystallized, so people postpone selling losing positions.
- Conversely, when a position is profitable, people rush to take the gain — locking in confirmation that the buy decision was right and avoiding potential regret.

**Why cutting winners and running losers is bad:**

- Taking profits early while letting losses run is usually a poor strategy.
- Backtests show that a take-profit-early rule generally underperforms a cut-losses-early rule.

## Simple Trading Rules

1. **Why systematic rules help:**
   - Systematic trading rules correct for serious flaws in human instinct.
   - They can also exploit the cognitive biases of other human traders, profiting from market anomalies.

2. **Cognitive biases and trading rules:**
   - A rule that cuts losses early, for example, does more than fix your own instincts — it earns extra return from other traders doing the opposite.

3. **Why rules keep working:**
   - Cognitive biases explain why certain trading rules have been persistently profitable, and why they are likely to remain effective in the future.

## Sticking to the Plan

1. **Why adherence matters:**
   - There is no point running a systematic strategy if you cannot stick to it.
   - Faced with losses (or gains), human instinct interferes with execution and pushes you off the predefined rules.

2. **Avoiding intervention:**
   - You need a commitment mechanism to enforce adherence to the system.
   - Odysseus had his crew bind him and ignore his orders so he could resist the Sirens.
   - A modern example: Victor Niederhoffer had an assistant with instructions to forcibly close his positions once a loss limit was hit.

3. **Objective systems:**
   - For commitment to be enforceable, the system must be objective.
   - Many systems are not fully objective and leave room for subjective interpretation.
   - A purely objective system is better at preventing human interference and also permits reliable backtesting.

## Automation — The Use of Dogs in Finance and Engineering

**Main points:**

1. **Objective systems can be automated**, and automation is itself a good commitment mechanism.
   - An automated system needs unambiguous rules — e.g. "buy when the 20-day moving average is above the 40-day moving average" — with no reliance on subjective judgment.

2. **Implementing automation:**
   - Simple systems can be run manually with pen and paper or a spreadsheet, but complex rules or higher trading frequency require automation.
   - A semi-automatic trader can automate the position and risk management framework that wraps around their discretionary decisions.

3. **Automation does not fully prevent interference:**
   - A system that generates trades automatically but requires manual execution is easy to override.
   - A fully automated system is harder to interfere with, but still requires trust and sound design.

4. **The dog analogy:**
   - The ideal systems trading setup consists of a computer, a dog, and a human. The computer runs the fully automated strategy, the human feeds the dog, and the dog bites the human if he touches the computer.

**Design traps to avoid:**

- **Over-fitting:**
  - The human brain is wired to see patterns in data that do not exist, which leads to over-fitting.
  - Over-fitted systems usually underperform simple rules in live trading.
- **Overtrading:**
  - Overconfidence leads to trading too frequently, driving up transaction costs.
  - Frequent trading resembles gambling and is rarely profitable.
- **Over-betting:**
  - Excessive leverage raises the risk of blowing up the account, and makes human intervention almost inevitable.

**Self-fulfilling prophecies:**

- **Definition**: Some technical trading systems have no theoretical basis but work because enough people use them.
- **Example**: Fibonacci levels in technical analysis.

**Where trading rule profits come from:**

- **Risk premia**: Extra return earned by bearing specific risks.
- **Skew**: Positive-skew strategies (e.g. buying options) and negative-skew strategies (e.g. selling options) have very different return profiles.
- **Behavioral effects**: Profiting from the behavioral weaknesses of other market participants.

**Classifying trading styles:**

- **Static vs dynamic**: Static strategies trade rarely; dynamic strategies trade frequently.
- **Technical vs fundamental**: Technical strategies use price data; fundamental strategies use micro and macro data.
- **Portfolio size**: Diversified portfolios reduce risk and improve returns.
- **Leverage**: Use leverage sensibly and avoid exposure to extreme losses.

**Trading speed:**

- **Very slow**: Behaves like a static portfolio, with lower returns.
- **Medium speed**: The most attractive returns for most traders.
- **Fast**: High-frequency trading requires careful attention to transaction costs and technology.

**Realistic Sharpe ratios:**

- Set reasonable expectations. Inflated expectations lead directly to overtrading and over-betting.

## Fitting

This chapter covers how to use data to select and calibrate trading rules — one of the critical steps in systematic trading. Carver stresses the danger of over-fitting and offers guidance for avoiding the common traps.

### The Dangers of Fitting

1. **Over-fitting**: Selecting and calibrating too many trading rules produces rules that look excellent on historical data but fail in live trading. Over-fitting yields deceptively optimistic backtest results.

2. **No time machines**: Do not fit and test on the same full dataset — that produces over-optimistic results. Use rolling or expanding windows instead.

3. **Rule-selection risk**: Picking a single rule and discarding the rest breeds overconfidence. Distinguishing genuinely better rules from worse ones usually requires very long data histories.

4. **Pooling data**: For sound fitting decisions, pool data across multiple instruments. Only fit instruments separately when they are statistically significantly different.

5. **Comparing like with like**: When comparing rules by Sharpe ratio (SR), beware of the difference between positive-skew and negative-skew rules — negative-skew rules tend to show inflated SRs.

6. **The future will differ from the past**: Focus on returns relative to a benchmark rather than absolute returns. Much of the high absolute return of the past 40 years came from the large fall in inflation, which will not repeat.

### Rules for Effective Fitting

If you do decide to fit, follow these guidelines:

1. **Keep it simple**: Avoid sophisticated fitting methods.
2. **Limit the candidate set**: Reduce the number of trading rules and variations under consideration, unless you have many years of data.
3. **Ban time machines**: Fit with rolling or expanding windows, and avoid windows that are too short.
4. **Don't discard rules lightly**: Before selecting one rule or variation and dropping the rest, carefully consider their correlations and performance differences.
5. **Pool your data**: Combine data across instruments to make fitting decisions more reliable.
6. **Compare like with like**: When using Sharpe ratios to compare rules, account for skew differences.
7. **The future will differ from the past**: Judge by benchmark-relative returns; do not rely on historical absolute returns.

### Carver's Own Approach to Rule Selection

Carver prefers to avoid fitting almost entirely, choosing rules and variations *without looking at their actual performance*:

1. **Propose a small number of trading rules** based on how markets are believed to behave.
2. **Choose a few variations of each rule** — at this stage based not on performance but on trading speed and correlation with other variations.
3. **Assign forecast weights** based on the uncertainty of Sharpe ratio estimates. Poor rules get lower weight but are rarely excluded entirely.

Under this approach, the actual historical performance data is reserved for setting forecast weights, which avoids over-fitting.

### The Time-Machine Trap: Testing Schemes

- **In-sample testing**: Fitting and testing on the entire dataset — produces over-optimistic results.
- **Out-of-sample testing**: Splitting the data, fitting on the first half and testing on the second — valid, but wastes data.
- **Expanding window**: Refit as each new data point is added, always using all history to date.
- **Rolling window**: Refit periodically, discarding old data and using only the most recent window.

### Key Takeaways

- Avoid over-fitting; use ample historical data.
- Keep the number of rules and variations small; avoid complexity.
- Pool data to make fitting decisions more reliable.
- Do not select rules on historical performance alone — factor in trading costs and correlations.

The chapter's core message: simplicity and data pooling are what make trading rules hold up in live use.

## Portfolio Allocation

This chapter covers how to allocate trading capital across instruments or trading rules — essential for systematic investors and traders, especially systems traders running multiple rules.

### Portfolio Optimization

Portfolio optimization seeks the combination of asset weights that delivers the best expected risk-adjusted return, usually measured by the Sharpe ratio. The method dates back to Harry Markowitz in the 1950s. Mathematically elegant, but applied naively it produces extreme portfolio weights.

### Limitations of Optimization

1. **Extreme weights:**
   - Classical Markowitz optimization produces extreme portfolio weights that are likely to fail in practice.
   - Extreme portfolios concentrate most capital in a few assets and ignore the potential value of the rest.

2. **Unstable weights:**
   - Example: allocating across NASDAQ, the S&P 500, and 20-year US Treasuries with single-period optimization yields highly unstable weights, sometimes excluding assets entirely.

### Fixing the Optimization Problem

Two methods improve the process:

1. **Bootstrapping:**
   - Repeat the optimization many times over resampled data and average the results, avoiding extreme weights.
   - Even with noisy data, the averaged result reflects the underlying reality.

2. **Handcrafting:**
   - Construct weights by hand, combining historical data with judgment-based estimates, to ensure the portfolio is stable and sensible.

### Handcrafting

Handcrafting builds the portfolio bottom-up: allocate weights within small groups first, then across groups. The method relies on asset correlations and expected Sharpe ratios.

**Steps:**

1. **Form asset groups**: Group assets by correlation — high correlation within groups, low correlation between groups.
2. **Allocate within groups**: Assign intra-group weights using the reference table (Table 8 in the book).
3. **Allocate across groups**: Assign inter-group weights based on cross-group correlations.
4. **Compute total weights**: Each asset's final weight is its intra-group weight multiplied by its group's weight.

**Worked example:** For a three-asset portfolio (US Treasuries, S&P 500, NASDAQ), handcrafting produces far more stable weights than direct optimization and avoids extreme values.

**More complex portfolios:** For portfolios spanning multiple countries and sectors across equities and bonds, use hierarchical grouping and build weights bottom-up. This is more flexible and robust in practice.

### Summary

The chapter presents practical defenses against over-fitting in allocation: bootstrapping and handcrafting both preserve diversification and stability, avoiding the extreme weights of classical optimization. These methods apply equally to allocating across trading rules and across actual assets.

## The Framework: Components

Using the book's Table 15 example, a small trading system contains two trading rules (A and B), each with variations (A1, A2, B1), and two instruments (X and Y). The system decomposes into these components:

1. **Trading rules and variations:**
   - Each rule variation generates a price forecast for a specific instrument — e.g. A1's forecast for instrument X, A2's forecast for X, B1's forecast for X, and so on.

2. **Forecasts:**
   - Each forecast expresses the expected price move for that instrument.

3. **Combined forecast:**
   - Multiple forecasts are merged into a single forecast per instrument, via weighted averaging.

4. **Volatility target:**
   - Set the level of risk you are willing to take (e.g. average daily loss). This keeps risk controlled and consistent with the overall strategy.

5. **Position sizing:**
   - Compute the position size for each instrument from the combined forecast and the volatility target.

6. **Subsystem positions:**
   - For each instrument, determine its position within its own trading subsystem.

7. **Instrument weights:**
   - Allocate capital to each trading subsystem (each instrument) according to a weighting scheme.

8. **Portfolio weighted positions:**
   - Combine the subsystem positions into the overall portfolio position.

**Example rule specification (semi-automatic style):**

| Item | Rule |
| ---- | ---- |
| Entry | Buy when the 20-day moving average crosses above the 40-day moving average, and vice versa. |
| Exit | Reverse when the entry condition breaks: if long, sell and go short when the 20-day MA crosses below the 40-day MA. |
| Position size | No more than 10 Eurodollar futures contracts, 1 FTSE 100 futures contract, or a £10-per-point CFD per trade. |
| Money management | Risk no more than 3% of total capital per trade. |
| Stop loss | Trailing stop; close the position when the loss reaches 3% of total capital. If stops trigger too frequently, widen them. |

## Forecasts

Forecasts are the foundation of a trading system. This chapter examines how to generate and use forecasts to drive the system — essential whether you are a staunch systems trader, a semi-automatic trader, or an asset allocating investor.

### 1. What Makes a Good Forecast

**A forecast is a number.**
A forecast is an estimate of an asset's coming price move. Positive means you expect the price to rise; negative means you expect it to fall. Forecasts should not be binary (buy/sell) but scalar: values near zero indicate a small expected move, larger absolute values a larger expected move.

**Forecasts should be proportional to risk-adjusted expected return.**
Example: if the Bund has an expected annual return of 2% with 8% annualized standard deviation, and the Schatz has an expected return of 1% with only 2% standard deviation, then on a risk-adjusted basis the Schatz (1% ÷ 2% = 0.5) is twice as attractive as the Bund (2% ÷ 8% = 0.25) — so the Schatz forecast should be twice the Bund forecast.

**Forecasts should be consistently scaled.**
Scale forecasts so their expected absolute value is 10. Then +10 means an average-strength buy and -10 an average-strength sell; values near zero indicate weak signals, while +20 or -20 indicate very strong ones. Semi-automatic traders must quantify the strength of their conviction on this scale; asset allocating investors use a fixed forecast value.

**Forecasts should be capped.**
Cap forecasts at a maximum absolute value — Carver recommends ±20. Reasons:

- **Risk control**: Limits the risk taken by any single trading rule.
- **Limited data**: Very large forecast values are rare in historical data, so their reliability is low.
- **Extremes are different**: Market behavior in extreme conditions may not resemble normal conditions.
- **Higher realized volatility**: A large forecast may promise a large return, but often comes with higher volatility too.
- **Limited downside**: Since most forecasts are small anyway, capping has little impact on total returns.

Systems traders should clip all out-of-range forecasts to the [-20, +20] range. Semi-automatic traders should keep all their forecasts within ±20. Asset allocating investors use a fixed forecast of +10.

### 2. Example Trading Rules

Two simple systematic rule examples:

1. **Trend-following rule**
   - Use a set of moving average crossovers to identify trends.
   - E.g. generate a positive forecast when the short MA is above the long MA.

2. **Value rule**
   - Generate forecasts from valuation metrics such as P/E.
   - E.g. generate a positive forecast when a stock's P/E is below its historical average.

### 3. Other People's Trading Rules

You can use publicly available trading rules or invent your own. Either way, standardize them so the forecasts they produce share a consistent scale.

### 4. Selecting Trading Rules and Variations

Apply the selection process from the earlier chapters, making sure the rules you choose work across different markets and time frames.

### Summary

Generating and using forecasts is the core of building a trading system. Whether you are a systems trader, semi-automatic trader, or asset allocating investor, standardizing and capping forecast values is what allows the system to operate consistently across market conditions.
