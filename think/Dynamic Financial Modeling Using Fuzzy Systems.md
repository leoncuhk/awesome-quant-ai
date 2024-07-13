

# Dynamical Models of Stock Prices Based on Technical Trading Rules Part I



## Introduction 

Understanding the dynamics of stock prices is a significant challenge, approached through random walk models, agent-based models, and technical analysis. This paper leverages fuzzy systems theory to transform technical trading rules into excess demand functions, driving price dynamics.

### Key Concepts 

- **Fuzzy Systems Theory** : Used to model technical trading rules expressed in natural language.

- **Excess Demand Functions** : Derived from fuzzy systems to drive stock price dynamics.

- **Price Dynamics** : Explores how various technical trading heuristics create complex and chaotic price movements.

### Price Dynamics Equation 
The general form of the price dynamics model is given by:
$$P_{t+1} = P_t + \sum_{i=1}^M a_i D_i(\mathbf{X}_t)$$
where $$P_t$$ is the stock price at time $$t$$, $$M$$ is the number of trader groups, $$a_i$$ represents the strength of traders in group $$i$$, $$D_i$$ is the excess demand function, and $$\mathbf{X}_t$$ includes variables computed from past prices and other information.



## Moving Average Rules 

### Heuristic 

- **Buy Signal** : When a shorter moving average crosses a longer moving average from below.

- **Sell Signal** : When a shorter moving average crosses a longer moving average from above.

### Fuzzy Sets and Rules 

- **Fuzzy Sets** : Define linguistic terms like "Positive Small (PS)," "Positive Medium (PM)," etc.

- **Fuzzy IF-THEN Rules** : Transform trading heuristics into fuzzy rules: 
  - Rule 1: IF $$\Delta MA$$ is PS, THEN demand is Buy Small (BS).

  - Rule 2: IF $$\Delta MA$$ is PM, THEN demand is Buy Big (BB).

  - Rule 3: IF $$\Delta MA$$ is PL, THEN demand is Sell Medium (SM).

  - Rule 4: IF $$\Delta MA$$ is NS, THEN demand is Sell Small (SS).

  - Rule 5: IF $$\Delta MA$$ is NM, THEN demand is Sell Big (SB).

  - Rule 6: IF $$\Delta MA$$ is NL, THEN demand is Buy Medium (BM).

  - Rule 7: IF $$\Delta MA$$ is AZ, THEN demand is Neutral (N).

### Price Dynamics Model 
The fuzzy system combining the rules is given by:
$$D_1 = \frac{\sum_{j=1}^7 \mu_j(\Delta MA) c_j}{\sum_{j=1}^7 \mu_j(\Delta MA)}$$
where $$\mu_j$$ are the membership functions, and $$c_j$$ are the centers of the fuzzy sets.
### Simulation Results 
The price dynamic equation incorporating the moving average rules is:
$$P_{t+1} = P_t + a_1 D_1$$
Simulation results show complex and chaotic price behavior driven by the moving average rules.



## Support and Resistance Rules 

### Heuristic 

- **Buy Signal** : When the current price breaks above a resistance point.

- **Sell Signal** : When the current price breaks below a support point.

### Definitions 

- **Resistance Point**  $$R_t$$: Highest peak in the interval $$[t-n, t-1]$$.

- **Support Point**  $$S_t$$: Lowest trough in the interval $$[t-n, t-1]$$.

### Fuzzy Sets and Rules 

- **Fuzzy Sets** : Define terms for price changes relative to support and resistance points.

- **Fuzzy IF-THEN Rules** : 
  - Rule 1: IF $$\Delta R$$ is PS, THEN demand is Buy Small (BS).

  - Rule 2: IF $$\Delta R$$ is PM, THEN demand is Buy Big (BB).

  - Rule 3: IF $$\Delta R$$ is PL, THEN demand is Sell Medium (SM).

  - Rule 4: IF $$\Delta S$$ is NS, THEN demand is Sell Small (SS).

  - Rule 5: IF $$\Delta S$$ is NM, THEN demand is Sell Big (SB).

  - Rule 6: IF $$\Delta S$$ is NL, THEN demand is Buy Medium (BM).

### Price Dynamics Model 
The fuzzy system for support and resistance rules is:
$$D_2 = \frac{\sum_{j=1}^6 \mu_j(\Delta R, \Delta S) c_j}{\sum_{j=1}^6 \mu_j(\Delta R, \Delta S)}$$
where $$\mu_j$$ are the membership functions, and $$c_j$$ are the centers of the fuzzy sets.
### Simulation Results 
The price dynamic equation incorporating support and resistance rules is:
$$P_{t+1} = P_t + a_2 D_2$$
Simulations illustrate price jumps when crossing support or resistance lines.



## Trend Line Rules 

### Heuristic 

- **Uptrend Line** : Buy if the price approaches from above.

- **Downtrend Line** : Sell if the price approaches from below.

### Definitions 

- **Uptrend Line**  $$U_t$$: Line connecting two lowest troughs.

- **Downtrend Line**  $$D_t$$: Line connecting two highest peaks.

### Fuzzy Sets and Rules 

- **Fuzzy Sets** : Define terms for price relative to trend lines.

- **Fuzzy IF-THEN Rules** : 
  - Rule 1: IF $$\Delta U$$ is PS, THEN demand is Buy Medium (BM).

  - Rule 2: IF $$\Delta D$$ is NS, THEN demand is Sell Medium (SM).

### Price Dynamics Model 
The fuzzy system for trend line rules is:
$$D_4 = \frac{\sum_{j=1}^2 \mu_j(\Delta U, \Delta D) c_j}{\sum_{j=1}^2 \mu_j(\Delta U, \Delta D)}$$
where $$\mu_j$$ are the membership functions, and $$c_j$$ are the centers of the fuzzy sets.
### Simulation Results 
The price dynamic equation incorporating trend line rules is:
$$P_{t+1} = P_t + a_4 D_4$$
Simulations show self-fulfilling trends and reversals.



## Big Buyer, Big Seller, and Manipulator Rules 

### Heuristics 

- **Big Sellers** : Sell if the price is increasing.

- **Big Buyers** : Buy if the price is decreasing.

- **Manipulators** : Use strategies like pump-and-dump.

### Fuzzy Sets and Rules 

- **Fuzzy Sets** : For price changes and trading actions.

- **Fuzzy IF-THEN Rules** : 
  - Big Seller Rules: 
    - Rule 1: IF $$\Delta P$$ is PS, THEN demand is Sell Small (SS).

    - Rule 2: IF $$\Delta P$$ is PM, THEN demand is Sell Medium (SM).

    - Rule 3: IF $$\Delta P$$ is PL, THEN demand is Sell Big (SB).

    - Rule 4: IF $$\Delta P$$ is AZ, THEN demand is Neutral (N).

  - Big Buyer Rules: 
    - Rule 1: IF $$\Delta P$$ is NS, THEN demand is Buy Small (BS).

    - Rule 2: IF $$\Delta P$$ is NM, THEN demand is Buy Medium (BM).

    - Rule 3: IF $$\Delta P$$ is NL, THEN demand is Buy Big (BB).

    - Rule 4: IF $$\Delta P$$ is AZ, THEN demand is Neutral (N).

### Price Dynamics Model 
The fuzzy systems for big buyers and sellers are:
$$D_6 = \frac{\sum_{j=1}^4 \mu_j(\Delta P) c_j}{\sum_{j=1}^4 \mu_j(\Delta P)}$$
$$D_7 = \frac{\sum_{j=1}^4 \mu_j(\Delta P) c_j}{\sum_{j=1}^4 \mu_j(\Delta P)}$$
where $$\mu_j$$ are the membership functions, and $$c_j$$ are the centers of the fuzzy sets.
### Simulation Results 
The price dynamic equations incorporating big buyers and sellers are:
$$P_{t+1} = P_t + a_6 D_6$$
$$P_{t+1} = P_t + a_7 D_7$$
Simulations illustrate the influence of large traders and manipulators.



## Band and Stop Rules 

### Heuristic 

- **Bands** : Buy or sell when the price breaks out of the band.

- **Stops** : Use protective and trailing stops to manage risk.

### Definitions 

- **Bollinger Bands** : Upper and lower boundaries around a moving average.

- **Protective Stop** : Set limits to maximum losses.

- **Trailing Stop** : Protect profits from deteriorating.

### Fuzzy Sets and Rules 

- **Fuzzy Sets** : Define terms for price relative to bands and stops.

- **Fuzzy IF-THEN Rules** : 
  - Band Rules: 
    - Rule 1: IF $$\Delta B$$ is PS, THEN demand is Buy Small (BS).

    - Rule 2: IF $$\Delta B$$ is PM, THEN demand is Buy Big (BB).

    - Rule 3: IF $$\Delta B$$ is NS, THEN demand is Sell Small (SS).

    - Rule 4: IF $$\Delta B$$ is NM, THEN demand is Sell Big (SB).

  - Stop Rules: 
    - Rule 1: IF $$\Delta P$$ is NL and $$\Delta P_{\text{max}}$$ is NL, THEN demand is Sell Big (SB).

    - Rule 2: IF $$\Delta P$$ is P and $$\Delta P_{\text{max}}$$ is NL, THEN demand is Sell Big (SB).

### Price Dynamics Model 
The fuzzy systems for bands and stops are:
$$D_9 = \frac{\sum_{j=1}^4 \mu_j(\Delta B) c_j}{\sum_{j=1}^4 \mu_j(\Delta B)}$$
$$D_{10} = \frac{\sum_{j=1}^2 \mu_j(\Delta P, \Delta P_{\text{max}}) c_j}{\sum_{j=1}^2 \mu_j(\Delta P, \Delta P_{\text{max}})}$$
where $$\mu_j$$ are the membership functions, and $$c_j$$ are the centers of the fuzzy sets.
### Simulation Results 
The price dynamic equations incorporating band and stop rules are:
$$P_{t+1} = P_t + a_9 D_9$$
$$P_{t+1} = P_t + a_{10} D_{10}$$
Simulations show trends driven by band breakouts and risk management.



## Volume and Strength Rules 

### Heuristic 

- **Volume** : Consider on-balance volume trends for trading decisions.

- **Relative Strength** : Compare stock performance to the market.

### Definitions 

- **On-Balance Volume (OBV)** : Accumulated volume indicator.

- **Relative Strength (RS)** : Ratio of stock performance to market index.

### Fuzzy Sets and Rules 

- **Fuzzy Sets** : For volume and strength indicators.

- **Fuzzy IF-THEN Rules** : 
  - Volume Rules: 
    - Rule 1: IF $$\Delta P$$ is NS and $$\Delta OBV$$ is N, THEN demand is Sell Medium (SM).

    - Rule 2: IF $$\Delta P$$ is PS and $$\Delta OBV$$ is P, THEN demand is Buy Medium (BM).

  - Strength Rules: 
    - Rule 1: IF $$\Delta P$$ is NS and $$\Delta RS$$ is N, THEN demand is Sell Medium (SM).

    - Rule 2: IF $$\Delta P$$ is PS and $$\Delta RS$$ is P, THEN demand is Buy Medium (BM).

### Price Dynamics Model 
The fuzzy systems for volume and strength are:
$$D_{11} = \frac{\sum_{j=1}^2 \mu_j(\Delta P, \Delta OBV) c_j}{\sum_{j=1}^2 \mu_j(\Delta P, \Delta OBV)}$$
$$D_{12} = \frac{\sum_{j=1}^2 \mu_j(\Delta P, \Delta RS) c_j}{\sum_{j=1}^2 \mu_j(\Delta P, \Delta RS)}$$
where $$\mu_j$$ are the membership functions, and $$c_j$$ are the centers of the fuzzy sets.
### Simulation Results 

Due to the lack of volume data, simulations for Rule 11 were not performed. For real stocks, these rules can be incorporated into the price dynamic model to study their effects.



## Concluding Remarks 

The paper builds a bridge between technical analysis and nonlinear dynamic equations through fuzzy systems theory. The resulting models demonstrate complex price dynamics driven by technical trading rules. Future work includes detailed analysis and trading strategy development based on these models.

These models provide a framework to detect hidden market operations and develop trading strategies to outperform the benchmark Buy-and-Hold strategy. Future research will further analyze the properties of these models and test trading strategies in real market conditions.



---

# Dynamical Models of Stock Prices Based on Technical Trading Rules Part II



## Introduction 

Part II of the series extends the analysis of the price dynamical model based on moving average rules from Part I. This paper explores how the interplay between trend-following and contrarian actions generates price chaos. Key findings include the instability of all equilibrium points, short-term predictability of price volatility, and the derivation of the Lyapunov exponent. The study also examines return correlations and the fat-tailed distribution of returns.

### Key Concepts 

- **Equilibrium** : Infinite and unstable equilibriums in the price dynamical model.

- **Volatility** : Fixed function of model parameters, revealing causes of phenomena like volatility clustering.

- **Return Predictability** : Short-term predictability characterized by the Lyapunov exponent.

- **Return Independence** : Analysis of return correlations changing from positive to negative.



## Chaos Generation 

### Model Description 
The model is driven by Heuristic 1 (Rule-1-Group) from Part I:
$$P_{t+1} = P_t + a_1 D_1(\mathbf{X}_t)$$
where:
$$\Delta MA = \log \left( \frac{MA_m}{MA_n} \right), \quad D_1 = f(\Delta MA)$$
$$MA_m$$ and $$MA_n$$ are moving averages of lengths $$m$$ and $$n$$ respectively, with $$m < n$$.
### Fuzzy System 
The fuzzy system $$f$$ constructed from the seven fuzzy IF-THEN rules is:
$$f(\Delta MA) = \sum_{i=1}^7 \mu_i(\Delta MA) c_i$$
where $$\mu_i$$ are the membership functions and $$c_i$$ are the centers of the fuzzy sets.
### Chaos Analysis 
The interplay between trend-followers and contrarians is analyzed by varying the strength parameter $$a_1$$: 
- **Convergence** : Small $$a_1$$ leads to a convergent price series.

- **Chaos** : Intermediate $$a_1$$ results in chaotic price behavior.

- **Oscillation** : Large $$a_1$$ causes oscillatory price movements.

### Simulation Results 

- **Figure 1** : Excess demand function $$f(\Delta MA)$$.

- **Figure 2** : Price trajectories for different $$a_1$$ values.

- **Figure 3** : Parameter ranges for convergence, chaos, and oscillation.



## Equilibrium Analysis 

### Definition 
An equilibrium point $$\mathbf{y}^*$$ satisfies:
$$\mathbf{y}_{t+1} = \mathbf{y}_t = \mathbf{y}^*$$
### Theorems 

- **Theorem 1** : Infinite equilibriums $$\mathbf{y}^* = k \mathbf{1}$$ for any positive $$k$$.

- **Theorem 2** : All equilibriums are unstable.

### Proofs 

- **Proof of Theorem 1** : Derived from the fixed points of the fuzzy system.

- **Proof of Theorem 2** : Using the Jacobian matrix and the Linearized Stability Theorem.

### Implications 

The instability of equilibriums suggests that the price series will not settle at a single point but may exhibit complex dynamics.



## Short-term Predictability 

### Chaotic vs. Random 

- **Chaos** : Small changes in initial conditions lead to predictable short-term behavior.

- **Random** : Immediate transition to steady-state behavior.

### Monte Carlo Simulations 

- **Figure 6** : Return trajectories for different initial volatilities.

- **Figure 7** : Corresponding volatilities.

### Volatility Definition 
$$v(t) = \sqrt{\frac{1}{S} \sum_{j=1}^S (r_j(t) - \overline{r(t)})^2}$$
where $$S$$ is the number of simulations.



## Lyapunov Exponent 

### Definition 
Quantifies sensitivity to initial conditions:
$$L = \lim_{t \to \infty} \frac{1}{t} \log \left( \frac{d(t)}{d(0)} \right)$$
### Calculation 

- **Figure 8** : Volatilities plotted in log-t scale.

- **Figure 9** : Volatilities for different $$a_1$$ values.

### Lemma 1 
Approximate formula for Lyapunov exponent:
$$L \approx \frac{1}{n} \log \left( 1 + a_1 \sum_{i=1}^n \left| c_i \right| \right)$$



## Volatility Convergence 

### Observation 

Volatility converges to a constant dependent on model parameters.

### Simulations 

- **Figure 10** : Converged volatility as a function of $$a_1$$.

- **Figure 11** : Converged volatility as a function of $$w$$.

### Result 1 
Approximate formula for converged volatility:
$$\sigma^2 \approx A \sin(B a_1) + C$$
### Lemma 2 
Formula for volatility in oscillation mode:
$$\sigma = \sqrt{2} w a_1$$



## Return Correlations 

### Drift Definition 
$$d(t) = \sqrt{\frac{1}{S} \sum_{j=1}^S (r_j(t) - \overline{r(t)})^2}$$
### Simulations 

- **Figure 12** : Drift for different $$a_1$$ values.

- **Figure 13** : Distance-to-uncorrelated as a function of $$a_1$$.

- **Figure 14** : Distance-to-uncorrelated as a function of $$w$$.

### Result 2 
Linear relation for uncorrelated returns:
$$a_1 \approx k w$$
### Auto-correlations 

- **Figure 15** : Auto-correlations for different $$a_1$$ values.



## Strange Attractor and Fat-tailed Distribution 

### Strange Attractor 

- **Figure 16** : Phase portrait of returns.

### Fat-tailed Distribution 

- **Figure 17** : Return distribution compared to Gaussian.



## Concluding Remarks 

The deterministic price dynamical models reveal insights into equilibrium, volatility, and return predictability in financial markets. The findings challenge classical concepts and suggest new stability definitions for social systems.

### Key Insights 

1. **Equilibrium** : Unstable equilibriums and set-stability.

2. **Volatility** : Deterministic function of model parameters.

3. **Predictability** : Short-term predictability defined by the Lyapunov exponent.

4. **Return Correlation** : Richer framework for analyzing return correlations compared to random walk models.



---

# Dynamical Models of Stock Prices Based on Technical Trading Rules Part III



## Introduction 

Part III of this study applies the price dynamical model with big buyers and big sellers, developed in Part I, to the daily closing prices of the top 20 banking and real estate stocks listed on the Hong Kong Stock Exchange. The study estimates the strength parameters of the big buyers and sellers to devise buy/sell decisions, proposing two trading strategies: Follow-the-Big-Buyer (FollowBB) and Ride-the-Mood (RideMood). The results demonstrate significant improvements over the benchmark Buy-and-Hold strategy.

### Key Concepts 

- **Big Buyers and Sellers** : Institutional investors who buy or sell stocks in large quantities.

- **Trading Strategies** : Methods to leverage the estimated strength parameters of big buyers and sellers for profitable trading.



## Price Dynamic Model and Trading Strategies 

### Price Dynamic Model 
Using fuzzy systems theory, the Big Buyer and Big Seller Heuristics are transformed into a price dynamical model:
$$P_{t+1} = P_t + \alpha (D_7 - D_6)$$
where: 
- $$P_t$$ is the price of the stock at time $$t$$.

- $$D_6$$ and $$D_7$$ are the excess demand functions for big sellers and big buyers, respectively.

- $$\alpha$$ is the price impact factor.

### Excess Demand Functions 
The excess demand functions $$D_6$$ and $$D_7$$ are derived from fuzzy systems:
$$D_6 = f(\Delta MA)$$
$$D_7 = g(\Delta MA)$$
where $$\Delta MA$$ is the log-ratio of the price to its moving average:
$$\Delta MA = \log \left( \frac{P_t}{MA_n} \right)$$
### Trading Strategies 

#### Follow-the-Big-Buyer (FollowBB) 

- **Buy** : When big buyer is detected ($$D_7 > 0$$) and no big seller is detected ($$D_6 \leq 0$$).

- **Sell** : When the big buyer stops buying ($$D_7 \leq 0$$).

#### Ride-the-Mood (RideMood) 

- **Buy** : When the strength of big buyers surpasses that of big sellers ($$D_7 - D_6 > 0$$).

- **Sell** : When the strength of big sellers surpasses that of big buyers ($$D_7 - D_6 \leq 0$$).



## Parameter Estimation Algorithm 

### Recursive Least Squares Algorithm 
To estimate the strength parameters of big buyers and sellers, the Recursive Least Squares Algorithm with Exponential Forgetting is used:
$$\hat{\theta}_{t+1} = \hat{\theta}_t + K_t (y_t - \phi_t^T \hat{\theta}_t)$$
$$K_t = \frac{P_t \phi_t}{\lambda + \phi_t^T P_t \phi_t}$$
$$P_{t+1} = \frac{1}{\lambda} (P_t - K_t \phi_t^T P_t)$$
where: 
- $$\hat{\theta}_t$$ are the parameter estimates.

- $$y_t$$ is the observed data.

- $$\phi_t$$ is the regressor vector.

- $$\lambda$$ is the forgetting factor.

### Simulation Results 

Simulations demonstrate the effectiveness of the parameter estimation algorithm under various noise conditions, indicating its robustness in detecting the strength of big buyers and sellers.



## The Trading Strategies: FollowBB, RideMood, TrendFL, and Buy&Hold 

### Follow-the-Big-Buyer (FollowBB) 

- **Algorithm** :
  1. Detect big buyers.

  2. Buy when $$D_7 > 0$$ and $$D_6 \leq 0$$.

  3. Sell when $$D_7 \leq 0$$.

### Ride-the-Mood (RideMood) 

- **Algorithm** :
  1. Compare the strength of big buyers and sellers.

  2. Buy when $$D_7 - D_6 > 0$$.

  3. Sell when $$D_7 - D_6 \leq 0$$.

### Trend-Following (TrendFL) 

- **Algorithm** :
  1. Buy when a shorter moving average crosses a longer moving average from below.

  2. Sell when the shorter moving average crosses the longer moving average from above.

### Buy-and-Hold (Buy&Hold) 

- **Algorithm** :
  1. Buy the stock on the first day.

  2. Hold the stock until the last day.
  
  

## Application to Hong Kong Stocks 

### Data and Methodology 

The four trading strategies were applied to the top 20 banking and real estate stocks listed on the Hong Kong Stock Exchange over a seven-year period from July 3, 2007, to July 2, 2014. The daily closing prices were used for analysis.

### Results 

The performance of each strategy was evaluated based on annual returns, standard deviations, and Sharpe ratios. The FollowBB and RideMood strategies significantly outperformed the Buy&Hold and TrendFL strategies.



## Portfolio Performance 

### Portfolio Scheme 

A simple portfolio scheme was used, distributing initial money across the 20 stocks according to their weights in the Hang Seng Index (HSI). The trading strategies were applied independently to each stock.

### Results 

The FollowBB and RideMood strategies showed superior performance compared to the TrendFL and Buy&Hold strategies. The portfolio with the RideMood strategy had the best overall performance, followed by FollowBB.



## Details of Buy/Sell Cycles 

### Analysis of Buy/Sell Points 

Detailed analysis of the buy/sell points for the FollowBB and RideMood strategies over a three-year period from January 3, 2011, to December 31, 2013, revealed that these strategies could effectively detect major trends and avoid significant losses.

### Strengths and Weaknesses 

- **Strengths** : Major up-trends were detected and followed, leading to significant profits.

- **Weaknesses** : Some negative return buy/sell cycles were observed, indicating areas for improvement.



## Concluding Remarks 

The study demonstrates the effectiveness of the FollowBB and RideMood strategies in leveraging the strength of big buyers and sellers for profitable trading. These strategies outperformed traditional Buy&Hold and TrendFL strategies, providing a robust framework for trading in stock markets.

### Key Insights 

1. **Profitability** : FollowBB and RideMood strategies significantly increase net profits compared to Buy&Hold.

2. **Risk Management** : These strategies also reduce risk by avoiding major down-trends.

3. **Market Conditions** : The Hong Kong stock market provided a suitable environment for testing these strategies, given its characteristics and the presence of "hot money".



# Comments

### Evaluation of the Recursive Least Squares Algorithm for Parameter Estimation 

#### Scientific Basis of the Recursive Least Squares Algorithm 

The Recursive Least Squares (RLS) algorithm is a well-established method in the field of system identification and control theory. It is particularly valued for its ability to recursively update parameter estimates as new data becomes available, making it suitable for online estimation in dynamic systems. The algorithm's foundational equations are:
$$\hat{\theta}_{t+1} = \hat{\theta}_t + K_t (y_t - \phi_t^T \hat{\theta}_t)$$
$$K_t = \frac{P_t \phi_t}{\lambda + \phi_t^T P_t \phi_t}$$
$$P_{t+1} = \frac{1}{\lambda} (P_t - K_t \phi_t^T P_t)$$
where:

- $$\hat{\theta}_t$$ are the parameter estimates at time $$t$$.

- $$y_t$$ is the observed data (output).

- $$\phi_t$$ is the regressor vector (input).

- $$K_t$$ is the gain matrix.

- $$P_t$$ is the error covariance matrix.

- $$\lambda$$ is the forgetting factor.

These equations ensure that the algorithm adjusts the parameter estimates to minimize the weighted sum of squared prediction errors, with a preference for recent data due to the forgetting factor.

#### Advantages of RLS in the Context of the Study 

1. **Adaptability** : The RLS algorithm's ability to update estimates in real-time makes it particularly suitable for financial markets, where conditions can change rapidly. This adaptability allows the model to remain relevant even as market dynamics evolve.

2. **Efficiency** : Given that financial markets produce large volumes of data, the recursive nature of the RLS algorithm offers computational efficiency. It avoids the need to process the entire data set repeatedly, which would be computationally prohibitive.

3. **Robustness** : The forgetting factor $$\lambda$$ helps the model to give more weight to recent observations, which is critical in financial markets where older data may become less relevant over time.

#### Contributions to Research 

The application of the RLS algorithm to estimate the strength parameters of big buyers and sellers in stock markets represents a significant contribution to the field of financial modeling. Specifically:

1. **Dynamic Modeling** : By incorporating RLS, the study presents a dynamic modeling approach that can adjust to the continuous flow of new data, thereby capturing the ever-changing behavior of institutional investors.

2. **Detecting Market Movers** : The ability to identify big buyers and sellers in real-time provides a strategic advantage. This detection mechanism can inform trading strategies that leverage the presence of these market movers, enhancing the potential for profitability.

3. **Improved Trading Strategies** : The study's trading strategies (Follow-the-Big-Buyer and Ride-the-Mood) based on RLS-derived parameters outperform traditional strategies like Buy-and-Hold and Trend-Following. This demonstrates the practical utility of the approach in real-world trading.

#### Limitations and Considerations 

Despite its advantages, the RLS algorithm is not without limitations:

1. **Sensitivity to Noise** : Financial data is often noisy, and while RLS is robust, extreme market events (e.g., crashes or surges) can still lead to significant estimation errors.

2. **Parameter Selection** : The choice of the forgetting factor $$\lambda$$ and the initial values for $$P_t$$ and $$\hat{\theta}_t$$ can significantly impact the performance of the algorithm. Improper selection can lead to suboptimal parameter estimates.

3. **Model Assumptions** : The effectiveness of RLS relies on the underlying model being an adequate representation of the real system. If the model is misspecified, the parameter estimates and resulting trading decisions may be flawed.

#### Conclusion 

The use of the Recursive Least Squares algorithm in the context of the study is scientifically sound and aligns well with the requirements of dynamic financial modeling. The approach leverages the algorithm's strengths in real-time adaptability, computational efficiency, and robustness to provide valuable insights and improved trading strategies. The research contributes meaningfully to the understanding of market dynamics and the development of effective trading strategies, though attention must be paid to the choice of algorithm parameters and the potential impact of noise in financial data.



### Analysis, Evaluation, and Summary of the Follow-the-Big-Buyer (FollowBB) Strategy 

#### Strategy Overview 

The Follow-the-Big-Buyer (FollowBB) strategy is designed to capitalize on the presence of large institutional investors—referred to as "big buyers"—who significantly influence stock prices due to their substantial buying power. The strategy operates based on the following rules:

- **Buy** : Initiate a buy position when the strength of big buyers ($$D_7$$) is positive and the strength of big sellers ($$D_6$$) is zero or negative.

- **Hold** : Maintain the position as long as the big buyers are actively purchasing ($$D_7 > 0$$).

- **Sell** : Liquidate the position when the big buyers cease their activity ($$D_7 \leq 0$$).

#### Scientific Basis 

The strategy's scientific basis lies in the hypothesis that institutional investors have the power to drive stock prices due to their large trade volumes. By identifying and following these big buyers, smaller investors can potentially ride the price momentum generated by these large trades.

#### Key Assumptions 

1. **Market Influence** : Institutional investors significantly influence stock prices.

2. **Detectability** : The activity of big buyers can be detected through price dynamics and excess demand functions derived from the Recursive Least Squares (RLS) algorithm.

3. **Persistence** : Once detected, big buyers will continue to influence the stock price in the same direction for a sufficient period to allow profitable trading.

#### Advantages 

1. **Alignment with Market Movers** : By focusing on the actions of big buyers, the strategy aligns with market forces that have a tangible impact on stock prices, increasing the potential for profitable trades.

2. **Data-Driven Decisions** : The use of RLS for parameter estimation provides a robust and adaptive mechanism to detect big buyers in real-time, ensuring the strategy remains responsive to market changes.

3. **Reduced Risk** : The strategy includes clear sell signals to exit positions when the big buyers are no longer active, helping to manage and mitigate risk.

#### Performance Evaluation 

**Historical Performance** 

The strategy was tested on the top 20 banking and real estate stocks listed on the Hong Kong Stock Exchange over a seven-year period from July 3, 2007, to July 2, 2014. The evaluation criteria included annual returns, standard deviations, and Sharpe ratios.

- **Annual Returns** : FollowBB consistently delivered higher annual returns compared to the Buy-and-Hold and Trend-Following strategies. For example, HSBC Holdings (HK0005) showed a 4.78% return with FollowBB compared to -1.82% with Buy-and-Hold.

- **Standard Deviation** : The strategy demonstrated a manageable level of risk, with standard deviations comparable to or lower than other strategies.

- **Sharpe Ratio** : FollowBB achieved higher Sharpe ratios, indicating better risk-adjusted returns.

**Market Conditions** 

FollowBB performed well in various market conditions, including rising markets, declines, and periods of volatility. This robustness suggests that the strategy can adapt to different market environments, providing reliable signals across diverse scenarios.

#### Limitations and Challenges 

1. **Detection Lag** : While RLS is efficient, there can be a lag in detecting big buyers, potentially leading to missed opportunities or late entries.

2. **False Positives** : In highly volatile markets, the algorithm may generate false signals, leading to unprofitable trades.

3. **Dependency on Model Accuracy** : The effectiveness of the strategy hinges on the accuracy of the RLS parameter estimates. Inaccurate estimates can undermine the strategy's performance.

#### Summary 

The Follow-the-Big-Buyer (FollowBB) strategy presents a scientifically grounded approach to leveraging the influence of institutional investors in stock markets. By systematically identifying and following big buyers, the strategy aims to capture the price momentum created by these large trades, offering a higher potential for profit and better risk management compared to traditional strategies.

#### Key Takeaways 

1. **Effectiveness** : The strategy has shown to outperform traditional Buy-and-Hold and Trend-Following strategies in historical tests, particularly in terms of annual returns and Sharpe ratios.

2. **Adaptability** : FollowBB adapts to changing market conditions through real-time parameter estimation, maintaining relevance and accuracy in various scenarios.

3. **Risk Management** : The strategy includes clear entry and exit rules based on the presence of big buyers, helping to manage and mitigate risk effectively.

Overall, FollowBB represents a valuable contribution to trading strategy development, providing a method to systematically exploit the market influence of institutional investors.



References:

- https://arxiv.org/abs/1401.1888
