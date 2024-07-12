## Markov-Switching Regression Model in Regime Switching



Study the Thesis: Construction of Effective Regime-Switching Portfolios Using a Combination of Machine Learning and Traditional Approaches, by Piotr Pomorski



###Introduction

**Research Question**
The thesis investigates the detection and prediction of market states (also referred to as financial regimes or market conditions). Specifically, it focuses on using various statistical and machine learning methods to identify and forecast different market states (such as bull markets, bear markets, and transitional states) to improve asset allocation and investment decisions.

**Importance of the Research**
Studying the detection and prediction of market states is crucial for investors and asset managers because different market states require different investment strategies. Accurately identifying and forecasting market state changes can help investors optimize their portfolios, reduce risk, and enhance returns. Additionally, understanding market state changes can aid in formulating more scientific macroeconomic policies and preventing financial crises.

**Main Research Methods**
The thesis explores a range of traditional statistical and machine learning methods for market state detection and prediction, including:

1. **Markov-switching Regression Model (MSR)**
   - Widely applied and effective in detecting market state transitions.
   - Validated across various financial markets and asset classes.
   - Main limitation is its lag in prediction.

2. **Generalized Autoregressive Conditional Heteroskedasticity (GARCH) and Extensions**
   - Used for estimating market volatility and indirectly detecting market states.
   - Limitations include failure to accurately capture structural breaks and overestimating volatility.
   - MS-GARCH attempts to combine the strengths of MSR and GARCH but underperforms in long-term volatility prediction.

3. **Change Point Detection Methods**
   - Focus on identifying specific time points where structural changes occur in the data.
   - Provide precise information about when market state changes occur but are less flexible than MSR in handling continuous transitions.

4. **Machine Learning Methods**
   - **Random Forest (RF)**: Performs exceptionally well in various studies, especially in predicting market states, surpassing traditional methods and other machine learning algorithms.
   - **Hidden Markov Model (HMM)**: Effective in predicting market states but sometimes outperformed by other machine learning algorithms.
   - **Support Vector Machine (SVM)**: Successful in predicting stock price direction but inferior to Random Forest in multi-state prediction tasks.
   - **Artificial Neural Networks (ANN)**: Widely used for predicting asset prices and market recessions but less effective than Random Forest in predicting market states.

**Comparative Analysis of Methods**
The author compares and analyzes these methods on several levels:

1. **Accuracy**
   - Machine learning algorithms (e.g., Random Forest and HMM) generally exhibit higher accuracy in market state prediction compared to traditional statistical methods.
   - Random Forest is particularly effective in long-term predictions and multi-state classification tasks.

2. **Complexity and Computational Efficiency**
   - Traditional methods (e.g., MSR and GARCH) are simpler but inadequate for handling high-dimensional data and complex non-linear relationships.
   - Machine learning methods (e.g., RF and SVM) have higher computational complexity but excel in processing large-scale data and capturing intricate patterns.

3. **Applicability**
   - MSR and GARCH are applicable to various financial markets and asset classes but lag in real-time market state prediction.
   - Machine learning methods are highly flexible, adaptable to diverse data types and problem settings, but require substantial data and computational resources.

4. **Robustness**
   - Random Forest is superior in handling noise and outliers, demonstrating greater robustness compared to other machine learning algorithms.

Through this comparative analysis, the thesis ultimately selects Random Forest as the primary tool for market state prediction, explored in detail in Chapter 4. HMM is used as a benchmark for comparison. The goal is to find a more accurate and efficient market state prediction method to support investment decisions in financial markets.



### Background and Related Work

**Financial Background**
- Overview of asset allocation and the significance of diversification.
- Introduction to Modern Portfolio Theory (MPT) and its limitations in dynamic market conditions.

**Asset Allocation**
- Examination of traditional methods and the necessity for dynamic approaches.
- Review of the portfolio construction process, including the application of technical analysis.

**Modern Portfolio Theory (MPT) and Its Extensions**
- Explanation of MPT's core concepts.
- Discussion on extensions to MPT to address its limitations, such as incorporating alternative assets and dynamic strategies.

**Portfolio Construction Process**
- Steps involved in constructing a robust investment portfolio.
- Importance of asset selection, risk management, and periodic rebalancing.

**Technical Analysis**
- Historical context and basic principles of technical analysis.
- Common technical indicators and their application in market prediction.

**Regime Switches**

- Definition and significance of regime switches in financial markets.
- Impact of regime changes on asset returns and risk profiles.

**Technical Background**

**Markov-switching Regression Model**
- Detailed explanation of the Markov-switching regression model (MSR).
- Application of MSR in detecting regime changes in financial time series.

**Kaufman’s Adaptive Moving Average (KAMA)**
- Introduction to KAMA and its advantages over traditional moving averages.
- Explanation of how KAMA adapts to market conditions by incorporating trend and volatility factors.

**Limitations of the Markov-switching Model**
- Discussion on the challenges and limitations of using MSR for real-time regime detection.
- Issues related to lag in detection and the need for predictive capabilities.

**Machine Learning Models for Regime Prediction**
- Overview of machine learning models used for predicting market regimes.
- Focus on Hidden Markov Model (HMM) and Random Forest (RF) as primary methods.
- Comparison of these models with traditional approaches.

**Related Work**

**Regime Detection Methods**
- Examination of various methods used for detecting market regimes.
- In-depth analysis of Markov-switching models, GARCH models, and change point detection techniques.

**Regime Prediction Methods**
- Review of traditional and machine learning methods for predicting market regimes.
- Discussion on Probit/Logit models, Kalman filters, and advanced machine learning techniques like SVM and ANN.

**Portfolio Allocation Methods Related to Regime Switching**
- Exploration of how regime detection and prediction can enhance portfolio allocation.
- Application of MSR, HMM, and GARCH models in constructing regime-switching portfolios.
- Analysis of multi-period optimization and model predictive control for dynamic asset allocation.

**Value of Technical Analysis in Building Portfolios**
- Historical debate on the efficacy of technical analysis in portfolio management.
- Modern perspectives showing the added value of technical analysis, particularly with machine learning integration.
- Discussion on the use of technical indicators like moving averages and their limitations.





### Improving on the Markov-switching Regression Model by the Use of an Adaptive Moving Average

#### Background

**Problem with Two-State Markov-Switching Regression Models:**
- The two-state Markov-switching regression model is commonly used in financial applications for regime detection. However, it has limitations:
  - The transition from low- to high-volatility regimes is often smooth, resulting in medium variance periods.
  - Volatility is not a definite indicator of market direction; markets can rise during high volatility and fall during low volatility.
  - This model fails to accurately capture these dynamics, leading to frequent and potentially spurious regime switches.

**Proposed Solutions:**
- Adding a third state to detect medium volatility regimes (Boldin, 1996; Kim, Nelson, & Startz, 1998).
- However, three-state models can be unstable and lead to overfitting.

**Alternative Approaches:**
- Combining the two-state Markov model with technical analysis tools like moving averages (Srivastava & Bhattacharyya, 2018).
  - Example: The WorldQuant (WQ) Model, which blends the two-state Markov model with Keltner Channels, divides the market into four states (advance, distribution, decline, and accumulation).

**New Approach:**
- The study proposes replacing the Keltner Channels with Kaufman’s Adaptive Moving Average (KAMA) to enhance model stability and applicability across various asset classes.



#### Data and Methodology

**Data:**
- Daily closing prices for 56 assets across four classes: equities (24), exchange rates (13), commodities (12), and fixed income (7).
- Data split: 85% for training and validation, 15% for out-of-sample testing.
- High and low prices are required for the WQ Model.

**Methodology:**
1. **Integration of KAMA with MSR:**
   - The model starts with a two-state MSR to detect high and low variance periods.
   - KAMA is overlaid to divide these periods into bullish and bearish regimes, creating a four-regime model:
     - Low variance/bullish
     - Low variance/bearish
     - High variance/bullish
     - High variance/bearish

2. **Optimization of KAMA:**
   - KAMA parameters are optimized using a custom misclassification score function and K-Means clustering.
   - The optimized parameters are used in the test period to evaluate the model's performance.

3. **Benchmark Models:**
   - The proposed model is compared against various Markov-switching regression models and the WQ Model.
   - Models include two-state MSR, three-state MSR, three-state MSR turned into two states, three-state KNS, and three-state KNS turned into two states.

4. **Trading Strategy:**
   - Weights are assigned to two assets (selected asset vs. USD Cash 3-Month Rebalancing).
   - Optimization involves calculating returns and adjusted Sharpe ratio (ASR) for each segment.
   - Trading costs are considered for each asset class.
   
   

#### Results and Discussion

**Phase 1: In-Sample Testing:**
- The proposed KAMA+MSR model showed strong performance across different asset classes.
- Key findings:
  - Equities: KAMA+MSR achieved the highest adjusted Sharpe ratio and annualized returns.
  - Commodities: The proposed model was the most profitable but also carried significant risk.
  - FX: Mixed results; KAMA+MSR was stable but not always the top performer.
  - Fixed Income: Two-state MSR model performed best, but KAMA+MSR was also strong.

**Phase 2: Out-of-Sample Testing:**
- The three best models from Phase 1 were tested on the holdout sample.
- Key findings:
  - Equities: KAMA+MSR outperformed other models in both ASR and annualized returns.
  - Commodities: KAMA+MSR was the most profitable.
  - FX: Results were mixed; KAMA+MSR showed stable performance.
  - Fixed Income: KAMA+MSR performed best in ASR and was strong in annualized returns.

**Discussion:**
- The proposed model demonstrates robustness and high performance, especially in equities.
- **Equities:** KAMA+MSR provided a good balance between returns and volatility.
- **Commodities:** The proposed model was profitable but risky; a more risk-averse trader might prefer the three-state MSR turned into two states.
- **FX:** Mixed performance; other regime-detection methods may be needed for more robust results.
- **Fixed Income:** KAMA+MSR outperformed other models despite initial limitations.



**Summary:**
- The chapter presents an enhancement of the two-state MSR model with KAMA, aiming to detect both transitory and persistent regimes accurately and smoothly.
- The KAMA+MSR model outperformed comparison models on average across all asset classes, particularly excelling in equities.

**Future Work:**
- Investigate alternative methods for FX and commodities to improve performance.
- Consider more complex trading strategies, such as derivatives-based hedging, to fully leverage the detected transitory regimes.
- Further explore the predictive capability of the proposed model for future regimes.



### Predicting Financial Regimes by the Use of the Random Forest-KAMA+MSR Framework

#### Background

**Fractional Differencing**:
- Traditional methods to convert non-stationary time series data to stationary data (like differencing) erase the memory of the time series.
- Fractional differencing is introduced as a solution that allows the data to retain memory while achieving stationarity.
- This method helps preserve the distinguishing features of different economic events, crucial for effective regime prediction.

**Feature Selection**:
- The chapter incorporates feature selection to enhance the accuracy of the Random Forest model.
- Random Forest's built-in feature importance methods (such as Mean Decrease Impurity) have limitations, including bias towards continuous or high-cardinality categorical variables.
- The BorutaShap package is employed for feature selection, leveraging Shapley values for a more accurate determination of feature importance.

**Candidate Features**:
- The features used for prediction include technical indicators and fundamental/macroeconomic variables.
- Technical indicators are particularly useful for high-frequency data, while macroeconomic variables are more suited for low-frequency data.
- Different asset classes (equities, commodities, foreign exchange) utilize specific sets of features relevant to their unique characteristics.



#### Data and Methodology

**Data**:
- The data covers three major asset classes: equities, commodities, and foreign exchange pairs.
- Specific assets within these classes are selected based on data availability and relevance, with benchmarks chosen for out-of-sample model performance comparison.

**Data Preparation**:
- **Feature Engineering**: Involves creating technical indicators and statistical features from High-Low-Close-Volume data using Python packages like TA and tsfresh.
- **Fractional Differencing**: The data is scaled using fractional differencing to achieve stationarity while preserving memory.
- **Label Generation and Prediction**: The regime labels are generated based on the KAMA+MSR model from Chapter 3, with adjustments for trading costs and market conditions.

**Modelling**:
- **Cross-Validation Setup**: Utilizes the Purged Group Time-Series Split (PGTS) method to avoid data leakage and ensure models are trained on past data and validated on future data.
- **Hyperparameter Tuning**: Uses the Optuna package to optimize Random Forest hyperparameters to maximize the Sortino ratio.
- **Feature Selection**: Implements the BorutaShap algorithm to select the most predictive features, ensuring consistency over time.

**Back-Testing Strategy**:
- The models' predictions are used to establish a contrarian trading strategy.
- **Performance Metrics**: The Sortino ratio, adjusted Sharpe ratio, and cumulative geometric returns are calculated to evaluate model performance.
- **Trading Performance Comparison**: The models' trading performance is compared to selected benchmarks, including index benchmarks and naive strategies like Hidden Markov Model and KAMA+MSR.



#### Results and Discussion

**Optimal Hyperparameters**:
- The best hyperparameters for each asset class model are identified, showing similarities and differences in their configurations.

**Feature Importance**:
- The top features driving the models' predictions are primarily technical momentum variables.
- The models exhibit contrarian behavior, with the "bullish" regime often predicting market downturns and the "bearish" regime predicting market upswings.

**Model Evaluation**:
- The models' financial performance metrics demonstrate significant outperformance compared to benchmarks.
- The Equity and Commodity models show high cumulative geometric returns and Sortino ratios.
- The FX model, despite lower returns, maintains high predictive accuracy (MCC scores).

**Trading Performance**:
- The models consistently outperform index benchmarks and long-only positions within their respective asset classes.
- The contrarian nature of the models' signals is evident, providing profitable shorting and buying opportunities.

**Asset Allocation**:
- The models effectively allocate assets based on predicted regimes, capturing market troughs and peaks.
- The strategies demonstrate robustness across different market conditions and asset classes.



#### Conclusions

- The chapter successfully demonstrates the use of Random Forest models to predict financial regimes ex ante.
- The models provide signals best interpreted as contrarian, generating substantial profits by accurately predicting market downturns and upswings.
- While frequent shorting contributes significantly to profits, it presents practical challenges. Alternatives like pair trading, asset class mixing, or early-warning systems are proposed.
- The methodology's success across equities, commodities, and foreign exchange rates indicates its robustness and potential applicability in the financial industry.



### Combining Regime Switching Predictive Framework with Model Predictive Control for Multi-period Portfolio Optimisation

#### Background

Chapter 5 introduces a novel approach by integrating multi-period portfolio optimization (MPO) with model predictive control (MPC), leveraging the Random Forest models developed in Chapter 4 for market regime prediction. This method aims to address the limitations identified in Chapter 4, such as frequent shorting and fixed cost assumptions, by employing a dynamic weight allocation strategy and a more sophisticated cost model. The chapter discusses the theoretical background and related work, including the concept of multi-period optimization, and explains how MPC can optimize portfolios by considering future returns, risk constraints, and transaction costs over multiple periods.

#### Data and Methodology

**Data** 
The data used includes the same equity and commodity assets from Chapter 4, along with daily returns, volatility, and trading volumes. The period of analysis is from April 2, 2018, to April 29, 2022. The methodology involves transforming the classification predictions from Chapter 4 into return estimates, applying the Kalman filter for accuracy improvement, estimating the covariance matrix, and defining the transaction cost function. The MPC parameters are then optimized using a training and holdout sample to ensure robust performance without overfitting.

**Methodology**  
1. **Transforming Classification Predictions**: The classification probabilities from Chapter 4 are converted into return estimates using a 10-day exponential weighted moving average (EMA) and adjusting these based on the predicted regime.
2. **Applying the Kalman Filter**: The Kalman filter is employed to enhance the accuracy of the estimated returns by correcting for prediction errors.
3. **Covariance Matrix Estimation**: A rolling covariance matrix over 504 days is used to capture the risk relationships between assets.
4. **Transaction Cost Function**: The cost function is estimated using asset price, volatility, and trading volume, incorporating a bid-ask spread and other potential explicit costs.
5. **MPC Parameter Optimization**: Parameters \(\gamma_{\sigma}\) and \(\gamma_{\text{trade}}\) are optimized to maximize the Sortino ratio using the Optuna package, balancing the trade-off between risk and turnover. The investment horizon \(H\) is set to 2, based on initial experiments showing superior performance with this value.

#### Results and Discussion

The implementation of MPC demonstrated significant outperformance of the benchmarks, including 1/N and buy-and-hold portfolios, in terms of financial metrics like the Sortino ratio and adjusted Sharpe ratio. The dynamic adjustment of asset weights allowed the MPC model to capture market changes effectively, providing robust trading signals. The results confirmed that multi-period optimization offers a superior framework for handling non-stationary financial time series, enabling better risk management and return maximization compared to single-period optimization.

**Performance Analysis**  
The Equity and Commodity Random Forest models performed exceptionally well, dynamically adjusting to market conditions and achieving higher cumulative geometric returns. The analysis highlighted the effectiveness of MPC in reallocating portfolio weights based on regime predictions, especially during market downturns where the model successfully minimized losses.

**Advantages of Multi-period Optimization**  
Multi-period optimization proved advantageous over single-period techniques by incorporating future estimates and adjusting for short-term and long-term benefits. This approach aligns better with the non-stationary nature of financial markets, providing a more resilient investment strategy.

#### Conclusions

This chapter successfully demonstrated the integration of multi-period optimization with a regime-switching predictive framework using MPC. By dynamically adjusting asset weights and incorporating a realistic cost model, the proposed method achieved superior financial performance compared to traditional strategies. The study addressed the limitations of frequent shorting and fixed costs, showcasing the potential of MPC in building robust, dynamic investment portfolios. Future research can explore the inclusion of more diverse asset classes and advanced risk management strategies to further enhance the effectiveness of the MPC model.



### Conclusions and Future Work

In this final chapter, the work of this thesis is summarized and discussed, prior to outlining proposed extensions of the work in terms of detecting and predicting financial regimes in order to construct portfolios capable of withstanding turbulent times in the markets, as well as benefiting from price rallies.

#### Summary and Discussion

The ultimate objective of this thesis was to propose and test a detection-prediction-optimization regime-switching framework that would also be efficient in a real-world environment, and thus practical for asset managers. This problem was split into three research objectives, each becoming a focus of a single chapter, each chapter building upon preceding work:

1. Implementation of a novel regime detection framework.
2. Prediction of financial regimes ex-ante based on the detected regimes from Objective 1.
3. Construction of realistic, regime-robust portfolios using the generated signals from Objective 2.

To ensure the proposed methods were both novel and likely to succeed, Chapter 2 thoroughly reviewed the relevant literature and built foundations upon which the subsequent chapters were constructed.

**Chapter 3** focused on the first objective by blending a technical indicator, Kaufman’s Adaptive Moving Average (KAMA), with a popular statistical method of regime detection, Markov-switching regression (MSR), to smoothly and accurately detect financial regimes. The effectiveness of this method, called the KAMA+MSR model, was tested against several other regime detection frameworks. The KAMA+MSR model demonstrated superior stability and accuracy over its benchmarks, particularly Markov-switching regression used alone, achieving the best financial results. This model could generate concise labels to be subsequently fed into a predictive algorithm, the topic of Chapter 4.

**Chapter 4** addressed the second objective by predicting financial regimes ex-ante using the regime labels generated in Chapter 3. Through feature engineering, feature selection, and the use of a Random Forest model, it was shown that financial regimes could be predicted efficiently across various asset classes. The models displayed solid out-of-sample accuracy and excellent financial results based on a long/short, cost-inclusive trading strategy exploiting the predicted signals. However, limitations included the inability of many institutions to short assets for ethical reasons and the trading strategy not fully reflecting real-world portfolio construction.

**Chapter 5** tackled the third research objective by addressing the contributions and limitations of Chapter 4. Using the model predictive control (MPC) algorithm, it was shown that the generated signals from Chapter 4 could be successfully exploited in portfolio construction, incorporating various real-world factors such as risk, transaction, and liquidity constraints. The constructed portfolio outperformed its benchmarks without allowing shorting positions, which could potentially enhance performance but were restricted due to institutional constraints.

The combined contributions of Chapters 3, 4, and 5 achieved the ultimate objective of this research, demonstrating practical applications for a quantitative trading system as outlined in Section 2.1.3.



#### Future Work

Several extensions to this work could be suggested, as described below:

1. **Incorporation of Additional Asset Classes and Financial Instruments**:
   - Future research could explore the framework's suitability for other asset classes, such as bonds or cryptocurrencies, and financial instruments, such as options. This could lead to a more comprehensive quantitative trading system that incorporates a wider range of assets and mimics short trading by buying bonds or put options, potentially increasing cumulative return performance.

2. **Investigation of Alternative Prediction Models**:
   - While the Random Forest model was effective, further investigation could determine if alternative machine learning models, such as XGBoost, LightGBM, or neural networks (including recurrent neural networks), could provide improved performance or efficiency in capturing the complex relationships between financial data and regime changes.

3. **Incorporation of More Advanced MPO Optimization Techniques**:
   - While multi-period optimization (MPO) via MPC was effective, other optimization techniques like genetic algorithms or particle swarm optimization could be explored to search for optimal portfolio weights, potentially further improving performance.

In summary, these future directions could lead to the development of more comprehensive and sophisticated quantitative trading systems capable of robustly navigating financial markets.



Reference:

- Introduction to Markov-Switching Models

  https://www.aptech.com/blog/introduction-to-markov-switching-models/

- Markov Switching Dynamic Regression Model

  https://medium.com/@NNGCap/markov-switching-dynamic-regression-model-2a558251c293

- KAMA: The Adaptive Moving Average

  https://medium.com/@NNGCap/kama-the-adaptive-moving-average-23a2a75540be

- 
