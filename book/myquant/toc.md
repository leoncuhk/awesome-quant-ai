# Quantitative Trading Strategies: A Panoramic Knowledge Base (量化交易策略全景知识库)

```
Quant Strategy R&D Framework v3.0 (量化策略研发框架)
 ├── 1. Trend-Following / Momentum Strategies (趋势跟踪/动量策略)
 │   │   // Core logic: capture the inertia of price movements (捕捉价格运动的惯性)
 │   │
 │   ├── 1.1 Time-Series Momentum (时间序列动量)
 │   │   ├── 1.1.1 Moving Average Systems (移动平均线系统)
 │   │   │   ├── Dual/Triple MA Crossovers (双/三均线交叉: Golden/Death Cross)
 │   │   │   ├── MACD and Variants (MACD及其变种)
 │   │   │   └── Adaptive Moving Averages (自适应均线: KAMA, FRAMA)
 │   │   ├── 1.1.2 Channel & Breakout Systems (通道与突破系统)
 │   │   │   ├── Donchian Channels & Turtle Trading (唐奇安通道与海龟交易系统)
 │   │   │   ├── Bollinger Band Breakouts (布林带突破)
 │   │   │   ├── ATR Channel Strategies (ATR通道策略)
 │   │   │   └── Pattern / Support-Resistance Breakouts (形态/支撑阻力突破)
 │   │   └── 1.1.3 Trend Strength & Oscillator Indicators (趋势强度与振荡指标)
 │   │       ├── Average Directional Index (ADX, 平均趋向指数)
 │   │       ├── Parabolic SAR (抛物线转向指标)
 │   │       └── Trend Applications of RSI/Stochastic Oscillators (振荡指标的趋势用法)
 │   │
 │   └── 1.2 Cross-Sectional Momentum (横截面动量)
 │       ├── 1.2.1 Classic Relative Strength (经典相对强度)
 │       │   ├── Ranking by Raw Return (收益率排序)
 │       │   └── Ranking by Risk-Adjusted Return (风险调整后收益排序: Sharpe/Sortino)
 │       ├── 1.2.2 Sector/Style/Industry Rotation (行业/风格/板块轮动)
 │       ├── 1.2.3 Factor Momentum (因子动量) -> recent performance of value/quality and other factors
 │       └── 1.2.4 Residual Momentum (残差动量) -> idiosyncratic momentum after stripping common risk
 │
 ├── 2. Mean-Reversion Strategies (均值回归策略)
 │   │   // Core logic: exploit temporary deviations from, and reversion to, a statistical mean (捕捉价格对均衡值的暂时性偏离与回归)
 │   │
 │   ├── 2.1 Statistical Arbitrage (统计套利)
 │   │   ├── 2.1.1 Methodologies (方法论)
 │   │   │   ├── Distance/Correlation-Based Methods (距离/相关性方法)
 │   │   │   ├── Cointegration (协整分析: Engle-Granger, Johansen)
 │   │   │   ├── Principal Component Analysis (PCA, 主成分分析)
 │   │   │   └── ML-Based Pairing (机器学习配对)
 │   │   └── 2.1.2 Applications (应用场景)
 │   │       ├── Pairs/Basket Trading (股票配对/篮子套利)
 │   │       ├── ETF vs Constituents Arbitrage (ETF与成分股套利)
 │   │       ├── Cash-Futures Arbitrage (期现套利)
 │   │       └── Calendar / Inter-Exchange Spreads (跨期/跨市价差套利)
 │   │
 │   ├── 2.2 Short-Term Price Reversal (短期价格反转)
 │   │   ├── 2.2.1 Oscillator Overshoot (振荡指标超调)
 │   │   │   ├── RSI/Stochastic/Williams %R Overbought-Oversold (超买超卖)
 │   │   │   └── Z-Score / Percentile Deviation (Z分数/百分位偏离)
 │   │   └── 2.2.2 Intraday / Short-Term Reversal Patterns (日内/短期模式反转)
 │   │       ├── Overnight Gap Filling (隔夜跳空缺口回补)
 │   │       └── Bollinger Band Mean Reversion (布林带均值回归)
 │   │
 │   └── 2.3 Relative Value Convergence (相对价值回归)
 │       ├── 2.3.1 Convertible Bond Arbitrage (可转债套利)
 │       ├── 2.3.2 Closed-End Fund Discount Arbitrage (封闭式基金折价套利)
 │       └── 2.3.3 Capital Structure Arbitrage (资本结构套利) -> debt vs equity of the same issuer (e.g. CDS vs stock)
 │
 ├── 3. Risk Premia Harvesting (风险溢价策略)
 │   │   // Core logic: systematically bear specific risks in exchange for long-run compensation (系统性承担特定风险以获取长期补偿)
 │   │
 │   ├── 3.1 Equity Risk Premia / Factor Investing (股权风险溢价/因子投资)
 │   │   ├── 3.1.1 Value (价值溢价) -> P/E, P/B, EV/EBITDA, DCF
 │   │   ├── 3.1.2 Size (规模溢价) -> small-cap tilt
 │   │   ├── 3.1.3 Quality (质量溢价) -> high ROE, low leverage, stable earnings
 │   │   ├── 3.1.4 Low Volatility (低波动溢价) -> low beta, low idiosyncratic volatility
 │   │   └── 3.1.5 Momentum (动量溢价) -> price momentum, earnings momentum
 │   │
 │   ├── 3.2 Volatility Risk Premium (波动率风险溢价)
 │   │   ├── 3.2.1 Implied vs Realized Volatility (隐含 vs 实现波动率)
 │   │   │   ├── Systematic Option Selling (系统性卖出期权: Covered Call, Cash-Secured Put)
 │   │   │   ├── Defined-Risk Selling (风险限定的卖方策略: Iron Condor, Butterfly)
 │   │   │   └── Volatility Index Products (波动率指数产品: VIX Futures/Options, VXX/UVXY)
 │   │   └── 3.2.2 Volatility Structure Arbitrage (波动率结构套利)
 │   │       ├── Skew Trading (偏斜交易)
 │   │       ├── Term Structure Trading (期限结构交易)
 │   │       └── Cross-Asset Volatility Arbitrage (跨资产波动率套利)
 │   │
 │   └── 3.3 Cross-Asset Risk Premia (跨资产风险溢价)
 │       ├── 3.3.1 Carry Trade & Roll Yield (套息交易与展期收益) -> FX, commodity futures
 │       ├── 3.3.2 Credit Risk Premium (信用风险溢价) -> corporate bonds vs treasuries
 │       ├── 3.3.3 Liquidity Risk Premium (流动性风险溢价) -> illiquid asset discounts
 │       └── 3.3.4 Inflation Risk Premium (通胀风险溢价) -> TIPS vs nominal bonds
 │
 ├── 4. Market Microstructure Strategies (市场微观结构策略)
 │   │   // Core logic: profit from market rules, order book information, and speed advantages (利用交易机制、订单信息和速度优势)
 │   │
 │   ├── 4.1 Liquidity Provision (流动性供给)
 │   │   └── Market Making (做市策略: Passive, Active, Statistical)
 │   ├── 4.2 Order Flow Prediction (订单流预测)
 │   │   └── Order Book Analysis (订单簿分析: Imbalance, Toxicity)
 │   ├── 4.3 Liquidity Detection (流动性侦测)
 │   │   └── Hidden Action Analysis (隐藏行为分析: Icebergs, Algo Detection, Dark Pools)
 │   └── 4.4 Latency Arbitrage (延迟套利)
 │       └── Cross-Exchange / Cross-Feed Arbitrage (跨市场/数据源套利)
 │
 └── 5. Event-Driven & Information-Based Strategies (事件驱动与信息策略)
     │   // Core logic: systematic, deep, and fast reaction to information (对各类信息的系统化、深度化、快速化反应)
     │
     ├── 5.1 Corporate Events (公司事件)
     │   ├── 5.1.1 M&A and Restructuring (并购与重组) -> merger arbitrage, SPAC arbitrage, bankruptcy restructuring
     │   ├── 5.1.2 Earnings Events (财报事件) -> post-earnings announcement drift (PEAD), earnings guidance
     │   └── 5.1.3 Capital Actions (资本运作事件) -> buybacks, secondary offerings, dividends, equity incentives
     │
     ├── 5.2 Regulatory & Institutional Events (监管与制度性事件)
     │   ├── 5.2.1 Index Rebalancing Arbitrage (指数调整套利)
     │   └── 5.2.2 Regulatory Policy Arbitrage (监管政策套利) -> trading halts/resumptions, rule changes
     │
     ├── 5.3 Quantitative Fundamental (基本面量化)
     │   ├── 5.3.1 Factor Engineering (因子工程) -> financial ratios, growth, profitability, ESG factors
     │   └── 5.3.2 Deep Fundamental Models (深度基本面模型) -> DuPont analysis, DCF, Piotroski F-Score
     │
     ├── 5.4 Alternative Data-Driven (另类数据驱动)
     │   ├── 5.4.1 Text Mining / NLP (文本挖掘) -> news sentiment, social media, filings and research parsing
     │   ├── 5.4.2 Image & Geo-Data (图像与地理数据) -> satellite imagery, logistics tracking
     │   └── 5.4.3 Web & Transaction Data (网络与交易数据) -> search trends, card spending, job postings
     │
     └── 5.5 Global Macro Quant (宏观量化)
         ├── 5.5.1 Business Cycle Models (经济周期模型)
         ├── 5.5.2 Monetary/Fiscal Policy Models (货币/财政政策模型)
         └── 5.5.3 Geopolitical Risk Models (地缘政治风险模型)
```

```
Quant Knowledge Framework (量化知识框架)
│
├── [Layer 1: Theoretical Foundations (理论基础)]
│   ├── Finance Theory (金融学理论)
│   │   ├── Efficient Market Hypothesis (有效市场假说)
│   │   ├── Behavioral Finance (行为金融学)
│   │   ├── Asset Pricing Theory (资产定价理论)
│   │   └── Market Microstructure Theory (市场微观结构理论)
│   ├── Mathematics & Statistics (数学与统计学)
│   │   ├── Probability & Stochastic Processes (概率论与随机过程)
│   │   ├── Time Series Analysis (时间序列分析)
│   │   ├── Multivariate Statistics (多元统计分析)
│   │   └── Optimization Theory (优化理论)
│   └── Computer Science (计算机科学)
│       ├── Algorithms & Data Structures (算法与数据结构)
│       ├── Machine Learning (机器学习)
│       ├── Deep Learning (深度学习)
│       └── High-Performance Computing (高性能计算)
│
├── [Layer 2: Strategy Taxonomy by Source of Profit (策略分类，按盈利来源)]
│   ├── 1. Price Trend Strategies (价格趋势策略)
│   │   ├── Momentum (动量策略)
│   │   ├── Trend Following (趋势跟踪)
│   │   ├── Breakout (突破策略)
│   │   └── Technical Indicator Strategies (技术指标策略)
│   ├── 2. Value Reversion Strategies (价值回归策略)
│   │   ├── Statistical Arbitrage (统计套利)
│   │   ├── Pairs Trading (配对交易)
│   │   ├── Mean Reversion (均值回归)
│   │   └── Cointegration Strategies (协整策略)
│   ├── 3. Structural Arbitrage Strategies (结构性套利策略)
│   │   ├── Cash-Futures Arbitrage (期现套利)
│   │   ├── Calendar Arbitrage (跨期套利)
│   │   ├── Cross-Market Arbitrage (跨市场套利)
│   │   └── ETF Arbitrage (ETF套利)
│   ├── 4. Risk Premia Strategies (风险溢价策略)
│   │   ├── Factor Investing (因子投资)
│   │   ├── Volatility Strategies (波动率策略)
│   │   ├── Risk Parity (风险平价)
│   │   └── Alternative Risk Premia (另类风险溢价)
│   ├── 5. Market Microstructure Strategies (市场微观结构策略)
│   │   ├── Market Making (做市策略)
│   │   ├── Liquidity Provision (流动性提供)
│   │   ├── Order Flow Analysis (订单流分析)
│   │   └── Latency Arbitrage (延迟套利)
│   └── 6. Quantitative Fundamental Strategies (基本面量化策略)
│       ├── Multi-Factor Models (多因子模型)
│       ├── Event-Driven (事件驱动)
│       ├── Alternative Data (另类数据)
│       └── Macro Quant (宏观量化)
│
├── [Layer 3: Implementation (实施技术)]
│   ├── Strategy Development (策略开发)
│   │   ├── Signal Generation (信号生成)
│   │   ├── Parameter Optimization (参数优化)
│   │   ├── Backtesting Systems (回测系统)
│   │   └── Paper Trading (模拟交易)
│   ├── Trade Execution (交易执行)
│   │   ├── Order Management (订单管理)
│   │   ├── Algorithmic Execution (算法交易)
│   │   ├── Transaction Cost Analysis (交易成本分析)
│   │   └── Slippage Control (滑点控制)
│   ├── Risk Management (风险管理)
│   │   ├── Position Management (头寸管理)
│   │   ├── Risk Measurement (风险度量: VaR, CVaR)
│   │   ├── Stress Testing (压力测试)
│   │   └── Dynamic Hedging (动态对冲)
│   └── System Architecture (系统架构)
│       ├── Data Management (数据管理)
│       ├── Strategy Engine (策略引擎)
│       ├── Trading Gateway (交易网关)
│       └── Monitoring Systems (监控系统)
│
├── [Layer 4: Portfolio Management (组合管理)]
│   ├── Asset Allocation (资产配置)
│   │   ├── Mean-Variance Optimization (均值方差优化)
│   │   ├── Black-Litterman Model (Black-Litterman模型)
│   │   ├── Risk Budgeting (风险预算)
│   │   └── Dynamic Rebalancing (动态再平衡)
│   ├── Strategy Portfolio (策略组合)
│   │   ├── Strategy Correlation Analysis (策略相关性分析)
│   │   ├── Strategy Weight Optimization (策略权重优化)
│   │   ├── Strategy Rotation (策略轮动)
│   │   └── Strategy Hedging (策略对冲)
│   └── Performance Evaluation (业绩评估)
│       ├── Return Analysis (收益分析)
│       ├── Risk Analysis (风险分析)
│       ├── Attribution Analysis (归因分析)
│       └── Benchmark Comparison (基准比较)
│
└── [Layer 5: Practice (实践应用)]
    ├── Market Environment (市场环境)
    │   ├── Asset Class Characteristics (资产类别特性)
    │   ├── Market Microstructure Differences (市场微结构差异)
    │   ├── Regulatory Requirements (监管要求)
    │   └── Transaction Cost Structures (交易成本结构)
    ├── Institutional Practice (机构实践)
    │   ├── Hedge Fund Model (对冲基金模式)
    │   ├── Proprietary Trading Model (自营交易模式)
    │   ├── Asset Management Products (资管产品模式)
    │   └── Market Maker Model (做市商模式)
    └── Frontier Developments (前沿发展)
        ├── AI/ML Applications (AI/ML应用)
        ├── Alternative Data (另类数据)
        ├── Quantum Computing (量子计算)
        └── DeFi Quant (DeFi量化)
```
