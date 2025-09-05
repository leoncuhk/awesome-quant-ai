# 量化交易策略全景知识库

```
量化策略研发框架 v3.0
 ├── 1. 趋势跟踪策略 (Trend-Following / Momentum Strategies)
 │   │   // 核心逻辑: 捕捉价格运动的惯性 (Inertia of price movements)
 │   │
 │   ├── 1.1 时间序列动量 (Time-Series Momentum)
 │   │   ├── 1.1.1 移动平均线系统 (Moving Average Systems)
 │   │   │   ├── 双均线/三均线交叉 (Dual/Triple MA Cross: Golden/Death Cross)
 │   │   │   ├── 异同移动平均线 (MACD) 及其变种
 │   │   │   └── 自适应移动平均线 (Adaptive MA: KAMA, FRAMA)
 │   │   ├── 1.1.2 通道与突破系统 (Channel & Breakout Systems)
 │   │   │   ├── 唐奇安通道 (Donchian Channels) 与海龟交易系统 (Turtle Trading)
 │   │   │   ├── 布林带突破 (Bollinger Band Breakouts)
 │   │   │   ├── ATR通道策略 (ATR Channel Strategy)
 │   │   │   └── 形态/支撑阻力突破 (Pattern/SR Breakout)
 │   │   └── 1.1.3 趋势强度与振荡指标 (Trend Strength & Oscillator Indicators)
 │   │       ├── 平均趋向指数 (ADX)
 │   │       ├── 抛物线转向指标 (Parabolic SAR)
 │   │       └── RSI/Stochastic 等振荡指标的趋势用法
 │   │
 │   └── 1.2 横截面动量 (Cross-Sectional Momentum)
 │       ├── 1.2.1 经典相对强度 (Classic Relative Strength)
 │       │   ├── 资产收益率排序 (Ranking by Return)
 │       │   └── 风险调整后收益排序 (Ranking by Sharpe/Sortino Ratio)
 │       ├── 1.2.2 行业/风格/板块轮动 (Sector/Style/Industry Rotation)
 │       ├── 1.2.3 因子动量 (Factor Momentum) -> (判断价值/质量等因子近期表现)
 │       └── 1.2.4 残差动量 (Residual Momentum) -> (剔除共同风险后的特质动量)
 │
 ├── 2. 均值回归策略 (Mean-Reversion Strategies)
 │   │   // 核心逻辑: 捕捉价格对均衡值的暂时性偏离与回归 (Reversion to a statistical mean)
 │   │
 │   ├── 2.1 统计套利 (Statistical Arbitrage)
 │   │   ├── 2.1.1 方法论 (Methodologies)
 │   │   │   ├── 基于距离/相关性 (Distance/Correlation)
 │   │   │   ├── 协整分析 (Cointegration: Engle-Granger, Johansen)
 │   │   │   ├── 主成分分析 (PCA)
 │   │   │   └── 机器学习配对 (ML-based Pairing)
 │   │   └── 2.1.2 应用场景 (Applications)
 │   │       ├── 股票配对/篮子套利 (Pairs/Basket Trading)
 │   │       ├── ETF与成分股套利 (ETF vs Constituents Arbitrage)
 │   │       ├── 期现套利 (Cash-Futures Arbitrage)
 │   │       └── 跨期/跨市价差套利 (Calendar/Inter-Exchange Spread)
 │   │
 │   ├── 2.2 短期价格反转 (Short-Term Price Reversal)
 │   │   ├── 2.2.1 振荡指标超调 (Oscillator Overshoot)
 │   │   │   ├── RSI/Stochastic/Williams %R 超买超卖
 │   │   │   └── Z-Score/百分位偏离 (Z-Score/Percentile Deviation)
 │   │   └── 2.2.2 日内/短期模式反转 (Intraday/Short-Term Patterns)
 │   │       ├── 隔夜跳空缺口回补 (Overnight Gap Filling)
 │   │       └── 布林带均值回归 (Bollinger Band Mean Reversion)
 │   │
 │   └── 2.3 相对价值回归 (Relative Value Convergence)
 │       ├── 2.3.1 可转债套利 (Convertible Bond Arbitrage)
 │       ├── 2.3.2 封闭式基金折价套利 (Closed-End Fund Discount)
 │       └── 2.3.3 资本结构套利 (Capital Structure Arbitrage) -> (普通股 vs 优先股)
 │
 ├── 3. 风险溢价策略 (Risk Premia Harvesting)
 │   │   // 核心逻辑: 系统性地承担特定风险以获取长期补偿 (Systematic risk exposure for compensation)
 │   │
 │   ├── 3.1 股权类风险溢价 (Equity Risk Premia / Factor Investing)
 │   │   ├── 3.1.1 价值溢价 (Value) -> (P/E, P/B, EV/EBITDA, DCF)
 │   │   ├── 3.1.2 规模溢价 (Size) -> (小市值公司)
 │   │   ├── 3.1.3 质量溢价 (Quality) -> (高ROE, 低杠杆, 盈利稳定)
 │   │   ├── 3.1.4 低波动溢价 (Low Volatility) -> (低Beta, 低特质波动率)
 │   │   └── 3.1.5 动量溢价 (Momentum) -> (价格动量, 盈余动量)
 │   │
 │   ├── 3.2 波动率风险溢价 (Volatility Risk Premium)
 │   │   ├── 3.2.1 隐含 vs 实现波动率 (IV vs RV)
 │   │   │   ├── 系统性卖出期权 (Systematic Option Selling: Covered Call, Cash-Secured Put)
 │   │   │   ├── 风险限定的卖方策略 (Defined-Risk Selling: Iron Condor, Butterfly)
 │   │   │   └── 波动率指数产品交易 (VIX Futures/Options, VXX/UVXY)
 │   │   └── 3.2.2 波动率结构套利 (Volatility Structure Arbitrage)
 │   │       ├── 偏斜交易 (Skew Trading)
 │   │       ├── 期限结构交易 (Term Structure Trading)
 │   │       └── 跨资产波动率套利 (Cross-Asset Volatility Arbitrage)
 │   │
 │   └── 3.3 跨资产风险溢价 (Cross-Asset Risk Premia)
 │       ├── 3.3.1 套息交易与展期收益 (Carry Trade & Roll Yield) -> (FX, 商品期货)
 │       ├── 3.3.2 信用风险溢价 (Credit Risk Premium) -> (公司债 vs 国债)
 │       ├── 3.3.3 流动性风险溢价 (Liquidity Risk Premium) -> (非流动资产折价)
 │       └── 3.3.4 通胀风险溢价 (Inflation Risk Premium) -> (TIPS vs 名义债券)
 │
 ├── 4. 市场微观结构策略 (Market Microstructure Strategies)
 │   │   // 核心逻辑: 利用交易机制、订单信息和速度优势盈利 (Exploiting market rules, order book information, and speed)
 │   │
 │   ├── 4.1 流动性供给 (Liquidity Provision)
 │   │   └── 做市策略 (Market Making: Passive, Active, Statistical)
 │   ├── 4.2 订单流预测 (Order Flow Prediction)
 │   │   └── 订单簿分析 (Order Book Analysis: Imbalance, Toxicity)
 │   ├── 4.3 流动性侦测 (Liquidity Detection)
 │   │   └── 隐藏行为分析 (Hidden Action Analysis: Icebergs, Algo Detection, Dark Pools)
 │   └── 4.4 延迟套利 (Latency Arbitrage)
 │       └── 跨市场/数据源套利 (Cross-Exchange/Feed Arbitrage)
 │
 └── 5. 事件驱动与信息策略 (Event-Driven & Information-Based Strategies)
     │   // 核心逻辑: 对各类信息的系统化、深度化、快速化反应 (Systematic reaction to information)
     │
     ├── 5.1 公司事件 (Corporate Events)
     │   ├── 5.1.1 并购与重组 (M&A and Restructuring) -> (并购套利, SPAC套利, 破产重组)
     │   ├── 5.1.2 财报事件 (Earnings Events) -> (发布前后漂移 PEAD, 业绩预告)
     │   └── 5.1.3 资本运作事件 (Capital Actions) -> (回购, 增发, 分红, 股权激励)
     │
     ├── 5.2 监管与制度性事件 (Regulatory & Institutional Events)
     │   ├── 5.2.1 指数调整套利 (Index Rebalancing)
     │   └── 5.2.2 监管政策套利 (Regulatory Policy Arbitrage: 停复牌, 规则变动)
     │
     ├── 5.3 基本面量化 (Quantitative Fundamental)
     │   ├── 5.3.1 因子工程 (Factor Engineering) -> (财务比率, 成长性, 盈利能力, ESG因子)
     │   └── 5.3.2 深度基本面模型 (Deep Fundamental Models) -> (杜邦分析, DCF, Piotroski F-Score)
     │
     ├── 5.4 另类数据驱动 (Alternative Data-Driven)
     │   ├── 5.4.1 文本挖掘 (NLP) -> (新闻情绪, 社交媒体, 研报/年报解析)
     │   ├── 5.4.2 图像与地理数据 (Image & Geo-Data) -> (卫星图, 物流追踪)
     │   └── 5.4.3 网络与交易数据 (Web & Transaction) -> (搜索趋势, 信用卡消费, 招聘信息)
     │
     └── 5.5 宏观量化 (Global Macro Quant)
         ├── 5.5.1 经济周期模型 (Business Cycle Models)
         ├── 5.5.2 货币/财政政策模型 (Monetary/Fiscal Policy Models)
         └── 5.5.3 地缘政治风险模型 (Geopolitical Risk Models)
```

```
量化知识框架
│
├── 【第一层：理论基础】
│   ├── 金融学理论
│   │   ├── 有效市场假说
│   │   ├── 行为金融学
│   │   ├── 资产定价理论
│   │   └── 市场微观结构理论
│   ├── 数学与统计学
│   │   ├── 概率论与随机过程
│   │   ├── 时间序列分析
│   │   ├── 多元统计分析
│   │   └── 优化理论
│   └── 计算机科学
│       ├── 算法与数据结构
│       ├── 机器学习
│       ├── 深度学习
│       └── 高性能计算
│
├── 【第二层：策略分类】（按盈利来源）
│   ├── 1. 价格趋势策略
│   │   ├── 动量策略
│   │   ├── 趋势跟踪
│   │   ├── 突破策略
│   │   └── 技术指标策略
│   ├── 2. 价值回归策略
│   │   ├── 统计套利
│   │   ├── 配对交易
│   │   ├── 均值回归
│   │   └── 协整策略
│   ├── 3. 结构性套利策略
│   │   ├── 期现套利
│   │   ├── 跨期套利
│   │   ├── 跨市场套利
│   │   └── ETF套利
│   ├── 4. 风险溢价策略
│   │   ├── 因子投资
│   │   ├── 波动率策略
│   │   ├── 风险平价
│   │   └── 另类风险溢价
│   ├── 5. 市场微观结构策略
│   │   ├── 做市策略
│   │   ├── 流动性提供
│   │   ├── 订单流分析
│   │   └── 延迟套利
│   └── 6. 基本面量化策略
│       ├── 多因子模型
│       ├── 事件驱动
│       ├── 另类数据
│       └── 宏观量化
│
├── 【第三层：实施技术】
│   ├── 策略开发
│   │   ├── 信号生成
│   │   ├── 参数优化
│   │   ├── 回测系统
│   │   └── 模拟交易
│   ├── 交易执行
│   │   ├── 订单管理
│   │   ├── 算法交易
│   │   ├── 交易成本分析
│   │   └── 滑点控制
│   ├── 风险管理
│   │   ├── 头寸管理
│   │   ├── 风险度量（VaR, CVaR）
│   │   ├── 压力测试
│   │   └── 动态对冲
│   └── 系统架构
│       ├── 数据管理
│       ├── 策略引擎
│       ├── 交易网关
│       └── 监控系统
│
├── 【第四层：组合管理】
│   ├── 资产配置
│   │   ├── 均值方差优化
│   │   ├── Black-Litterman模型
│   │   ├── 风险预算
│   │   └── 动态再平衡
│   ├── 策略组合
│   │   ├── 策略相关性分析
│   │   ├── 策略权重优化
│   │   ├── 策略轮动
│   │   └── 策略对冲
│   └── 业绩评估
│       ├── 收益分析
│       ├── 风险分析
│       ├── 归因分析
│       └── 基准比较
│
└── 【第五层：实践应用】
    ├── 市场环境
    │   ├── 不同资产类别特性
    │   ├── 不同市场微结构
    │   ├── 监管要求
    │   └── 交易成本结构
    ├── 机构实践
    │   ├── 对冲基金模式
    │   ├── 自营交易模式
    │   ├── 资管产品模式
    │   └── 做市商模式
    └── 前沿发展
        ├── AI/ML应用
        ├── 另类数据
        ├── 量子计算
        └── DeFi量化
```
