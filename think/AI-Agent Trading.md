# AI-Agent Trading: LLM-Based Multi-Agent Frameworks

Survey notes on LLM-based (multi-)agent trading systems — what the main frameworks actually do, how they differ, and what remains unsolved. Updated 2026-07.

## From Rules to Agents

Quantitative trading systems can be roughly ordered by decision autonomy:

| Dimension | Traditional quant | Algorithmic / ML-optimized | LLM agent-based |
|---|---|---|---|
| Decision process | Static rules from models and historical data | Predefined logic with parameter optimization | Autonomous agents reasoning over heterogeneous inputs |
| Adaptivity | Low; manual re-tuning | Medium; periodic re-optimization | High in principle; memory and reflection loops, but unproven out of regime |
| Data handled | Structured prices/fundamentals | Multiple structured sources | Unstructured text, news, filings; sometimes charts (multimodal) |
| Transparency | High; explicit rules | Medium; traceable but complex | Natural-language rationales, yet underlying decisions remain hard to audit |
| Cost profile | Low | Medium | High per decision (API calls, multi-round debate) |

The LLM-agent column is the newest and least validated. Four frameworks define the current design space.

## Framework Survey

### TradingAgents (Xiao et al., 2024)

TradingAgents organizes LLM agents into the role structure of a trading firm: fundamental, sentiment, and technical analysts; bull and bear researchers who debate each thesis; traders with different risk appetites; and a risk-management layer that gates final decisions. The core mechanism is structured communication — analysts produce reports, researchers argue both sides, and the trader synthesizes the debate into an action. The authors report improvements in cumulative return, Sharpe ratio, and maximum drawdown over baselines; the design bet is that adversarial debate reduces confident, one-sided reasoning by a single model.

### FinAgent (Zhang et al., 2024)

FinAgent is a tool-augmented, multimodal single-agent design rather than a role-playing ensemble. It ingests numerical, textual, and visual market data (including price charts) through a market-intelligence module, and adds a dual-level reflection module for adapting to market dynamics, a diversified memory-retrieval system, and tool augmentation that lets the agent invoke established trading strategies and expert insight rather than generate every decision from scratch. Evaluated on six datasets spanning stocks and cryptocurrency, the authors report an average profit improvement of over 36% against nine baselines. Its distinctive claim is being the first multimodal foundation agent designed specifically for trading.

### FinMem (Yu et al., 2023)

FinMem is the memory-centric design in this group. It gives a single trading agent a human-inspired layered memory — working memory plus long-term layers whose retention decays at different rates depending on the time-sensitivity of the information — together with an explicit character design (professional background and risk inclination) that conditions how the agent weighs evidence. The layered structure lets the agent prioritize recent critical news while retaining slower-moving fundamental context, improving single-stock trading over unstructured-memory baselines. FinMem established the memory-and-profile template that later systems, including FinCon, build on.

### FinCon (Yu et al., 2024)

FinCon, from the FinMem group, scales the idea to a synthesized multi-agent system with a manager-analyst hierarchy modeled on real investment firms: analyst agents distill unimodal market information, and a single manager agent consolidates their input, makes trading decisions, and carries an explicit risk-control component that monitors episodic risk. Its main methodological contribution is conceptual verbal reinforcement — a self-critique mechanism that updates the manager's investment beliefs in natural language across episodes, selectively propagating only revised beliefs to the agents that need them to cut communication cost. It is evaluated on both single-stock trading and portfolio management, making it the broadest in task scope of the four.

## Framework Comparison

| | TradingAgents | FinAgent | FinMem | FinCon |
|---|---|---|---|---|
| Architecture | Multi-agent, firm role structure | Single agent, tool-augmented | Single agent | Multi-agent, manager-analyst hierarchy |
| Key mechanism | Bull/bear debate, role specialization | Multimodal perception, dual-level reflection | Layered memory with decay, character design | Conceptual verbal reinforcement, belief updates |
| Modality | Text, numerical | Text, numerical, visual (charts) | Text, numerical | Text, numerical |
| Risk control | Dedicated risk-management agents | Implicit (reflection) | Risk-inclination profile | Explicit risk-control component (episodic monitoring) |
| Tasks evaluated | Stock trading | Stock and crypto trading | Single-stock trading | Stock trading and portfolio management |
| Reference | [arXiv:2412.20138](https://arxiv.org/abs/2412.20138) | [arXiv:2402.18485](https://arxiv.org/abs/2402.18485) | [arXiv:2311.13743](https://arxiv.org/abs/2311.13743) | [arXiv:2407.06567](https://arxiv.org/abs/2407.06567) |

For engineering rather than research, the adjacent open-source stack includes [FinRL](https://github.com/AI4Finance-Foundation/FinRL) (RL agents), [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (financial LLM fine-tuning), and [RD-Agent](https://github.com/microsoft/RD-Agent) (autonomous factor/model R&D on Qlib), plus the open-source [TradingAgents](https://github.com/TauricResearch/TradingAgents) implementation itself.

## Open Problems

The reported results above should be read with caution. Several validity issues are common to all four papers and largely unresolved:

- **Pretraining data leakage.** Backtests run over periods inside the LLM's training window are contaminated: the model may have memorized how specific stocks, earnings, and macro events played out. A "prediction" about 2022 from a model trained through 2023 is not out-of-sample. Truly clean evaluation requires post-cutoff live or paper trading, which few papers report at meaningful length.
- **Lookahead in news corpora.** Retrieved news and analyst text often carry timestamps that are approximate, revised, or aggregated, so information can leak backward into the decision point even when prices are handled correctly.
- **Cost and latency.** Multi-agent debate multiplies API calls per decision. This is tolerable for daily rebalancing on a handful of tickers but scales poorly to broad universes or intraday horizons, and per-decision cost is rarely reported alongside returns.
- **Non-stationarity.** Memory and verbal-reinforcement loops adapt beliefs to the evaluated regime; there is no evidence yet that these adaptations survive regime shifts rather than encode them.
- **Weak evaluation standards.** Typical evaluations use small stock universes, short windows, and baselines like buy-and-hold or vanilla RL. LLM output is stochastic and prompt-sensitive, yet results are seldom reported across seeds or prompt variants. The field lacks a shared benchmark with enforced information-availability constraints.

Until these are addressed, the honest reading is that LLM agent frameworks are promising architectures for research, not validated sources of alpha.

## References

1. Xiao, Y., Sun, E., Luo, D., & Wang, W. (2024). TradingAgents: Multi-Agents LLM Financial Trading Framework. *arXiv preprint arXiv:2412.20138*. [https://arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138)
2. Zhang, W., Zhao, L., Xia, H., Sun, S., Sun, J., Qin, M., Li, X., Zhao, Y., Zhao, Y., Cai, X., Zheng, L., Wang, X., & An, B. (2024). A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist. *arXiv preprint arXiv:2402.18485*. [https://arxiv.org/abs/2402.18485](https://arxiv.org/abs/2402.18485)
3. Yu, Y., et al. (2023). FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design. *arXiv preprint arXiv:2311.13743*. [https://arxiv.org/abs/2311.13743](https://arxiv.org/abs/2311.13743)
4. Yu, Y., et al. (2024). FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making. *arXiv preprint arXiv:2407.06567*. [https://arxiv.org/abs/2407.06567](https://arxiv.org/abs/2407.06567)
