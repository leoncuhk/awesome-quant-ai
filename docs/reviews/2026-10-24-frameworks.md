# Framework and research submission review — 2026-10-24

This is a dated review, not a change to the project's admission criteria. Decisions below apply the existing [contribution guide](../../CONTRIBUTING.md). All responses are drafts; this review did not post comments, close requests, or merge changes.

## Verification boundary

Reviewed the actual PR patches and current first-party repository files, documentation, licenses, and paper. No submitted software was executed, no account was created, and no trading or paid model calls were made. Source inspection establishes documented implementation and fit, not investment performance, reproducibility of proprietary evaluations, or production reliability. Repository popularity is only a dated supporting signal.

| Request | Reviewed PR head | Decision |
| --- | --- | --- |
| [#65 oracle3](https://github.com/leoncuhk/awesome-quant-ai/pull/65) | `0f19363415d0368d03b18feac116074a87fe0df3` | Accept with clearer execution wording. |
| [#66 AxonX](https://github.com/leoncuhk/awesome-quant-ai/pull/66) | `c1544b8728eeff9b7b41d6abdb4025671812b4cf` | Defer inclusion; invite evidence of external use. |
| [#75 AgonAlpha](https://github.com/leoncuhk/awesome-quant-ai/pull/75) | `bdfc21e42adb9fc98fcdbad275b04fe8cdf31b80` | Accept as a preprint with code; remove unverified venue claim. |
| [#76 LumiBot](https://github.com/leoncuhk/awesome-quant-ai/pull/76) | `2a41cf9ce82cb49666a4721c763edecb285ff76b` | Accept with a concise, neutral row. |
| [#24 Rug Munch](https://github.com/leoncuhk/awesome-quant-ai/issues/24) | Issue, not a PR. | Decline under existing scope/access criteria. |
| [#70 OpenBacktesting](https://github.com/leoncuhk/awesome-quant-ai/issues/70) | Issue, not a PR. | Decline for current scope and maturity. |

## #65 oracle3

The renamed repository exists and documents Apache-2.0 licensing. The current README separates MCP market reads, relationship calculations and a local paper ledger from the CLI/Python live execution engine. The relationship checker computes fees and candidate baskets but assumes the stated event relationship and does not establish available depth or fillability. The submitted wording could obscure that separation. This is maintenance of an existing entry, not a new maturity exception. Sources: [README](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/README.md), [arbitrage implementation](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/oracle3/arbitrage.py).

Suggested entry in the existing LLM-based agents section:

```markdown
- [oracle3](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent) - Prediction-market research tools with MCP market access, fee-aware checks across related event contracts, a local paper ledger, and a separate Python engine for paper/live execution; Apache-2.0.
```

Draft response:

> Thanks for keeping the existing entry current. The renamed repository and fee-aware relationship checks are documented. I would update the link and lightly edit the description to distinguish MCP research and paper tools from the separate execution engine, without implying that a calculated edge guarantees an executable arbitrage.

## #66 AxonX

The Apache-2.0 harness has a concrete quantitative workflow, rather than just an LLM wrapper: Qlib-derived Alpha158 features, LightGBM training, prediction and backtesting, with task/artifact tracking. The current plugin is `plugins/qlib_a158`, superseding the PR's `plugins/a158` link. Its data preparation needs a Tushare token, and its adapted protocol differs from original Qlib. Sources: [license](https://github.com/FlowLLM-AI/AxonX/blob/main/LICENSE), [Alpha158 plugin](https://github.com/FlowLLM-AI/AxonX/blob/main/plugins/qlib_a158/README.md).

The PR discloses a September 2026 creation and no established independent adoption. The repository page showed 5 stars at review. Its own README acknowledges that the reported enhancement experiments lack an independent confirmation window. These observations support deferral under the existing new-repository guidance; transparent limitations are a positive signal but do not replace external validation. Sources: [submission](https://github.com/leoncuhk/awesome-quant-ai/pull/66), [repository and evidence limits](https://github.com/FlowLLM-AI/AxonX).

Draft response:

> Thank you for the clear disclosure and the inspectable Alpha158 workflow. The project fits our subject area, but it is still very new and independent adoption or validation is not yet established. I am deferring inclusion under the current maturity guidance. Please resubmit with documented external research use or an independently reproduced workflow; a concrete baseline and clearly separated confirmation period would also strengthen the evidence. The current plugin URL should point to `plugins/qlib_a158`.

## #75 AgonAlpha

The paper's identity, authors and 2026 preprint status are verified on [arXiv](https://arxiv.org/abs/2608.11250). The [paper](https://arxiv.org/html/2608.11250v1) reports 60 BRAIN submissions by five co-authors across six backends, using a shared US TOP3000 2019–2023 setting. Those are author-reported platform evaluations, not an independently reproduced strategy result or prospective trading record. The public [repository](https://github.com/AutoResearch-Factory/AgonAlpha) provides MIT-licensed orchestration, prompts and an MCTS scheduler; users supply their own evaluator and data. Its [reviewer instructions](https://github.com/AutoResearch-Factory/AgonAlpha/blob/main/agents/alpha-reviewer.md) allow evaluation reruns when needed and zero scoring for evidenced cheating, so saying every factor is necessarily rerun overstates the current prompt contract.

No official acceptance record was found for the PR's “NeurIPS 2026 Workshop (Agenthon)” claim. The [Agenthon organizer page](https://www.agenthon.net/about/) calls it a Competition Track; that does not verify this paper's acceptance. Remove the venue label until an official paper-specific record is available. The research paper plus inspectable implementation supports inclusion in Research Papers independently of the repository's small audience; this is not a blanket exception for new tools.

Suggested entry in the submitted Research Papers → AI Agents for Trading section:

```markdown
- [AgonAlpha: Autonomous Alpha Discovery via Prompt Economy and Scalable Agentic Search](https://arxiv.org/abs/2608.11250) by Ye et al. (2026) - Preprint on alpha discovery with an MCTS scheduler and an adversarial reviewer that can re-evaluate candidates and reject unsupported results; includes [open-source orchestration](https://github.com/AutoResearch-Factory/AgonAlpha), with user-supplied data and evaluator.
```

Draft response:

> Thanks for sharing the paper and code. The artifact-based search and adversarial review are relevant research contributions. I would include this as a 2026 preprint with open-source orchestration, noting the user-supplied evaluator. I would remove the workshop attribution until a paper-specific official acceptance link is available, and use “can re-evaluate” to match the public reviewer contract. The BRAIN results remain attributed to the authors rather than presented as independently reproduced performance.

## #76 LumiBot

The public framework has a substantial existing community (about 2.2k stars in the reviewed repository page) and an actual built-in agent workflow. The canonical example creates separate researcher and trading agents within the Strategy lifecycle, with trading disabled for the researcher. The source explicitly requires a model API key and says model calls incur charges. Its direct entrypoint starts a historical backtest; live operation needs a separately configured broker runtime. Sources: [repository](https://github.com/Lumiwealth/lumibot), [example source](https://github.com/Lumiwealth/lumibot/blob/dev/lumibot/example_strategies/ai_researcher_trader.py), [agent guide](https://github.com/Lumiwealth/lumibot/blob/dev/docs/AI_TRADING_AGENTS.md), [GPL-3.0 license](https://github.com/Lumiwealth/lumibot/blob/dev/LICENSE).

Accept on open implementation, maintenance/community and AI workflow fit. Omit the exact broker count: adapter capabilities vary, and this review did not test individual integrations. Avoid implying that shared strategy code proves equivalent backtest/live outcomes.

Suggested row in Strategy Development Frameworks:

```markdown
| **[LumiBot](https://github.com/Lumiwealth/lumibot)** | Python backtesting and live-trading framework with a built-in LLM agent runtime and shared strategy lifecycle. | GPL-3.0; broker and data access depend on configuration; AI examples require a model API key. |
```

Draft response:

> Thanks for the contribution and affiliation disclosure. The public agent runtime and researcher/trader example establish a clear fit alongside the existing frameworks. I would include a shorter row covering the shared strategy lifecycle and GPL-3.0 license, while making the model-key requirement clear and leaving broker-specific capabilities to the documentation.

## #24 Rug Munch Intelligence

The [submission](https://github.com/leoncuhk/awesome-quant-ai/issues/24) describes payment per call and token-safety, launch and wallet-monitoring tools. It does not supply a public research implementation or a persistent free data API. Both scope and access fall short of the current guide. The website returned no readable content through the research browser and the submitted worker endpoint was inaccessible through that tool; neither observation proves the service is offline.

Draft response:

> Thanks for the suggestion. This list focuses on inspectable AI/ML quantitative-finance research and supporting tools. The submission currently centers on token-safety and launch monitoring, and its pay-per-call access does not meet our persistent-free-API criterion. I am declining inclusion on that basis. A future submission with a relevant open-source research component, public methodology and suitable access terms can be reconsidered.

## #70 OpenBacktesting

The submission candidly describes a new general-purpose backtester without established external adoption or a native AI/ML workflow. The [current README](https://github.com/lucidcog/openbacktesting) confirms a single-asset historical trading model and no core live engine. An open license and useful backtest mechanics alone do not justify another general backtester in this list. The stated research maturity and scope are sufficient grounds to decline without requesting additional work that would not fix the current fit.

Draft response:

> Thank you for the candid scope and maturity disclosure. I am declining inclusion for now: the current project is a general historical backtester, while this list concentrates on AI/ML research and already links to broader quantitative-tool lists. Its short public history and lack of independent adoption also fall below our usual maturity guidance. Please reconsider submission if a substantive AI/ML research workflow and external validation become established.
