# Data-provider submission review — 2026-10-24

This is a dated review of public primary sources, using the repository's existing [contribution criteria](../../CONTRIBUTING.md). Decisions below are recommendations; this file does not mean a PR was merged or a response sent. Service documentation establishes the advertised terms, not independent data-quality, point-in-time correctness, uptime, or investment-performance validation. No vendor software was installed, no accounts or keys were created, and no paid access was accepted.

| Request | Reviewed PR head | Recommended disposition | Main reason |
| --- | --- | --- | --- |
| [#67 Aperiodic](https://github.com/leoncuhk/awesome-quant-ai/pull/67) | `245c5b53c64072ccb9b89c5546a29a61b9808fb0` | Hold for corrected access description | Fixed historical preview is distinct from the newly documented ongoing delayed public API; pricing in the PR is stale. |
| [#68 Equibles](https://github.com/leoncuhk/awesome-quant-ai/pull/68) | `ba4be29da048498d09439805f83e5873d0482364` | Accept with wording edits | Self-hostable data backend plus a documented recurring free hosted quota; distinguish cloud-only datasets. |
| [#69 Tapetide MCP](https://github.com/leoncuhk/awesome-quant-ai/pull/69) | `8c46d75c6107d9366459bbe1aaafe42f5653cc3d` | Accept with wording edits | Adds Indian-market research data; current free limits differ from the PR. |
| [#71 roic.ai](https://github.com/leoncuhk/awesome-quant-ai/pull/71) | `42c365dd0ec3aad43b88598030a49135ce16c29f` | Accept with wording edits | Global standardized fundamentals and ratios; spell out free endpoint/history restrictions. |
| DataSinking outreach | No PR reviewed | Hold; invite a normal evidence-backed PR | Plausible cross-market filing-text use case and documented free tier; apply the same public submission process. |

## Aperiodic — #67

The [official agent reference](https://aperiodic.io/llms.txt) distinguishes three products: a fixed Binance BTC-perpetual metrics preview (5-minute bars, May 1–31, 2025); a different June 2025 raw-trade preview; and an ongoing public board/series API with 15-minute delay, 5-minute bars and up to 24 hours of rolling history. The latter is more relevant to the persistent-free-access criterion than a permanently downloadable sample. The [pricing page](https://aperiodic.io/pricing) starts at $199/month billed annually, rather than the PR's $79. Prices should be omitted from the entry to avoid another unnecessary moving target.

The [MIT CLI](https://github.com/aperiodic-io/cli) is a client, not an open-source data backend. Its [HTTP implementation](https://github.com/aperiodic-io/cli/blob/main/client.go) requests hosted data and presigned files. Do not describe the proprietary metrics computation or the claimed absence of look-ahead bias as independently verified.

Verification limit: the documented public series URL was attempted once without credentials using a standard-library HTTP client and returned HTTP 403 from this environment; the web fetch also failed. This does not establish global unavailability, but it leaves the current free API unverified. Ask for a working public example and any access/rate/licence restrictions before accepting revised wording. The old sample alone should not be silently reclassified as a general free API tier.

Possible row after that clarification:

```markdown
| **[Aperiodic](https://aperiodic.io/)** | Crypto perpetual-futures microstructure, liquidity and order-flow metrics through REST, a CLI and a Python SDK | Feature exploration for microstructure research. Documented no-key public API: 15-minute delay, 5-minute bars and up to 24 hours of rolling history; a fixed historical preview is separate, and longer history is paid. |
```

Draft response:

> Thanks for the submission and affiliation disclosure. The microstructure focus is relevant, but the access description needs updating. Your current agent reference documents both a fixed historical preview and an ongoing delayed public board/series API; those are different access models. Please base the entry on the latter, state its rolling-history, rate and usage restrictions, and provide one working no-key request. Our bounded request returned 403, which may be environment-specific. Please also remove the stale $79 price. We can complete the review once these details are clear.

## Equibles — #68

The [official MCP page](https://equibles.com/mcp) documents 100 requests per day shared between MCP and REST, with no credit card. The [pricing page](https://equibles.com/pricing), initially unavailable, was readable during the independent final review and confirmed that shared quota; its free prices are end-of-day, released four hours after market close. OAuth and API-key access are documented on the MCP page. No authenticated call was exercised.

The [AGPL-3.0 repository](https://github.com/daniel3303/Equibles) contains a self-hostable scraper/database/MCP stack rather than merely a transport wrapper. Its README explicitly assigns earnings-call transcripts, real-time quotes and options chains to Equibles Cloud. At upstream revision `034135e0ce590336fc40860ffc66b683adc3a580`, the [financial-statement tool](https://github.com/daniel3303/Equibles/blob/034135e0ce590336fc40860ffc66b683adc3a580/src/Equibles.Sec.FinancialFacts.Mcp/Tools/FinancialStatementTools.cs) queries structured facts, while the [Yahoo worker](https://github.com/daniel3303/Equibles/blob/034135e0ce590336fc40860ffc66b683adc3a580/src/Equibles.Yahoo.HostedService/YahooPriceScraperWorker.cs) imports prices. The statement tool documents latest-restated values: this is not, by itself, proof of a point-in-time backtest dataset.

The outreach claim of having every Alpha Vantage feature is unverified. No endpoint-by-endpoint, coverage, delay, licence or reliability comparison was performed. The README should describe Equibles on its own merits and retain Alpha Vantage as a separate resource.

Recommended row:

```markdown
| **[Equibles](https://equibles.com)** | SEC filings, XBRL fundamentals, institutional holdings and insider trades via REST and MCP, with an AGPL-3.0 self-hostable data core | Source-linked filing and ownership research for LLM workflows. Hosted free tier: 100 requests/day shared across REST and MCP, no credit card; transcripts, live quotes and options are cloud-only. |
```

Draft response:

> Thanks for the PR and clear affiliation disclosure. Equibles fits through its self-hostable data core and documented recurring free quota. The listing should distinguish the AGPL core from cloud-only transcripts, live quotes and options, and state that the 100 daily requests are shared across REST and MCP. We will describe those verified access boundaries rather than repeat an unverified feature-parity or price comparison with Alpha Vantage.

## Tapetide MCP — #69

The [official product reference](https://tapetide.com/mcp) and [connection/tool documentation](https://mcp.tapetide.com/) now state 50 successful calls per day with **no monthly cap**, rather than the PR's 1,000/month. The product reference distinguishes latest available quotes from a live trading feed and identifies option chains as end-of-day snapshots. Its coverage and screening counts vary slightly between first-party pages; omit those counts.

The [MIT repository](https://github.com/Tapetide-hq/nse-bse-indian-stock-market-data-mcp) is an open client, not a self-hostable market-data backend. The [source](https://github.com/Tapetide-hq/nse-bse-indian-stock-market-data-mcp/blob/main/src/index.ts) forwards requests to the hosted service. The inclusion case is Indian-market coverage useful to research agents plus ongoing free access; a generic MCP wrapper alone would not be enough. Authentication and dataset correctness were not exercised.

Recommended row:

```markdown
| **[Tapetide MCP](https://github.com/Tapetide-hq/nse-bse-indian-stock-market-data-mcp)** | Hosted MCP access to NSE/BSE financials, shareholding, FII/DII flows, screening and company filings | Indian-market research for AI agents. MIT client; free account: 50 successful tool calls/day, no monthly cap; data freshness varies and index option chains are end-of-day. |
```

Draft response:

> Thanks for the submission and affiliation disclosure. Indian-market financials, ownership and institutional flows add useful coverage to this section. The PR's monthly limit is now stale: your current official pages state 50 successful calls per day with no monthly cap. Please use that wording, distinguish the MIT client from the hosted backend, and avoid implying a live trading feed; your documentation describes varying freshness and end-of-day option chains.

## roic.ai — #71

The [official API overview](https://www.roic.ai/api) documents global public-company fundamentals, precomputed ratios, daily prices and transcripts with a free key and no credit card. The [pricing matrix](https://www.roic.ai/pricing) specifies 5 requests/minute; two years for annual financials, ratios and prices; **no quarterly financials**; and **two quarters** of transcripts. The PR's unqualified “2 years of history” obscures those restrictions.

The [MIT MCP server](https://github.com/roicai/agents-plugin) is a client of the hosted financial API, not an open-source data collection backend. Public documentation differs on tool/company totals; leave those counts out. No authenticated data call or cross-provider data-quality comparison was performed.

Recommended row:

```markdown
| **[roic.ai](https://www.roic.ai/api)** | Global company financial statements, pre-calculated ratios, daily prices and earnings-call transcripts via REST and MCP | Fundamental factor and earnings-call NLP research. Free API key: 5 requests/minute, 2 years of annual financials, ratios and prices, 2 quarters of transcripts; quarterly financials require a paid plan. |
```

Draft response:

> Thanks for the submission and affiliation disclosure. Standardized global fundamentals and ratios are useful for this list. Please make the free-tier boundaries explicit: your pricing matrix lists 5 requests/minute, two years of annual financials, ratios and prices, two quarters of transcripts, and no quarterly financials. We will omit changing coverage counts and use the API documentation as the primary link. The open-source client should not be described as an open-source data backend.

## DataSinking — outreach

The [public pricing page](https://datasink.ing/pricing) documents an ongoing free plan: 3 requests/second, 8,191 documents per rolling 7 days per key, batches up to three, inside a shared 524,287-document pool. The [API reference](https://datasink.ing/docs) documents full-report and section-level Markdown access across US, UK and Asian markets, which is a plausible filing-NLP/RAG differentiator. No API key was requested or data downloaded.

The [MIT repository](https://github.com/heubme2020/datasinking) contains examples, clients and MCP adapters; the [Python MCP implementation](https://github.com/heubme2020/datasinking/blob/main/datasinking/mcp_server.py) calls the hosted API. It does not establish that the parsing pipeline and corpus are open source. The [service terms](https://datasink.ing/terms) restrict standalone redistribution and resale; the MIT client licence does not grant unrestricted data rights. These restrictions should be disclosed in any future entry. Source fidelity, parsing accuracy and market coverage remain untested.

Recommended disposition: invite a normal PR with affiliation disclosure, free-tier terms, data-use restrictions, and representative source-to-Markdown examples. Do not accept a private benefit or add an affiliate URL as a reason to list it. Do not publish private correspondence, redemption codes or sender details in the repository.

Draft response:

> Thanks for reaching out. Cross-market filing text and section-level Markdown access could be relevant to financial NLP research. Please submit a PR through our contribution process, disclose your affiliation, state the public free-tier and redistribution restrictions, and link representative source-to-Markdown examples. Please distinguish the open-source client from the hosted parsing/data service. Any review will use the same public criteria as other submissions; complimentary access is not needed for a listing decision.
