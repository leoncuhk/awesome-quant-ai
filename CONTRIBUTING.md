# Contributing to Awesome Quant AI

Thanks for taking the time to suggest a resource. This list is curated rather than comprehensive — the goal is to keep it useful for people doing serious AI/ML work in quantitative finance, not to be a directory of every tool that exists.

A few minutes spent reading the criteria below saves both of us time.

## What we look for

A good addition usually has at least two of the following:

- **Open source** with a public repository (any OSI-approved license)
- **Established traction** — meaningful star count, active maintenance, or visible adoption in research/industry
- **Academic or research origin** — a paper, lab, or recognized research group behind it
- **Genuine differentiation** — covers something existing entries don't, or does it materially better
- **Free or open API** for data providers: a persistent free tier (a free API key is fine) that works without payment, a trial, or a sales call. Pay-per-call-only access doesn't count. Please state free-tier limits in the entry

The existing entries are the bar. If your submission is in the same league as the projects already listed in that section, it's likely a fit.

## What usually doesn't fit

- **Commercial SaaS without an open-source component or significant third-party adoption** — especially if the value prop overlaps with what's already public (SEC EDGAR, FRED, DBnomics, established free APIs)
- **Brand-new repositories** (less than a few months old, low single-digit stars) without external validation
- **Marketing-style accuracy claims** (e.g. "X% backtest accuracy") without reproducible methodology, sample size, and out-of-sample evidence
- **Closed-source alert tools, signal services, or trading bots** with no public repo
- **Pump detection, signal copying, or anything optimizing for retail FOMO trading** — outside the scope of this list
- **Generic "AI for trading" wrappers** that don't add a research or engineering contribution beyond an LLM API call

## Self-submission

If you built it, that's fine — please **disclose the affiliation in the PR description**. Self-submitted commercial products are held to a higher bar (real traction, clear differentiation, or an open-source core). Third-party recommendations from active users are easier to accept.

Lower pricing, an MCP connector, or an offer of free access does not establish research value. Placement is not for sale; gifts, discounts, and reciprocal links do not influence inclusion. Disclose affiliate relationships relevant to the submitted resource.

## Evidence for a useful entry

- Link to the official documentation, repository/license, or paper supporting the description. Explain what a reader gains over the closest existing entry.
- For data services, link to public pricing/access terms and state the recurring free allowance, dataset coverage, history, delay, and material usage restrictions. Distinguish an ongoing free service from a fixed sample download, temporary trial, or maintainer-only offer.
- Distinguish an open-source client or MCP wrapper from an open-source backend and from rights to the underlying data. A connector alone does not demonstrate a quant research contribution.
- For research and agent systems, label preprints and author-reported results accurately. Link to evaluation data, baselines, costs, and out-of-sample methodology when making performance claims. Tests and demos do not establish investment performance.
- For related directories, show useful curation, source attribution, or maintenance information beyond republishing another list. Disclose sponsorship and affiliate links; avoid reciprocal-link requests.

Documentation review establishes what is publicly documented, not independent reproduction, data-quality certification, or production readiness. Descriptions should reflect that boundary and omit volatile promotional counts unless they help readers choose.

## How to submit

1. **One PR per resource** (or a small group of closely related ones).
2. **Add to the most relevant existing section** — don't create new sections without discussion.
3. **Match the existing format** in that section (table row, bullet, or paragraph).
4. **Description**: one sentence, ending with a period. Lead with what it is, then the differentiator.
5. **No duplicates** — search the README first.

PR description should briefly cover:
- What the project is
- Why it fits this list (and this specific section)
- Any affiliation disclosure
- Primary-source evidence and access limitations (where applicable)

## What to expect

Every submission gets a reply with the reason for the decision. The maintainer may lightly edit an entry before merging (placement, wording, or stating pricing and free-tier limits) to keep the list consistent. A decline is not permanent: if the project gains adoption, an open-source core, or a public methodology, please resubmit.

Please use a PR for a proposed entry, or an issue for a fit question. Email pitches are subject to the same criteria and do not receive priority. Existing entries can also be corrected or removed when their access terms, maintenance, or evidence change; previous inclusion is not an automatic exception for new submissions.

## Scope reminder

This list focuses on **AI/ML applied to quantitative finance** — research frameworks, foundation models, data sources useful for ML workflows, agentic trading systems, and the supporting infrastructure. General trading platforms, broker SDKs, and non-AI quant tools have their own awesome-lists; we link to those rather than duplicate them.

Thanks again — quality submissions are genuinely appreciated.
