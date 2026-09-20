# Uncertainty-Driven Position Sizing

Notes on a design pattern that keeps reappearing under different names: the predictor must emit not only a forecast but how much it distrusts that forecast, and the sizing layer consumes the distrust directly. Written 2026-09 after reading CAST (ICDM 2026), which is the cleanest recent instance of it.

## The Pattern

Most published pipelines stop at the forecast. A model produces ŷ, a rule converts ŷ into a signal (sign, rank, decile), and sizing is bolted on afterwards as a constant, an equal weight, or a volatility target. The forecast's own reliability never enters the position.

The alternative:

```
predictor  →  (ŷ, σ̂)  →  controller  →  position
```

Two requirements, and both are load-bearing:

1. The predictor exports a **per-step, per-asset uncertainty** alongside the point forecast.
2. The controller's objective contains a term that is **increasing in σ̂ and in trade size**, so the optimizer shrinks exposure by itself when the model's own confidence falls.

Nothing in this loop detects regimes, classifies crises, or retrains. A shock widens σ̂ mechanically, the penalty grows, positions shrink. The risk response is a side effect of the estimator's arithmetic rather than a judgement call layered on top.

## Why This Is Not Volatility Targeting

The usual objection is that volatility targeting already does this. It does not — it answers a different question.

| | Volatility targeting | Uncertainty-driven sizing |
|---|---|---|
| Input | Realized or forecast **asset** volatility | Forecast **error** of the model |
| Question answered | How much does this asset move? | How wrong is my view likely to be? |
| Behaviour on a calm but unpredictable asset | Sizes up | Sizes down |
| Behaviour on a volatile asset the model tracks well | Sizes down | Can stay invested |
| Blind spot | Model degradation, distribution shift | Tail moves the model happens to be calm about |

They are close to orthogonal, and the second one is the one that fires during distribution shift, because a model that has drifted away from the data generating process produces wide intervals before it produces losses. The two combine cleanly — vol targeting sets the scale, uncertainty sets the conviction within it.

## The Objective

CAST's controller is the reference implementation. Over a horizon of L days, trades u, forecast price path P̄:

```
max_u   Σ u_{k+l} · ΔP̄_{k+l|k}  −  λ Σ |u_{k+l}| · ω_l
```

where ω_l is the l-step forecast standard deviation and λ is a fixed risk-aversion scalar. Absolute values are linearised with slack variables, so the whole thing stays a linear program solvable in sub-milliseconds; only the first step is executed, then the problem is re-solved at the next close.

Three properties worth stealing independently of the rest of the system:

- **ω grows with l.** Uncertainty compounds along the horizon, so the penalty prices distant trades more heavily than near ones without any hand-tuned decay.
- **L1 in trade size, not L2.** A variance penalty would give a QP and a dense solution. The `|u|·ω` form gives an LP whose solutions are sparse — it trades nothing at all when the drift does not clear the uncertainty hurdle. Abstention is a first-class outcome.
- **It is a robust-optimization counterpart, not a mean-variance objective.** Maximising a linear payoff under box uncertainty on the drift, `|Δp − ΔP̄| ≤ ω`, produces exactly this penalty as its worst case. That is the honest reading: λ is not a risk-aversion parameter in the utility sense, it is the width of the uncertainty set you are willing to defend against.

The horizon matters more than it looks. CAST's ablation collapses the controller to a single step and NASDAQ Sharpe goes from 0.52 to −0.27, MDD from 0.114 to 0.467. Re-planning over a path lets the controller decline a trade now because the uncertainty three days out makes the round trip unattractive; a myopic version cannot express that.

## Where σ̂ Comes From

The pattern is only as good as the uncertainty estimate, and most ML forecasters do not produce a usable one.

| Source | Online? | Calibrated? | Cost | Notes |
|---|---|---|---|---|
| Kalman / state-space posterior covariance | Yes | Under model assumptions only | Negligible | Free by construction; wrong if the state model is wrong |
| GARCH-family conditional variance | Yes | Asset volatility, not forecast error | Low | Answers the vol-targeting question, not this one |
| Quantile regression / pinball loss | Retrain | Marginal, not conditional | Medium | Gives intervals directly; widths often too flat across regimes |
| Conformal prediction | Yes, with online variants | Yes, distribution-free coverage | Low on top of any model | Strongest guarantee available; adaptive variants handle shift |
| Deep ensembles | Retrain | Underconfident to poorly calibrated | High | Disagreement is a proxy, not a probability |
| MC dropout | Inference-time | Poorly calibrated in practice | Medium | Cheap but the weakest of the group |

Two practical points. First, the Kalman route is popular in this pattern precisely because Σ_k arrives for free and updates every step — you get the uncertainty channel without adding a second model. The price is that the covariance is only meaningful if the linear-Gaussian state model is roughly right; it reports confidence about its own misspecified world. Second, **online conformal prediction is the most interesting thing to bolt onto an existing forecaster**, because it wraps any black box, needs no retraining, and its coverage guarantee survives distribution shift by construction. If we want to retrofit this pattern to a model we already run, that is the cheapest path.

Regardless of source, σ̂ should be checked for calibration before it drives money: bin predictions by forecast σ̂, measure realized error per bin, and confirm the relationship is monotone and roughly proportional. An uncalibrated σ̂ does not make sizing conservative, it makes it arbitrary.

## Relation to Kelly

The pattern is the same shrinkage that fractional Kelly encodes, arrived at from the control side rather than the growth-optimal side.

Full Kelly with known parameters sizes at f = μ/σ². With μ estimated rather than known, the growth-optimal fraction shrinks by the posterior variance of μ, which is why practitioners use half-Kelly or quarter-Kelly as a crude stand-in for estimation error. Uncertainty-driven sizing makes that shrinkage explicit, per-asset and per-step, instead of a constant haircut chosen by folklore. The difference from Kelly proper is the objective: Kelly maximises long-run log growth and tolerates deep drawdowns on the way; this pattern targets the drawdown directly and accepts a lower growth rate for it.

That trade is real and should not be glossed. CAST's reported drawdowns during COVID and the 2022 rate-hike cycle are 0.9%–10% against 15%–34% for buy-and-hold — but its 15-year annualized return is 0.8%–3.2%, far below passive exposure to the same markets, on a backtest with zero transaction costs and a zero risk-free rate. The mechanism controls drawdown. It does not produce alpha, and a system that is mostly in cash will post a low MDD whether or not the uncertainty channel is doing anything.

## Failure Modes

Collected from reading the CAST implementation, most of which generalize:

- **Low exposure masquerading as risk control.** If the penalty dominates, the system barely trades and every risk metric looks excellent. Always report average gross exposure alongside MDD; a drawdown of 3% on 5% average exposure is not risk management, it is abstention.
- **Hidden scale knobs on σ̂.** CAST's experiment script multiplies ω by a constant 1.5 that does not appear in the paper's equations. Any multiplier in front of the uncertainty term is a second λ, and it belongs in the parameter list.
- **λ selected on the evaluation window.** CAST's repo sweeps λ ∈ {0.05, 0.1, 0.3, 0.6} and keeps the best test-window Sharpe per market. The paper says λ is fixed before evaluation. Whichever is true, λ has to be chosen on calibration data or the reported risk-adjusted numbers are selection artifacts.
- **Frictionless backtests.** A controller that re-solves at every close and caps single trades at 50% of equity turns over aggressively. Zero-cost assumptions flatter this pattern more than they flatter a slow strategy.
- **Uncertainty that never widens.** A misspecified state model can report shrinking covariance while errors grow. The forecast-error decay test — normalize each model's error by its own first sub-period and plot across multi-year sub-periods — is a cheap check that the uncertainty channel is alive, and it doubles as a distribution-shift diagnostic for any forecaster.

## Minimal Recipe to Try This

On an existing forecaster, without rebuilding anything:

1. Wrap the model in online conformal prediction to get per-step intervals with coverage. Verify calibration by the binning check above.
2. Replace fixed or signal-proportional sizing with the LP above: linear expected-payoff term, `λ|u|·σ̂` penalty, existing exposure and per-trade caps as constraints.
3. Pick λ on the calibration window only. Record the full λ path as a sensitivity result, not as a selection step.
4. Report MDD, Calmar, average gross exposure, turnover, and results net of realistic costs. The first two alone can be bought with inactivity.
5. Ablate by setting λ = 0. If nothing much changes, the uncertainty channel is not driving anything and the complexity is not earning its place.

## References

- Peng, Khushi, Poon — *CAST: A Cross-Asset State-Space Trading System for Drawdown Control in Stock Markets*, ICDM 2026. [arXiv:2609.14205](https://arxiv.org/abs/2609.14205), [code](https://github.com/FanBroWell/CAST). Source of the objective discussed above; read the appendix in the repo for the account and constraint design.
- Gibbs and Candès — *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021. The online conformal variant relevant to step 1.
- MacLean, Thorp, Ziemba — *The Kelly Capital Growth Investment Criterion*, 2011. For the growth-optimal side of the shrinkage argument.
- See also `HMM Quantitative Trading Strategy An Overview.md` and `Markov-Switching Model Application.md` in this directory for the discrete-regime alternative — those detect a state and switch behaviour; this pattern never names a state and adjusts continuously.
