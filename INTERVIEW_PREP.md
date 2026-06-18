# Permian Well Economics Engine — Interview Prep

## 30 Seconds

I built a full Permian Basin well economics platform from scratch — Arps decline curve fitting with AIC model selection, EUR estimation with P10/P50/P90 confidence bands, and a discounted cash flow engine that outputs PV10, IRR, payback period, F&D cost, and NPV per lateral foot. The technical decision I'm most proud of is defaulting to Modified Hyperbolic rather than pure Hyperbolic — Permian unconventional wells exhibit super-hyperbolic early transient flow where b exceeds one, and if you don't force a terminal switch to exponential decline, you overestimate EUR by 15 to 25 percent. That's a real mistake operators make.

---

## 60 Seconds

I built a production forecasting and capital efficiency platform for Permian Basin horizontal wells. The core engine fits Arps decline curves using AIC model selection — AIC penalizes parameter count so you don't overfit a three-parameter hyperbolic when a two-parameter exponential works equally well. The default model is Modified Hyperbolic, which matters a lot: Permian wells show super-hyperbolic early transient flow — b-factors above one — but that behavior can't persist forever. Without forcing a switch to exponential terminal decline when the instantaneous rate drops below roughly six percent annually, you overestimate EUR by 15 to 25 percent. That's not a software quirk — it's a reservoir physics constraint.

The economics engine runs a full monthly cash flow model: oil, gas, and NGL revenue streams each priced separately, with the Permian Midland basis differential baked in, full LOE, gathering and transport, production taxes, and D&C capex at month zero. What surprised me was the sensitivity asymmetry: a one million dollar change in D&C cost moves F&D more than a ten percent change in EUR, because D&C dominates the numerator. That's the kind of insight that changes how operators think about frac stage optimization versus simply drilling more wells.

---

## 120 Seconds

I built a Permian Basin well economics engine that implements the full upstream analyst workflow: decline curve fitting, EUR estimation, and discounted cash flow analysis. Let me walk through the key technical decisions.

On the production forecasting side, the engine tests all four Arps model families — exponential, hyperbolic, harmonic, and modified hyperbolic — and selects the best fit using AIC. Akaike Information Criterion adds a complexity penalty so you don't just pick the model with the most parameters. But the real technical decision was defaulting to Modified Hyperbolic. Permian unconventional wells exhibit super-hyperbolic early transient flow — b-factors above one — driven by hydraulic fracture dominance in early months. If you fit a pure hyperbolic and extrapolate it, the decline rate asymptotically approaches zero and you get wildly optimistic EUR estimates. The industry fix is to force a switch to exponential terminal decline when the instantaneous annual rate drops below a threshold, typically five to eight percent. I implemented that switch explicitly.

For uncertainty quantification, I built P10/P50/P90 confidence bands around the forecast using ±1.645 standard deviations on initial rate, calibrated to roughly ten percent qi uncertainty — consistent with how operators report proved versus possible reserves.

The economics model is a full monthly cash flow: oil revenue adjusted for Permian Midland basis differential, separate gas and NGL revenue streams, LOE, gathering and transport, production taxes at 4.6% of gross for Texas, and D&C capex as a month-zero outflow. Outputs are PV10 at the SEC-standard ten percent discount rate, IRR, payback period, F&D cost in dollars per BOE, and NPV per lateral foot — the capital efficiency metric operators actually use to rank drilling programs.

Two things surprised me. First, AIC almost always selects Modified Hyperbolic even when the b-factor is between zero and one, because the terminal switch improves late-life fit even for wells that aren't super-hyperbolic. Second, the sensitivity heatmap revealed that D&C cost has roughly five times more leverage on F&D cost than EUR changes of the same proportional magnitude. That's non-obvious until you build the model — it means operators should think about completions efficiency more than incremental EUR when optimizing capital programs.
