# Research Plan: Reasoning Trace Length as a Proxy for Robustness

## Motivation & Novelty Assessment

### Why This Research Matters
Chain-of-thought reasoning has become the default paradigm for improving LLM performance, yet we lack understanding of how trace length interacts with out-of-distribution robustness. Practitioners need guidance on whether longer reasoning helps or hurts when models encounter unfamiliar tasks. This directly impacts deployment decisions for reasoning-capable LLMs.

### Gap in Existing Work
Based on the literature review: Wu et al. (2025) established the inverted U-curve for accuracy vs trace length; Yang et al. (2025) showed training-free length reduction degrades behavioral consistency; L1 demonstrated RL-trained length control generalizes OOD. **However, no paper systematically measures OOD robustness as a function of controlled trace length across multiple domains.** The gap: we know length affects accuracy (ID), but not how the length-accuracy relationship shifts when moving OOD.

### Our Novel Contribution
We systematically vary reasoning trace length in state-of-the-art LLMs using budget-forcing prompts and measure: (1) the accuracy vs trace length curve on in-distribution vs OOD benchmarks, (2) the "robustness gap" (ID-OOD accuracy drop) as a function of trace length, (3) whether uncertainty signals (model-expressed confidence) correlate with optimal trace length, and (4) whether adaptive length selection outperforms fixed budgets for OOD generalization.

### Experiment Justification
- **Experiment 1 (Length-Accuracy Curves):** Establishes baseline inverted U-curves on ID and OOD tasks. Needed to confirm the phenomenon holds across domains, not just math.
- **Experiment 2 (Robustness Gap Analysis):** Tests our core hypothesis—that robustness gap varies non-monotonically with trace length. Novel measurement.
- **Experiment 3 (Uncertainty-Length Correlation):** Tests whether model uncertainty signals can predict when longer traces help vs hurt. Explores adaptive potential.

## Research Question
Does a non-monotonic relationship exist between reasoning trace length and OOD robustness? Can moderate trace lengths maximize generalization, and can uncertainty signals guide adaptive length selection?

## Hypothesis Decomposition
H1: Accuracy vs trace length follows an inverted U-curve on both ID and OOD tasks, but the peak shifts.
H2: The robustness gap (ID-OOD accuracy drop) is minimized at moderate trace lengths, not at the longest or shortest.
H3: Model-expressed confidence correlates with whether a given trace length is near-optimal for a particular problem.

## Proposed Methodology

### Approach
Use real LLM APIs (GPT-4.1 via OpenAI) with budget-forcing prompts to control reasoning trace length at 5 levels. Evaluate on 4 benchmarks spanning different domains and difficulties. Measure accuracy, trace length, confidence, and compute robustness gap.

### Experimental Steps
1. Sample balanced subsets from each dataset (~100 questions per dataset)
2. Design 5 budget-forcing prompt conditions: NoCoT, Short (~100 tok), Medium (~500 tok), Long (~1000 tok), Unconstrained
3. Query GPT-4.1 for each (question, budget) pair, extracting: answer, reasoning trace, token count, confidence
4. Score accuracy using dataset-specific answer matching
5. Compute accuracy curves, robustness gaps, and uncertainty correlations
6. Statistical analysis with bootstrap CIs and appropriate tests

### Baselines
- NoCoT (zero reasoning): lower bound
- Unconstrained CoT: upper bound / standard practice
- Fixed budgets at each level: tests fixed vs adaptive

### Evaluation Metrics
- Accuracy (exact match / multiple choice)
- Reasoning token count (actual)
- Robustness gap: Accuracy_ID - Accuracy_OOD
- Token efficiency: Accuracy / token_count
- Confidence calibration: ECE of model-expressed confidence

### Statistical Analysis Plan
- Bootstrap confidence intervals (N=1000) for all accuracy estimates
- Spearman correlation for length-accuracy and length-robustness relationships
- Paired t-tests (or Wilcoxon) for pairwise condition comparisons
- α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- Inverted U-curve for accuracy confirmed on ID tasks (replicating prior work)
- OOD curve shifted left (shorter optimal length OOD) or flattened
- Robustness gap minimized at moderate trace lengths
- Unconstrained traces show larger robustness gaps than moderate-length traces

## Timeline
- Phase 0-1 (Planning): 20 min ✓
- Phase 2 (Setup + Implementation): 45 min
- Phase 3-4 (Experiments): 90 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- API rate limits → use async with backoff
- Budget forcing imprecision → measure actual token counts, analyze by actual length bins
- Small sample sizes → bootstrap CIs, acknowledge power limitations
- Answer extraction difficulty → use structured output format

## Success Criteria
- Clear evidence for or against non-monotonic robustness gap
- Statistical significance on at least the primary comparison (moderate vs extreme lengths)
- Actionable insight about when longer reasoning helps vs hurts OOD
