# Rubber Band Monte Carlo (1D Freely-Jointed Chain)

Monte Carlo simulation of a one-dimensional freely-jointed chain (rubber band).

The model consists of **N links** of fixed length `a`, each pointing in either the `+` or `-` direction.

This script studies:
- **Unbiased sampling** of configurations and a comparison to the **analytic binomial distribution**
- A **chi-squared goodness-of-fit test** between Monte Carlo counts and the analytic prediction
- A **force-biased ensemble** using
  - **reweighting** of unbiased samples by Boltzmann weights
  - **direct biased sampling** using the exact single-link probability

## What the code does

1. Generates `M` independent chains with random link directions (unbiased ensemble)
2. Builds a histogram of total extension `L`
3. Compares the histogram to the analytic prediction \( P(L) \propto \Omega(L) \), where \( \Omega(L) = \binom{N}{n} \)
4. Computes a chi-squared / ndf statistic
5. Reweights samples to a biased ensemble under external force `f` using weights \( \exp(\beta f L) \)
6. Provides a function to compute mean extension versus force using direct biased sampling

## Requirements

- Python 3.13 (3.9+ likely fine)
- `numpy`
- `matplotlib`

Install:
```bash
pip install numpy matplotlib
