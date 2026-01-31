from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Monte Carlo simulation of a one-dimensional rubber band (freely-jointed chain).

The system consists of N links of fixed length a, each pointing either
in the + or - direction. The total extension L is the sum of all link
contributions. Both unbiased and force-biased ensembles are studied
and compared with analytic results from statistical mechanics.
"""

# --- Random setup ---
rng: np.random.Generator = np.random.default_rng(12345)


# -----------------------------
#  Classes
# -----------------------------
class Link:
    """
    Single link in a one-dimensional rubber band.

    Each link has fixed length `a` and a direction ±1.
    """

    a: float
    direction: int

    def __init__(self, a: float = 1.0, direction: int | None = None) -> None:
        """
        Create a link with given length and direction.

        If direction is None, it is sampled uniformly from {+1, -1}.
        """
        self.a = a
        if direction is None:
            self.direction = int(np.random.choice([-1, 1]))
        else:
            self.direction = direction

    def length_contribution(self) -> float:
        """
        Return the contribution of this link to the total extension L.
        """
        return self.a * self.direction


class RubberBand:
    """
    One-dimensional rubber band consisting of N independent links.

    The rubber band can be sampled either in an unbiased ensemble
    or in a force-biased ensemble using Boltzmann statistics.
    """

    N: int
    a: float
    T: float
    f: float
    links: List[Link]

    def __init__(self, N: int, a: float = 1.0, T: float = 1.0, biased: bool = False, f: float = 0.0) -> None:
        """
        Initialize a rubber band configuration.

        If biased=True, link directions are sampled from the
        force-dependent Boltzmann distribution.
        """
        self.N = N
        self.a = a
        self.T = T
        self.f = f

        if biased:
            beta: float = 1.0 / T
            p_plus: float = 1.0 / (1.0 + np.exp(-2.0 * beta * f * a))
            self.links = [Link(a=a) for _ in range(N)]
            for link in self.links:
                link.direction = 1 if rng.random() < p_plus else -1
        else:
            self.links = [Link(a=a) for _ in range(N)]

    def total_length(self) -> float:
        """
        Compute the total extension L of the rubber band.
        """
        return sum(link.length_contribution() for link in self.links)

    def boltzmann_weight(self, f: float) -> float:
        """
        Return the Boltzmann weight exp(beta * f * L).

        Used for reweighting unbiased samples to a force-biased ensemble.
        """
        beta: float = 1.0 / self.T
        L: float = self.total_length()
        return float(np.exp(beta * f * L))


# -----------------------------
#  Parameters
# -----------------------------
a: float = 1.0
N: int = 100
M: int = 100_000
T: float = 1.0
f: float = 0.1


# -----------------------------
# Simulation task I
# -----------------------------
"""
Unbiased Monte Carlo sampling of rubber-band configurations.

The histogram of total extensions L is compared with the analytic
binomial distribution.
"""
bands: List[RubberBand] = [RubberBand(N, a, T) for _ in range(M)]
L_values: np.ndarray = np.array([b.total_length() for b in bands])

bins: np.ndarray = np.arange(-N - 1, N + 2, 2)
hist, edges = np.histogram(L_values, bins=bins, density=True)

centers: np.ndarray = 0.5 * (edges[1:] + edges[:-1])
bin_width: float = float(edges[1] - edges[0])


# -----------------------------
# Analytic prediction
# -----------------------------
def analytic_P(L: float, N: int, a: float = 1.0) -> float:
    """
    Analytic probability P(L) for the unbiased ensemble.

    The number of microstates is Ω(L) = C(N, n),
    where n = (L/a + N)/2.
    """
    n: int = int((L / a + N) / 2)
    if n < 0 or n > N:
        return 0.0
    return math.comb(N, n) / (2.0 ** N)


L_theory: np.ndarray = np.arange(-N, N + 1, 2)
P_theory: np.ndarray = np.array([analytic_P(L, N, a) for L in L_theory])
P_theory = P_theory / np.sum(P_theory) / bin_width


# -----------------------------
# χ² test
# -----------------------------
"""
Chi-squared goodness-of-fit test between Monte Carlo
histogram and analytic prediction.
"""
ratio: np.ndarray = hist / P_theory
counts, _ = np.histogram(L_values, bins=bins)
theory_counts: np.ndarray = M * P_theory * bin_width

chi2: float = 0.0
ndf: int = 0

for c, t in zip(counts, theory_counts):
    if t > 5:
        chi2 += (c - t) ** 2 / t
        ndf += 1

ndf -= 1
chi2_ndf: float = chi2 / ndf if ndf > 0 else float("nan")


# -----------------------------
# Plotting
# -----------------------------
fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[2, 1], figsize=(8, 6))

axs[0].bar(centers, hist, width=1.8, alpha=0.6, color="skyblue", edgecolor="black", label="Monte Carlo")
axs[0].plot(L_theory, P_theory, "r-", lw=2, label="Analytic")
axs[0].legend()
axs[0].set_ylabel("P(L)")
axs[0].set_xlabel("L")
axs[0].set_title(f"P(L) for N={N}")
axs[0].text(0.05, 0.9, rf"$\chi^2$/ndf = {chi2_ndf:.3f}", transform=axs[0].transAxes)

axs[1].step(centers, ratio, where="mid")
axs[1].axhline(1.0, color="black")
axs[1].set_ylim(0.8, 1.2)
axs[1].set_ylabel("MC / Analytic")

plt.tight_layout()
plt.show()


# -----------------------------
# Task II: reweighting
# -----------------------------
"""
Reweighting unbiased samples to obtain force-biased distributions.
"""
weights: np.ndarray = np.array([b.boltzmann_weight(f) for b in bands])
hist_bised, edges = np.histogram(L_values, bins=bins, weights=weights, density=True)
centers = 0.5 * (edges[1:] + edges[:-1])


def analytic_P_bised(L: float, N: int, a: float = 1.0, T: float = 1.0, f: float = 0.0) -> float:
    """
    Analytic probability P_f(L) in the presence of an external force f.

    Defined as:
        P_f(L) = Ω(L) exp(beta f L) / Z
    where Z is the partition function.
    """
    beta: float = 1.0 / T
    L_vals: np.ndarray = np.arange(-N, N + 1, 2)

    Z: float = 0.0
    for Lp in L_vals:
        nLp: int = int((Lp / a + N) / 2)
        if 0 <= nLp <= N:
            Z += math.comb(N, nLp) * math.exp(beta * f * Lp)

    n: int = int((L / a + N) / 2)
    return math.comb(N, n) * math.exp(beta * f * L) / Z


# -----------------------------
# Task III: direct biased sampling
# -----------------------------
def mean_extension(M: int, N: int, a: float, T: float, f_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean extension ⟨L⟩ and standard deviation as a function of force.

    Uses direct sampling from the force-biased ensemble.
    """
    f_values: np.ndarray = np.linspace(0.0, f_max, 50)
    mean_L: List[float] = []
    std_L: List[float] = []

    for f in f_values:
        bands = [RubberBand(N, a, T, biased=True, f=f) for _ in range(M)]
        Ls: np.ndarray = np.array([b.total_length() for b in bands])
        mean_L.append(float(np.mean(Ls)))
        std_L.append(float(np.std(Ls, ddof=1)))

    return f_values, np.array(mean_L), np.array(std_L)
