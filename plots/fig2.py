"""Graph telegrapher trajectories below and above the slowest-mode threshold."""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

from plots.style import panel_label, apply_style, save_figure

apply_style()



def get_L(graph):
    return nx.laplacian_matrix(graph).astype(float).toarray()


def hyperbolic(L, tau, x0, v0, t_span=(0, 3), max_step=0.01):
    n = L.shape[0]

    def rhs(t, state):
        x, v = state[:n], state[n:]
        return np.concatenate([v, -(v + L @ x) / tau])

    solution = solve_ivp(rhs, t_span, np.concatenate([x0, v0]),
                         max_step=max_step)
    return solution.t, solution.y[:n]


np.random.seed(7)
n_nodes, degree = 40, 4
graph = nx.random_regular_graph(degree, n_nodes, seed=7)
L = get_L(graph)
lambda2 = np.linalg.eigvalsh(L)[1]
tau_c = 1 / (4 * lambda2)

x0 = np.random.randn(n_nodes)
v0 = np.zeros_like(x0)
times_low, trajectories_low = hyperbolic(L, tau_c / 3, x0, v0)
times_high, trajectories_high = hyperbolic(L, 3 * tau_c, x0, v0)

fig = plt.figure(figsize=(4.1, 2.0))
grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
fig.subplots_adjust(left=0.13, right=0.95, bottom=0.20, top=0.85)
# A consistent node-to-color assignment in both damping regimes.
colors = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, n_nodes))
# Mix hues across the drawing order without changing the simulation RNG.
colors = np.random.default_rng(7).permutation(colors)

for index, (times, trajectories, title) in enumerate((
    (times_low, trajectories_low, r"$\tau<\tau_c$"),
    (times_high, trajectories_high, r"$\tau>\tau_c$"),
)):
    ax = fig.add_subplot(grid[0, index])
    for values, color in zip(trajectories, colors):
        ax.plot(times, values, lw=1, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("$t$", fontsize=9, labelpad=-1.5)
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 1, 2] if index == 0 else [0, 1, 2, 3])
    ax.spines[["top", "right"]].set_visible(False)
    if index == 0:
        ax.set_yticks([-1.5, 0, 1.5])
        ax.set_ylabel(r"$x_i(t)$", labelpad=-5, fontsize=9)
    else:
        ax.set_yticks([])
    panel_label(ax, chr(97 + index))

output = Path(__file__).resolve().parents[1] / "figures/main"
output.mkdir(parents=True, exist_ok=True)
save_figure(fig, output / "fig2.pdf", 4.0)
plt.close(fig)
print(f"Figure 2: saved {output / 'fig2.pdf'}")
