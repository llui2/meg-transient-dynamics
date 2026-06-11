import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
IN_CSV = ROOT_DIR / "c3-tau.csv"
OUT_FIG = ROOT_DIR / "figures/fig_tau_100.pdf"
CAP_SIZE = 2
CAP_LINEWIDTH = 1.2
LINE_WIDTH = 1.5
FIT_SLOPE_IN_PLOT = True
SLOPE_SIGNIFICANT_DIGITS = 1
SLOPE_AS_INTEGER = False


def load_tau_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Tau CSV not found: {path}")

    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"Tau CSV is empty: {path}")

    def f64(key):
        return np.array([float(r[key]) for r in rows], dtype=float)

    data = {
        "roi": int(float(rows[0]["roi"])),
        "band": [r["band"] for r in rows],
        "fmin_hz": f64("fmin_hz"),
        "fmax_hz": f64("fmax_hz"),
        "freq_center_hz": f64("freq_center_hz"),
        "tau_mean_s": f64("tau_mean_s"),
        "tau_std_s": f64("tau_std_s"),
        "n_subjects": int(float(rows[0]["n_subjects"])),
        "tau_critical_s": float(rows[0]["tau_critical_s"]),
        "power_exp": float(rows[0]["power_exp"]),
    }
    return data


def fit_power_slope(freqs, tau_vals):
    positive = (freqs > 0) & (tau_vals > 0)
    if positive.sum() < 2:
        return 0.0
    log_f = np.log(freqs[positive])
    log_tau = np.log(tau_vals[positive])
    slope, _ = np.polyfit(log_f, log_tau, 1)
    return float(slope)


def round_to_significant_digits(value, digits):
    if value == 0:
        return 0.0
    if digits <= 0:
        return value
    return round(value, digits - int(np.floor(np.log10(abs(value)))) - 1)


def slope_reference_segment(freqs, tau_vals, slope):
    positive = (freqs > 0) & (tau_vals > 0)
    if positive.sum() < 2:
        return np.array([]), np.array([])

    f_valid = freqs[positive]
    t_valid = tau_vals[positive]

    # Draw a short reference line toward the bottom-right of the plot.
    f1 = np.percentile(f_valid, 58)
    f2 = np.percentile(f_valid, 78)
    f_anchor = np.median(f_valid)
    tau_anchor = np.percentile(t_valid, 68)

    ref_f = np.array([f1, f2], dtype=float)
    ref_tau = tau_anchor * (ref_f / f_anchor) ** slope
    return ref_f, ref_tau


def plot_tau(data, out_fig):
    freq = data["freq_center_hz"]
    tau_mean = data["tau_mean_s"]
    tau_std = data["tau_std_s"]
    tau_c = data["tau_critical_s"]
    power_exp_csv = data["power_exp"]

    # Ignore the last (highest-frequency) point for visualization.
    freq_plot = freq[:-1]
    tau_mean_plot = tau_mean[:-1]
    tau_std_plot = tau_std[:-1]

    # Fit slope directly from the plotted points if requested.
    if FIT_SLOPE_IN_PLOT:
        slope_raw = fit_power_slope(freq_plot, tau_mean_plot)
    else:
        slope_raw = power_exp_csv

    if SLOPE_AS_INTEGER:
        slope_used = float(np.round(slope_raw))
    else:
        slope_used = float(round_to_significant_digits(slope_raw, SLOPE_SIGNIFICANT_DIGITS))

    fig, ax = plt.subplots(figsize=(4, 2.6), dpi=300)
    color_tau = "#0400ff"
    color_crit = "#555555"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.min(data["fmin_hz"]), np.max(data["fmax_hz"]))

    # Use fixed frequency ticks requested for readability.
    x_ticks = [4, 8, 13, 30, 80]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks])

    ax.axhline(tau_c, color=color_crit, linestyle="--", linewidth=1, label=r"$\tau_c$ (structural)")

    tau_label = r"$\tau$ (fitted)"
    ax.errorbar(
        freq_plot,
        tau_mean_plot,
        yerr=tau_std_plot,
        color=color_tau,
        fmt="none",
        capsize=CAP_SIZE,
        elinewidth=LINE_WIDTH,
        capthick=CAP_LINEWIDTH,
        label=tau_label,
    )

    ref_f, ref_tau = slope_reference_segment(freq_plot, tau_mean_plot, slope_used)
    if ref_f.size:
        ax.plot(
            ref_f,
            ref_tau,
            color=color_tau,
            linestyle="-",
            linewidth=LINE_WIDTH,
        )
        k_mag = abs(slope_used)
        x_txt = np.sqrt(ref_f[0] * ref_f[1])
        y_txt = np.sqrt(ref_tau[0] * ref_tau[1]) * 1.04
        ax.text(x_txt, y_txt, rf"$\propto f^{{-{k_mag:.{SLOPE_SIGNIFICANT_DIGITS - 1}f}}}$", color=color_tau, fontsize=12)

    ax.set_xlabel("Frequency $f$ (Hz)")
    ax.set_ylabel(r"Relaxation Time $\tau$ (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"Slope raw = {slope_raw:.5f}, slope used = {slope_used:.5f}")
    print(f"Saved figure: {out_fig}")
    plt.close(fig)


plot_tau(load_tau_csv(IN_CSV), OUT_FIG)
