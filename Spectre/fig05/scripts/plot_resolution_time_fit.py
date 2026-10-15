from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ARTIFACT_ROOT))
from figure_style import LINE_WIDTHS, apply_arial_style, manuscript_width
FIG_ROOT = SCRIPT_DIR.parent
DATA_DIR = FIG_ROOT / "data" / "raw"
DERIVED_DIR = FIG_ROOT / "data" / "derived"
FIGURE_DIR = FIG_ROOT / "figure"
PANEL_PDF = "fig05_resolution_time_fit.pdf"
PARAMETER_CSV = "metastability_resolution_fit_parameters.csv"


COLORS = ["#0072B2", "#D55E00", "#009E73"]
VOLTAGES = [0.9, 1.1, 1.3]
VOLTAGE_LABELS = ["0.9 V", "1.1 V", "1.3 V"]
# The Spectre export fixes the abs(VA-VB) crossing threshold at 0.81 V for
# every supply corner; preserving that setting avoids fitting a different
# observation rule from the one that produced the data.
OBSERVATION_THRESHOLD_V = 0.81


def log_cosh(x):
    abs_x = np.abs(x)
    return abs_x + np.log1p(np.exp(-2 * abs_x)) - np.log(2)


def metastability_pdf(t, theta_reg, mu, sigma_eff, v_th):
    """Metastability resolution-time PDF, with t in seconds and density in 1/s."""
    sigma_eff = max(float(sigma_eff), 1e-12)
    exp_theta_t = np.exp(np.clip(-theta_reg * t, -700, 700))
    coeff = np.sqrt(2 / np.pi) * (theta_reg * v_th / sigma_eff)
    exp_arg = -theta_reg * t - (v_th**2 * exp_theta_t**2 + mu**2) / (2 * sigma_eff**2)
    cosh_arg = (mu * v_th * exp_theta_t) / (sigma_eff**2)
    log_f = np.log(coeff) + exp_arg + log_cosh(cosh_arg)
    return np.exp(np.clip(log_f, -745, 700))


def metastability_pdf_ps(t_ps, theta_reg, mu, sigma_eff, v_th):
    """Same PDF on a ps x-axis, with density in 1/ps."""
    return metastability_pdf(t_ps * 1e-12, theta_reg, mu, sigma_eff, v_th) * 1e-12


def save_paper_figure(fig, filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = FIGURE_DIR / filename
    fig.savefig(local_path)
    return local_path


def load_data():
    df = pd.read_csv(DATA_DIR / "VA.txt", skiprows=1, header=None, engine="python")
    df.columns = ["cycle", "time_09v", "time_11v", "time_13v"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    cycle = df["cycle"].to_numpy()
    offset = cycle * 1e-9 - 5e-10
    data_ps = [
        (df["time_09v"].to_numpy() - offset) * 1e12,
        (df["time_11v"].to_numpy() - offset) * 1e12,
        (df["time_13v"].to_numpy() - offset) * 1e12,
    ]
    return data_ps


def fit_one_voltage(t_ps, v_th, bins=50):
    counts, bin_edges = np.histogram(t_ps, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = np.isfinite(counts) & (counts > 0)
    bin_centers = bin_centers[mask]
    counts = counts[mask]

    if len(bin_centers) < 10:
        raise ValueError("Not enough nonempty histogram bins for fitting")

    theta_bounds = (1e9, 2e11)
    mu_bounds = (1e-6, 0.1)
    sigma_bounds = (1e-6, 0.1)
    lower = np.log([theta_bounds[0], mu_bounds[0], sigma_bounds[0]])
    upper = np.log([theta_bounds[1], mu_bounds[1], sigma_bounds[1]])

    mode_ps = float(bin_centers[np.argmax(counts)])
    y_peak = max(float(np.max(counts)), 1e-30)
    log_floor = y_peak * 1e-8

    starts = [[3e10, 1e-3, 2e-3]]
    for mu0 in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        theta0 = np.log(v_th / mu0) / max(mode_ps * 1e-12, 1e-15)
        theta0 = np.clip(theta0, *theta_bounds)
        for sigma0 in [3e-4, 7e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
            starts.append([theta0, mu0, sigma0])

    def residual(log_params):
        theta_reg, mu, sigma_eff = np.exp(log_params)
        y_model = metastability_pdf_ps(bin_centers, theta_reg, mu, sigma_eff, v_th)
        if not np.all(np.isfinite(y_model)) or np.max(y_model) <= 0:
            return np.full_like(counts, 1e6)
        return np.log(y_model + log_floor) - np.log(counts + log_floor)

    best_result = None
    best_cost = np.inf
    for start in starts:
        result = least_squares(
            residual,
            np.log(start),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.2,
            max_nfev=5000,
        )
        if result.cost < best_cost:
            best_result = result
            best_cost = result.cost

    if best_result is None:
        raise RuntimeError("All initial points failed")

    popt = np.exp(best_result.x)
    try:
        dof = max(1, len(counts) - len(popt))
        residual_var = 2.0 * best_result.cost / dof
        pcov_log = np.linalg.pinv(best_result.jac.T @ best_result.jac) * residual_var
        perr = popt * np.sqrt(np.maximum(np.diag(pcov_log), 0))
    except np.linalg.LinAlgError:
        perr = np.full_like(popt, np.nan)
    y_model = metastability_pdf_ps(bin_centers, popt[0], popt[1], popt[2], v_th)
    ss_res = float(np.sum((counts - y_model) ** 2))
    ss_tot = float(np.sum((counts - np.mean(counts)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, perr, r_squared


def fit_all(data_ps):
    rows = []
    fit_results = []
    for label, v_dd, t_ps in zip(VOLTAGE_LABELS, VOLTAGES, data_ps):
        v_th = OBSERVATION_THRESHOLD_V
        print(f"\nFitting {label} (V_T = {v_th:.3f} V)")
        print(f"  range min/mean/max = {np.min(t_ps):.2f}/{np.mean(t_ps):.2f}/{np.max(t_ps):.2f} ps")
        try:
            popt, perr, r_squared = fit_one_voltage(t_ps, v_th)
            theta, mu, sigma = popt
            theta_err, mu_err, sigma_err = perr
            fit_results.append((theta, mu, sigma, v_th, r_squared, theta_err, mu_err, sigma_err))
            rows.append(
                {
                    "voltage_v": v_dd,
                    "v_t_v": v_th,
                    "theta_reg_s_inv": theta,
                    "theta_reg_s_inv_se": theta_err,
                    "mu_v": mu,
                    "mu_v_se": mu_err,
                    "sigma_eff_v": sigma,
                    "sigma_eff_v_se": sigma_err,
                    "r_squared_histogram_density": r_squared,
                }
            )
            print(
                f"  theta_reg={theta:.3e} s^-1, mu={mu * 1000:.3f} mV, "
                f"sigma_eff={sigma * 1000:.3f} mV, R^2={r_squared:.6f}"
            )
        except Exception as exc:
            print(f"  fit failed: {exc}")
            fit_results.append((np.nan, np.nan, np.nan, v_th, np.nan, np.nan, np.nan, np.nan))
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DERIVED_DIR / PARAMETER_CSV, index=False)
    return fit_results


def plot_panels(data_ps, fit_results):
    apply_arial_style()
    fig, axes = plt.subplots(1, 3, figsize=(manuscript_width(0.85), 1.65), sharey=True)
    for idx, (ax, t_ps, label, color, result) in enumerate(zip(axes, data_ps, VOLTAGE_LABELS, COLORS, fit_results)):
        theta, mu, sigma, v_th, r_squared, *_ = result
        ax.hist(t_ps, bins=50, density=True, alpha=0.32, color=color, edgecolor="none", label="Simulated")
        if not np.isnan(theta):
            t_plot = np.linspace(float(np.min(t_ps)), float(np.max(t_ps)), 700)
            y_fit = metastability_pdf_ps(t_plot, theta, mu, sigma, v_th)
            ax.plot(t_plot, y_fit, color="black", lw=LINE_WIDTHS["primary"], label="Model fit")
        ax.text(0.03, 0.94, f"({chr(97 + idx)}) {label}", transform=ax.transAxes, va="top", ha="left")
        ax.text(
            0.96,
            0.08,
            rf"$R^2 = {r_squared:.4f}$",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
        )
        ax.legend(
            loc="upper right",
            fontsize=5.5,
            handlelength=1.2,
            handletextpad=0.3,
            borderpad=0.2,
            labelspacing=0.15,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            fancybox=False,
        )
        ax.set_xlabel("Resolution time (ps)")
        if idx == 0:
            ax.set_ylabel("Density (1/ps)")
        ax.grid(True, linestyle=":", linewidth=LINE_WIDTHS["grid"], alpha=0.55)
        ax.tick_params(width=LINE_WIDTHS["axis"], length=3)
    # Preserve the manuscript figure canvas but enlarge the effective plotting
    # area to match Fig. 4's three-panel layout.
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.30, top=0.93, wspace=0.22)
    save_paper_figure(fig, PANEL_PDF)
    plt.close(fig)


def main():
    data_ps = load_data()
    fit_results = fit_all(data_ps)
    plot_panels(data_ps, fit_results)
    print(f"\nSaved {PANEL_PDF} and {PARAMETER_CSV}")


if __name__ == "__main__":
    main()
