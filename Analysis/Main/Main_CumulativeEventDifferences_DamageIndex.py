


#### -------
#### Imports
#### -------


# Standard library imports
import os
from pathlib import Path
import pandas as pd
import warnings
import re

# Third-party imports
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)




#### --------------
#### Sample Loading 
#### --------------


def collect_grouped_sample_data(base_dir: str, sample_groups: dict[str, list[int]]) -> pd.DataFrame:
    """
    Load and merge fission-fusion mechanism data with morphological data.

    Parameters:
        base_dir (str): Root path containing sample folders.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.

    Returns:
        pd.DataFrame: Merged dataframe containing mechanism events.
    """
    
    all_dfs = []

    for condition, sample_ids in sample_groups.items():
        for sid in sample_ids:
            mech_path = Path(base_dir) / str(sid) / "processed_data_analysis" / "fission_fusion_mechanisms.csv"
            morph_path = Path(base_dir) / str(sid) / "processed_data_analysis" / "general_morphological_features_per_object.csv"

            if not mech_path.exists() or not morph_path.exists():
                print(f"[WARN] Missing files for sample {sid} (condition {condition})")
                continue

            mech_df = pd.read_csv(mech_path)
            morph_df = pd.read_csv(morph_path)

            morph_gen = morph_df[["Label", "Time (Seconds)", "Region"]].rename(columns={"Region": "Generation Region"})
            morph_term = morph_df[["Label", "Time (Seconds)", "Region"]].rename(columns={"Region": "Termination Region"})

            mech_df = mech_df.merge(
                morph_gen,
                left_on=["Label", "Time of Generation (Seconds)"],
                right_on=["Label", "Time (Seconds)"],
                how="left"
            ).drop(columns=["Time (Seconds)"])
            mech_df = mech_df.merge(
                morph_term,
                left_on=["Label", "Time of Termination (Seconds)"],
                right_on=["Label", "Time (Seconds)"],
                how="left"
            ).drop(columns=["Time (Seconds)"])

            mech_df["SampleID"] = sid
            mech_df["Condition"] = condition
            all_dfs.append(mech_df)

    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def filter_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Filter mechanism events by spatial domain.

    Parameters:
        df (pd.DataFrame): Dataframe containing mechanisms and Generation/Termination Regions.
        domain (str): Domain string name to filter by.

    Returns:
        df (pd.DataFrame): Filtered dataframe containing only rows relevant to the specific domain.
    """

    if domain == "Global":
        return df
    elif domain == "Perinuclear (Combined)":
        return df[
            df["Generation Region"].isin(["Perinuclear", "Transition"]) |
            df["Termination Region"].isin(["Perinuclear", "Transition"])
        ]
    else:
        return df[
            df["Generation Region"].eq(domain) |
            df["Termination Region"].eq(domain)
        ]




#### -----------------
#### Fitting Functions 
#### -----------------


def linear(t, a, b):
    return a * t + b

def saturating_exp(t, K, k):
    return K * (1 - np.exp(-k * t))

def gompertz(t, K, b, t0):
    return K * np.exp(-np.exp(-b * (t - t0)))

def weibull_cdf(t, K, lam, k):
    return K * (1 - np.exp(-(t / lam) ** k))

def logistic_3pl(t, D, C, B):
    return D / (1.0 + (t / C) ** B)

def logistic_4pl(t, A, D, C, B):
    return A + (D - A) / (1.0 + (t / C) ** B)


def fit_and_score(func, t: np.ndarray, y: np.ndarray, p0: tuple | None = None, bounds: tuple = (-np.inf, np.inf), maxfev: int = 20000) -> dict[str, object]:
    """
    Fit a parameteric model to data and compute goodness-of-fit metrics.
    
    Parameters:
        func: Callable model function f(t, *params).
        t (np.ndarray): Time array.
        y (np.ndarray): Cumulative difference of fission-fusion event counts array.
        p0 (tuple): Initial guess for parameters.
        bounds (tuple): Lower and upper bounds for parameters.
        maxfev (int): Maximum number of function evaluations for fitting.

    Returns:
        Dictionary of fitted parameters and fit metrics (or an error message).
    """

    try:
        popt, pcov = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=maxfev)
        y_pred = func(t, *popt)
        resid = y - y_pred
        rss = np.sum(resid ** 2)
        n = len(y)
        k = len(popt)
        aic = 2 * k + n * np.log(rss / n + 1e-12)
        bic = k * np.log(n) + n * np.log(rss / n + 1e-12)
        rmse = np.sqrt(rss / n)
        return {'popt': popt, 'pcov': pcov, 'y_pred': y_pred, 'rss': rss, 'aic': aic, 'bic': bic, 'rmse': rmse}
    except Exception as e:
        return {'error': str(e)}




#### ----------------------------------------------------------------------------
#### Cumulative Event Difference (Fission - Fusion) Parametric Fitting & Plotting 
#### ----------------------------------------------------------------------------


def plot_combined_average_delta_curves_with_domains_and_parametric_fits(df: pd.DataFrame, sample_groups: dict[str, list[int]], palette: dict[str, str], output_dir: str = "figures/multi_domain_parametric", time_unit: str = "seconds") -> None:
    """
    Calculate and plot cumulative fission-fusion difference curves with multiple parametric fits.
    
    Parameters:
        df (pd.DataFrame): Mechanism dataframe from collect_group_sample_data.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.
        palette (dict): Dictionary mapping condition names to colors.
        output_dir (str): Directory to save figures.
        time_unit (str): Time unit for plotting (either 'seconds' or 'minutes').

    Returns:
        None
    """
    
    os.makedirs(output_dir, exist_ok=True)
    condition_order = list(sample_groups.keys())
    time_divisor = 60 if time_unit == "minutes" else 1
    time_label = "Time (min)" if time_unit == "minutes" else "Time (s)"

    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 4), sharey=True)

    fit_summary_rows = []
    param_names = {
            "linear": ["a", "b"],
            "saturating_exp": ["K", "k"],
            "gompertz": ["K", "b", "t0"],
            "weibull": ["K", "lambda", "k"],
            "3pl": ["D", "C", "B"],
            "4pl": ["A", "D", "C", "B"]
        }
    
    for ax, domain in zip(axes, domains):
        for condition in condition_order:
            # take only rows in this domain + condition
            domain_df = filter_by_domain(df[df["Condition"] == condition], domain)
            if domain_df.empty:
                continue

            base_color = palette.get(condition, 'gray')
            fission_samples, fusion_samples = [], []
            all_times_combined = []

            # per-sample cumulative construction
            for _, sample_df in domain_df.groupby('SampleID'):
                fission_df = sample_df[sample_df['Termination Mechanism'].str.contains('Fission', na=False)]
                fusion_df = sample_df[sample_df['Generation Mechanism'].str.contains('Fusion', na=False)]

                f_times = np.array(sorted(fission_df['Time of Termination (Seconds)'])) if len(fission_df) > 0 else np.array([])
                u_times = np.array(sorted(fusion_df['Time of Generation (Seconds)'])) if len(fusion_df) > 0 else np.array([])

                if f_times.size == 0 and u_times.size == 0:
                    continue

                all_times = np.sort(np.unique(np.concatenate([f_times, u_times])))
                all_times_combined.extend(all_times)

                if f_times.size > 0:
                    f_counts = pd.Series(f_times).value_counts().sort_index().cumsum()
                    f_vals = f_counts.reindex(all_times, method='ffill').fillna(0).values
                else:
                    f_vals = np.zeros_like(all_times, dtype=float)

                if u_times.size > 0:
                    u_counts = pd.Series(u_times).value_counts().sort_index().cumsum()
                    u_vals = u_counts.reindex(all_times, method='ffill').fillna(0).values
                else:
                    u_vals = np.zeros_like(all_times, dtype=float)

                f_interp = interp1d(all_times, f_vals, kind='previous', bounds_error=False,
                                    fill_value=(0, f_vals[-1] if f_vals.size else 0))
                u_interp = interp1d(all_times, u_vals, kind='previous', bounds_error=False,
                                    fill_value=(0, u_vals[-1] if u_vals.size else 0))
                fission_samples.append(f_interp)
                fusion_samples.append(u_interp)

            if not all_times_combined:
                continue

            # common time grid in seconds, then convert to t_plot using time_divisor
            common_times_seconds = np.linspace(0, max(all_times_combined), 500)
            t_plot = common_times_seconds / time_divisor

            f_mat = np.array([f(common_times_seconds) for f in fission_samples])
            u_mat = np.array([u(common_times_seconds) for u in fusion_samples])
            delta_mean = f_mat.mean(axis=0) - u_mat.mean(axis=0)
            delta_std = (f_mat - u_mat).std(axis=0)

            # mask out t==0 to avoid some fit issues (optional)
            mask = t_plot != 0
            t = t_plot[mask]
            y = delta_mean[mask]
            y_std = delta_std[mask]

            # plot mean +/- std on domain axis
            ax.plot(t, y, label=condition, color=base_color, linewidth=1.5)
            ax.fill_between(t, y - y_std, y + y_std, color=base_color, alpha=0.3)

            # model fits
            data_max, data_min = np.nanmax(y), np.nanmin(y)

            def r2(y_true, y_pred):
                ss_res = np.nansum((y_true - y_pred) ** 2)
                ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
                return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            fit_funcs = {
                "linear": (linear, ((y[-1] - y[0]) / (t[-1] - t[0] + 1e-12), y[0])),
                "saturating_exp": (saturating_exp, (max(data_max, 1.0), 0.01)),
                "gompertz": (gompertz, (max(data_max, 1.0), 0.1, np.median(t))),
                "weibull": (weibull_cdf, (max(data_max, 1.0), max(1.0, np.mean(t)), 1.5)),
                "3pl": (logistic_3pl, (max(data_max, 1.0), max(1.0, np.median(t)), 1.0)),
                "4pl": (logistic_4pl, (max(0.0, data_min), max(data_max, 1.0), max(1.0, np.median(t)), 1.0))
            }

            models = {}
            for name, (func, p0) in fit_funcs.items():
                try:
                    # light bounds to help stability for logistic params
                    if name == "3pl":
                        bounds = (0, [np.inf, np.inf, np.inf])
                    elif name == "4pl":
                        bounds = ([-np.inf, -np.inf, 1e-6, 0.01], [np.inf, np.inf, np.inf, np.inf])
                    else:
                        bounds = (-np.inf, np.inf)
                    res = fit_and_score(func, t, y, p0=p0, bounds=bounds)
                    if "y_pred" in res:
                        res["r2"] = r2(y, res["y_pred"])
                    models[name] = res
                except Exception as e:
                    models[name] = {"error": str(e)}

            # choose best by AIC if available
            valid_models = {k: v for k, v in models.items() if "aic" in v and not np.isnan(v["aic"])}
            chosen = min(valid_models, key=lambda m: valid_models[m]["aic"]) if valid_models else None

            # overlay chosen on the domain-level plot (solid main + dashed chosen)
            if chosen and "y_pred" in models[chosen]:
                ax.plot(t, models[chosen]["y_pred"], "--", color=base_color, linewidth=1.5,
                        label=f"{condition} ({chosen})")

            # save per-domain-per-condition comparison plot (all model overlays)
            plt.figure(figsize=(7, 4.5))
            plt.plot(t, y, "k", label="Data", linewidth=2)
            for mname, mres in models.items():
                if "y_pred" in mres:
                    lab = f"{mname} (R²={mres.get('r2', np.nan):.3f})"
                    plt.plot(t, mres["y_pred"], label=lab)
            plt.xlabel(time_label)
            plt.ylabel("Δ (Fission - Fusion)")
            plt.title(f"{condition} — {domain} — Model Fits")
            plt.legend(fontsize="small")
            plt.tight_layout()
            out_png = Path(output_dir) / f"{condition.replace(' ', '_')}_{domain.replace(' ', '_')}_all_model_fits.png"
            plt.savefig(out_png, dpi=300, transparent=True)
            plt.close()
            
            print(f"[INFO] Saved {out_png}")
            
            for name, res in models.items():
                base_row = {
                    "Domain": domain,
                    "Condition": condition,
                    "Model": name,
                    "AIC": res.get("aic", np.nan),
                    "RMSE": res.get("rmse", np.nan),
                    "R2": res.get("r2", np.nan),
                    "Chosen": bool(name == chosen),
                    "Params": "; ".join([f"{p:.4g}" for p in res.get("popt", [])]) if "popt" in res and isinstance(res.get("popt"), (list, np.ndarray)) and len(res.get("popt"))>0 else ""
                }

                # populate parameter-specific columns using the param_names mapping
                pnames = param_names.get(name, [])
                popt = res.get("popt", None)
                for i, pname in enumerate(pnames):
                    if isinstance(popt, (list, np.ndarray)) and len(popt) > i:
                        # format numbers succinctly, but keep them as strings so "N/A" can be used consistently
                        try:
                            base_row[pname] = f"{popt[i]:.6g}"
                        except Exception:
                            base_row[pname] = str(popt[i])
                    else:
                        base_row[pname] = "N/A"

                if isinstance(popt, (list, np.ndarray)) and len(popt) > len(pnames):
                    for j in range(len(pnames), len(popt)):
                        extra_name = f"p{j}"
                        try:
                            base_row[extra_name] = f"{popt[j]:.6g}"
                        except Exception:
                            base_row[extra_name] = str(popt[j])

                fit_summary_rows.append(base_row)

        ax.axhline(0, linestyle="--", color="gray")
        ax.tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)

    fig.tight_layout()
    combined_out = Path(output_dir) / "Combined_Avg_Delta_Curves_by_domain.png"
    fig.savefig(combined_out, dpi=300, transparent=True)
    plt.close(fig)

    print(f"[INFO] Saved combined multi-domain figure: {combined_out}")

    # Save CSV summary with separated parameter columns
    if fit_summary_rows:
        summary_df = pd.DataFrame(fit_summary_rows)

        # Collect all params from mapping and any dynamic p# columns present in rows
        mapped_params = set(sum([v for v in param_names.values()], []))
        # also pick up any 'p#' columns created dynamically
        dynamic_pcols = [c for c in summary_df.columns if re.fullmatch(r"p\d+", str(c))]
        all_param_cols = sorted(mapped_params.union(dynamic_pcols))

        # Make sure columns exist
        for c in all_param_cols:
            if c not in summary_df.columns:
                summary_df[c] = "N/A"

        # Reorder columns: keep main metadata first, then param columns, then Params compact string
        meta_cols = ["Domain", "Condition", "Model", "AIC", "RMSE", "R2", "Chosen"]
        rest = [c for c in summary_df.columns if c not in meta_cols + all_param_cols + ["Params"]]
        col_order = meta_cols + all_param_cols + ["Params"] + rest
        # keep only unique and existent columns
        col_order = [c for c in col_order if c in summary_df.columns]

        summary_df = summary_df[col_order].fillna("N/A")

        out_csv = Path(output_dir) / "fit_summary_extended_by_domain.csv"
        summary_df.to_csv(out_csv, index=False)

        print(f"[INFO] Saved fit summary CSV: {out_csv}")




#### ---------------------
#### Damage Index Plotting 
#### ---------------------


def plot_damage_index(csv_path: str, palette: dict[str, str], domain: str = "Global", model: str = "4pl", save_path: str = "Damage_Index.png") -> tuple[float, float, float, float]:
    """
    Compute and plot damage index derived from parameteric model parameters. Damage index is defined as (A-D)/C.

    Parameters:
        csv_path (str): Path to the fitting summary CSV file generated by the plot_combined_average_delta_curves_with_domains_and_parametric_fits function.
        palette (dict): Dictionary mapping condition names to colors.
        domain (str): Spatial domain to analyse.
        model (str): Parametric model name to use.
        save_path (str): Output path.

    Returns:
        tuple of floats containing (slope, y-intercept, r_squared, x-intercept)
    """

    # Load and Filter Data
    df = pd.read_csv(csv_path)
    four_pl_df = df[(df['Model'] == model) & (df['Domain'] == domain)]

    required_cols = ['A', 'D', 'C']
    if not all(col in four_pl_df.columns for col in required_cols):
        raise ValueError(f"Missing one of required columns: {required_cols}")

    for col in required_cols:
        four_pl_df[col] = pd.to_numeric(four_pl_df[col], errors='coerce')

    valid_models = four_pl_df.dropna(subset=required_cols).copy()


    # Damage Index Calculation
    valid_models['Condition_numeric'] = (valid_models['Condition'].str.extract(r'(\d+\.?\d*)').astype(float))
    valid_models['Damage Index'] = ((valid_models['A'] - valid_models['D']) / valid_models['C'])

    x_orig = valid_models['Condition_numeric'].values
    x_log = np.log10(x_orig)
    y = valid_models['Damage Index'].values


    # Linear Regression Fit
    slope, intercept, r_value, p_value, std_err = linregress(x_log, y)
    line = slope * x_log + intercept
    x_intercept = 10 ** (-intercept / slope)


    # Plot
    plt.figure(figsize=(6, 6))

    colors = valid_models['Condition'].map(palette)
    plt.scatter(x_orig, y, c=colors, s=75)

    plt.plot(
        x_orig,
        line,
        linestyle='--',
        color="#000000",
        label=f'Fit: y={slope:.3f}log10(x)+{intercept:.3f}'
    )

    plt.xscale('log')
    plt.minorticks_on()

    plt.tick_params(axis='x', which='both', direction='out', labelsize=14, width=1, length=6)
    plt.tick_params(axis='x', which='minor', length=3)
    plt.tick_params(axis='y', which='major', direction='out', labelsize=14, width=1, length=6)
    plt.tick_params(axis='y', which='minor', bottom=False, top=False, left=False, right=False)

    irradiation_ticks = [10, 50, 100, 200, 300]
    plt.xticks(irradiation_ticks, [str(t) for t in irradiation_ticks])

    plt.axhline(0, linestyle="--", color="gray", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()


    # Final Fit Reports for Damage Index
    print("\nLinear fit results:")
    print(f"  Slope:      {slope:.4f}")
    print(f"  Intercept:  {intercept:.4f}")
    print(f"  R-squared:  {r_value**2:.4f}")
    print(f"  Fit:        y = {slope:.3f}·log10(x) + {intercept:.3f}")
    print(f"  X-Int.:     {x_intercept}")

    return slope, intercept, r_value**2, x_intercept



if __name__ == "__main__":


    #### ------------
    #### General
    #### ------------

    sample_groups = {
        "9.6mW": [17, 18, 20],
        "27mW": [25, 26, 28],
        "128.7mW": [1, 2, 4],
        "252.3mW": [21, 22, 23]
    }

    fission_palette = {
        '9.6mW': '#FAA3FA',
        '27mW': '#E15EF9',
        '128.7mW': '#8303C8',
        '252.3mW': '#330084'
    }

    fusion_palette = {
        '9.6mW': '#A3FAA3',
        '27mW': '#76F95E',
        '128.7mW': '#48C803',
        '252.3mW': '#518400'
    }

    domains = ["Global", "Telenuclear", "Perinuclear (Combined)"]



    #### ---------------------------------------------------------------
    #### Cumulative Event Difference Curves (Fission - Fusion) with Fits
    #### ---------------------------------------------------------------

    mechanism_df = collect_grouped_sample_data(base_dir="processed_outputs_60secbin", sample_groups=sample_groups)

    plot_combined_average_delta_curves_with_domains_and_parametric_fits(
        mechanism_df,
        sample_groups=sample_groups,
        palette=fission_palette,
        output_dir="figures/multi_domain_parametric_test",
        time_unit="minutes"
    )



    #### ------------
    #### Damage Index
    #### ------------

    plot_damage_index(
        csv_path="figures/multi_domain_parametric_test/fit_summary_extended_by_domain.csv",
        palette=fission_palette,
        domain="Global", 
        model="4pl", 
        save_path="3C_Damage_Index_Photoirradiation.png"
        )