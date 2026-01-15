


#### -------
#### Imports
#### -------


# Standard library imports
from pathlib import Path
import pandas as pd

# Third-party imports
import numpy as np
from scipy.signal import correlate, detrend
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator




#### ------------------------
#### Sample Loading
#### ------------------------


def collect_time_series_data(base_dir: str = "processed_outputs", sample_groups: dict[str, list[int]] = None, domains: list[str] = None, rate_cols: list[str] = None) -> pd.DataFrame:
    """
    Reads CSVs named like `{domain}_fission_fusion_rates_over_time.csv` for each sample,
    tags them by SampleID and Domain, and concatenates into one DataFrame.

    Parameters:
        base_dir (str): Root path containing sample folders.
        sample_groups (dict): dict mapping condition names to sample ID lists.
        domains (list): list of strings of cellular spatial domain names used in the fission-fusion rate calculations.
        rate_cols (list): list of strings containing fission/fusion normalisation rate column names.

    Returns
        pd.DataFrame: long-form DataFrame with columns ["Time (Minutes)", "RateType", "RateValue", "SampleID", "Condition", "Domain"].
    """

    all_dfs = []

    for condition, sample_ids in sample_groups.items():
        for sid in sample_ids:
            
            sample_dir = Path(base_dir) / str(sid) / "processed_data_analysis"

            for dom in domains:
                csv_path = sample_dir / f"{dom}_fission_fusion_rates_over_time.csv"
                if not csv_path.exists():
                    print(f"Missing {csv_path}")
                    continue

                df = pd.read_csv(csv_path, engine='python')
                if df.columns[0] != "Time (Minutes)":
                    df = df.rename(columns={df.columns[0]: "Time (Minutes)"})
                df = df[["Time (Minutes)"] + rate_cols].copy()
                long = df.melt(
                    id_vars="Time (Minutes)",
                    value_vars=rate_cols,
                    var_name="RateType",
                    value_name="RateValue"
                )
                long["SampleID"] = sid
                long["Condition"] = condition
                long["Domain"] = dom
                all_dfs.append(long)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=[
            "Time (Minutes)", "RateType", "RateValue",
            "SampleID", "Condition", "Domain"
        ])




#### -----------------------------------------
#### Cross-Correlation Processing and Plotting
#### -----------------------------------------

      
def cross_correlate_fission_fusion(df: pd.DataFrame, detrend_data: bool = True) -> tuple:
    """
    Compute cross-correlation between fusion and fission rates for each condition/domain.
    Returns results DataFrame and full xcorr data for plotting (with SEM).

    Parameters:
        df (pd.DataFrame): long-form DataFrame (produced by the collect_time_series_data function) with columns ["Time (Minutes)", "RateType", "RateValue", "SampleID", "Condition", "Domain"].
        detrend_data (bool): if True, removes linear trends before correlation.

    Returns:
        pd.DataFrame(results), xcorr_data (tuple): Final correlation results as a pd.DataFrame for easy tabular access and as a dictionary for ease of plotting.
    """

    results = []
    xcorr_data = {}
    
    for condition in df['Condition'].unique():
        for domain in df['Domain'].unique():
            subset = df[(df['Condition'] == condition) & (df['Domain'] == domain)]
            
            # Get all samples for this condition/domain
            sample_ids = subset['SampleID'].unique()
            xcorr_list = []
            
            # Compute xcorr for each sample
            times_ref = None
            for sid in sample_ids:
                sample_subset = subset[subset['SampleID'] == sid]
                pivot = sample_subset.pivot_table(
                    index='Time (Minutes)',
                    columns='RateType',
                    values='RateValue'
                ).dropna()
                
                if len(pivot) < 2:
                    continue
                
                fusion = pivot['Normalized Fusion Rate (Initial Total Mass)'].values
                fission = pivot['Normalized Fission Rate (Initial Total Mass)'].values
                times = pivot.index.values
                
                if times_ref is None:
                    times_ref = times
                
                # Detrend
                if detrend_data:
                    fusion = detrend(fusion)
                    fission = detrend(fission)
                
                # Normalize (z-score)
                fusion = (fusion - np.mean(fusion)) / (np.std(fusion))# + 1e-10)
                fission = (fission - np.mean(fission)) / (np.std(fission))# + 1e-10)
                
                # Cross-correlate
                xcorr = correlate(fusion, fission, mode='same')
                xcorr = xcorr / len(fusion)
                xcorr_list.append(xcorr)
            
            if not xcorr_list:
                continue
            
            # Average xcorr and compute SEM
            xcorr_array = np.array(xcorr_list)
            xcorr_mean = np.mean(xcorr_array, axis=0)
            xcorr_sem = sem(xcorr_array, axis=0)
            
            # Create lag array (in minutes)
            center = len(xcorr_mean) // 2
            time_step = times_ref[1] - times_ref[0] if len(times_ref) > 1 else 1
            lags = np.arange(len(xcorr_mean)) - center
            lags_minutes = lags * time_step
            
            # Find max correlation and corresponding lag
            max_idx = np.argmax(np.abs(xcorr_mean))
            max_corr = xcorr_mean[max_idx]
            lag_idx = max_idx - center
            lag_minutes = lag_idx * time_step
            
            results.append({
                'Condition': condition,
                'Domain': domain,
                'Max Correlation': max_corr,
                'Lag (minutes)': lag_minutes,
                'N_samples': len(xcorr_list)
            })
            
            xcorr_data[(condition, domain)] = (lags_minutes, xcorr_mean, xcorr_sem)
    
    return pd.DataFrame(results), xcorr_data


def plot_cross_correlation(conditions: list[str], domains: list[str], palette: dict[str, str], save_name: str, with_chart_elements: bool = False) -> None:
    """
    Plots cross-correlation subplots in a grid.

    Parameters:
        conditions (list): list containing string names of each condition identical to key names in paletts.
        domains (list): list of strings of cellular spatial domain names used in the fission-fusion rate calculations.
        palette (dict): dict mapping condition names to colors.
        save_name (str): optional string to save plots and name the output as "{save_name}.png".
        with_chart_elements (bool): if True, plot with axes labels, legend, and grid.

    Returns:
        None
    """
        
    fig, axes = plt.subplots(4, len(domains), figsize=(20, 14), sharey=True, sharex=True)

    for ax, condition in zip(axes, conditions):
        color = palette.get(condition)
        for d, domain in enumerate(domains):
            if (condition, domain) in xcorr_data:
                lags, xcorr_mean, xcorr_sem = xcorr_data[(condition, domain)]
                ax[d].plot(lags, xcorr_mean, color=color, marker='o', label=domain, linewidth=1.5, markersize=2)
                ax[d].fill_between(lags, xcorr_mean - xcorr_sem, xcorr_mean + xcorr_sem, color=color, alpha=0.3, linewidth=0.0)
        
            ax[d].axhline(0, color='k', linestyle='--', alpha=0.3)
            ax[d].axvline(0, color='k', linestyle='--', alpha=0.3)
                        
            # Set ticks with MultipleLocator
            ax[d].xaxis.set_major_locator(MultipleLocator(5))
            ax[d].yaxis.set_major_locator(MultipleLocator(0.1))
            ax[d].autoscale(axis='y')
            
            ax[d].tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)
            # ax[d].set_xlim([-5,5])
            if with_chart_elements:
                ax[d].set_xlabel('Lag (minutes)', fontsize=11)
                ax[d].set_ylabel('Cross-Correlation', fontsize=11)
                ax[d].legend()
                ax[d].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=600, bbox_inches="tight", transparent=True)
    plt.show()


if __name__=="__main__":

    # Colour Palettes
    palette = {
            'WT': '#FF6000',
            'IRSp53 KO': '#2E58FF',
            '9.6mW': '#FAA3FA',
            '27mW': '#E15EF9',
            '128.7mW': '#8303C8',
            '252.3mW': '#330084'
        }

    # Condition Name: Replicate Folder IDs
    sample_groups = {
                    # "WT": [5, 7, 10],
                    # "IRSp53 KO": [33, 34, 35],
                    "9.6mW": [17, 18, 20],
                    "27mW": [25, 26, 28],
                    "128.7mW": [1, 2, 4],
                    "252.3mW": [21, 22, 23]
                    }
    
    # Condition Names
    conditions = ['9.6mW', '27mW', '128.7mW', '252.3mW']

    # Spatial Subregions to Plot
    domains = ["Global", "Telenuclear", "Perinuclear (Combined)"]

    # Specific normalised rate columns to extract from data
    rate_cols = [
        "Normalized Fusion Rate (Initial Total Mass)",
        "Normalized Fission Rate (Initial Total Mass)"
    ]

    # Load data
    df = collect_time_series_data(
        base_dir="processed_outputs_30secbin/",
        sample_groups=sample_groups,
        domains=domains,
        rate_cols=rate_cols
        )

    # Run Cross-Correlation
    xcorr_results, xcorr_data = cross_correlate_fission_fusion(df, detrend_data=True)


    # Plot Cross-Correlation
    plot_cross_correlation(
        conditions=conditions, 
        domains=domains, 
        palette=palette, 
        save_name="Supplementary_Fission_Fusion_30sec_CrossCorrelation_xGlobalTelenuclearPerinuclear_y9.6mW27mW128.7mW252.3mW", 
        with_chart_elements=False
        )

    print(xcorr_results)