


#### -------
#### Imports
#### -------


# Standard library imports
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd




#### --------------------------------------
#### Sample Loading and Metric Calculation
#### --------------------------------------


def collect_branching_metrics_over_time(base_dir: str = "processed_outputs", sample_groups: dict[str, list[int]] = None, bin_size: int = 20) -> pd.DataFrame:
    """
    Loads and merges morphology and topology data for each sample, computes normalized
    topology metrics, and aggregates them per domain (Global, Telenuclear, Perinuclear),
    binned over time.

    Parameters:
        base_dir (str): Root path containing sample folders.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.
        bin_size (int): Integer in seconds for time bin width to reduce noise.

    Returns:
        pd.DataFrame: DataFrame with columns ["Time (Minutes)", "Domain", "SampleID", "Condition", "Branch Density", "Endpoint Density", ..., "Complexity Index"]
    """

    all_dfs = []

    for condition, sample_ids in sample_groups.items():
        for sid in sample_ids:
            morph_path = Path(base_dir) / str(sid) / "processed_data_analysis" / "general_morphological_features_per_object.csv"
            topo_path = Path(base_dir) / str(sid) / "processed_data_analysis" / "endpoint_branching_per_object_metrics.csv"
            
            if not morph_path.exists() or not topo_path.exists():
                print(f"Missing data for sample {sid}")
                continue

            morph_df = pd.read_csv(morph_path)
            topo_df = pd.read_csv(topo_path)

            df = pd.merge(morph_df, topo_df, on=["Label", "Time (Seconds)"])
            df = df.replace(0, pd.NA)

            # Create time bin
            df["Time Bin (Seconds)"] = (df["Time (Seconds)"] // bin_size) * bin_size
            df["Time (Minutes)"] = df["Time Bin (Seconds)"] / 60.0  # binned time in minutes

            # Normalized metrics
            df["Branch Density"] = df["Branch Count"] / df["Area"]
            df["Endpoint Density"] = df["Endpoint Count"] / df["Area"]
            df["Total Network Density"] = (df["Branch Count"] + df["Endpoint Count"]) / df["Area"]
            df["% Branch Nodes"] = 100 * df["Branch Count"] / (df["Branch Count"] + df["Endpoint Count"])
            df["% Endpoint Nodes"] = 100 * df["Endpoint Count"] / (df["Branch Count"] + df["Endpoint Count"])
            df["Complexity Index"] = df["Branch Count"] / df["Endpoint Count"]

            group_list = []

            # Global
            global_means = (
                df.groupby("Time (Minutes)")[["Branch Density", "Endpoint Density", "Total Network Density", "% Branch Nodes", "% Endpoint Nodes", "Complexity Index"]]
                .mean()
                .reset_index()
            )
            global_means["Domain"] = "Global"
            group_list.append(global_means)

            # Perinuclear (Combined)
            perinuc_mask = df["Region"].isin(["Perinuclear", "Transition"])
            perinuc_df = df[perinuc_mask]
            perinuc_means = (
                perinuc_df.groupby("Time (Minutes)")[["Branch Density", "Endpoint Density", "Total Network Density", "% Branch Nodes", "% Endpoint Nodes", "Complexity Index"]]
                .mean()
                .reset_index()
            )
            perinuc_means["Domain"] = "Perinuclear (Combined)"
            group_list.append(perinuc_means)

            # Individual Regions
            reg_df = df[df["Region"].isin(["Telenuclear", "Perinuclear", "Transition"])]
            reg_means = (
                reg_df
                .groupby(["Time (Minutes)", "Region"])[["Branch Density", "Endpoint Density", "Total Network Density", "% Branch Nodes", "% Endpoint Nodes", "Complexity Index"]]
                .mean()
                .reset_index()
                .rename(columns={"Region": "Domain"})
            )
            group_list.append(reg_means)

            merged = pd.concat(group_list, ignore_index=True)
            merged["SampleID"] = sid
            merged["Condition"] = condition
            all_dfs.append(merged)

    return pd.concat(all_dfs, ignore_index=True)




#### --------------------------------
#### Branching Metrics Plotting
#### --------------------------------


def plot_average_branching_metrics(ts_df: pd.DataFrame, sample_groups: dict[str, list[int]], domains: list[str], title_map: dict[str, str], palette: dict[str, str], metric: str = "Branch Density", save_prefix: str | None = None) -> None:
    """
    Plots average ± SEM of a selected topology metric over time, with one subplot per domain, grouped by condition.

    Parameters:
        ts_df (pd.DataFrame): Long-form DataFrame generated using the collect_fragmentation_metrics_over_time function.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.
        domains (list): List of domain strings.
        title_map (dict): Dictionary mapping renaming region labels with desired labels.
        palette (dict): Dictionary mapping condition names to colors.
        metric (str): String of metric column heading.
        save_prefix (str): Optional string to save plots and save them as "{save_prefix}_{metric.replace(' ', '')}.png".

    Returns:
        None
    """

    # Group statistics
    stats = (
        ts_df
        .groupby(['Time (Minutes)', 'Condition', 'Domain'])[metric]
        .agg(['mean', 'sem'])
        .reset_index()
    )

    n_dom = len(domains)
    figsize = (4 * n_dom, 4)
    fig, axes = plt.subplots(1, n_dom, sharey=True, figsize=figsize)
    if n_dom == 1:
        axes = [axes]

    for j, dom in enumerate(domains):
        ax = axes[j]
        sub = stats[stats['Domain'] == dom]

        for cond in sample_groups:
            dcond = sub[sub['Condition'] == cond].copy()
            color = palette.get(cond, None)

            # Ensure all plotting columns are numeric
            dcond["Time (Minutes)"] = pd.to_numeric(dcond["Time (Minutes)"], errors="coerce")
            dcond["mean"] = pd.to_numeric(dcond["mean"], errors="coerce")
            dcond["sem"] = pd.to_numeric(dcond["sem"], errors="coerce")

            # Drop invalid rows
            dcond = dcond.dropna(subset=["Time (Minutes)", "mean", "sem"])

            if not dcond.empty:
                ax.plot(dcond['Time (Minutes)'], dcond['mean'], label=cond, color=color, lw=1.5)
                ax.fill_between(
                    dcond['Time (Minutes)'],
                    dcond['mean'] - dcond['sem'],
                    dcond['mean'] + dcond['sem'],
                    alpha=0.8,
                    color=color,
                    edgecolor="face",
                    linewidth=0
                )

        # --- Plot styling ---
        # ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)
        # ax.set_title(title_map.get(dom, dom), fontsize=11)
        # ax.set_xlabel("Time (Minutes)", fontsize=10)
        # if j == 0:
        #     ax.set_ylabel(metric, fontsize=10)

        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.set_aspect("auto")

        # Add legend to last plot
        if j == n_dom - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

    plt.tight_layout()
    if save_prefix:
        save_name = f"{save_prefix}_{metric.replace(' ', '')}.png"
        plt.savefig(save_name, dpi=600, bbox_inches="tight", transparent=True)
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


    domains = ["Global", "Telenuclear", "Perinuclear (Combined)"]

    title_map = {
        "Global": "Global",
        "Telenuclear": "Telenuclear",
        "Perinuclear (Combined)": "Perinuclear"
    }
    



    # Condition 1: Irradiation conditions
    sample_groups_condition_1 = {
                                "9.6mW": [17, 18, 20],
                                # "27mW": [25, 26, 28],
                                # "128.7mW": [1, 2, 4],
                                "252.3mW": [21, 22, 23]
                                }
    # Collect the metrics
    metrics_df = collect_branching_metrics_over_time(
        base_dir="processed_outputs_60secbin",
        sample_groups=sample_groups_condition_1,
        bin_size=1  #seconds
    )

    # Plot each metric separately
    for metric in ["% Branch Nodes", "% Endpoint Nodes", "Complexity Index"]:
        plot_average_branching_metrics(
            ts_df=metrics_df,
            sample_groups=sample_groups_condition_1,
            domains=domains,
            title_map=title_map,
            palette=palette,
            metric=metric,
            save_prefix="4_MitoTopology_Analysis_Irradiation_1SecBin"
        )




    # Condition 2: WT vs IRSp53 KO
    sample_groups_condition_2 = {
                                "WT": [5, 7, 10],
                                "IRSp53 KO": [33, 34, 35]
                                }
    
    # Collect the metrics
    metrics_df = collect_branching_metrics_over_time(
        base_dir="processed_outputs_60secbin",
        sample_groups=sample_groups_condition_2,
        bin_size=1  #seconds
    )

    # Plot each metric separately
    for metric in ["% Branch Nodes", "% Endpoint Nodes", "Complexity Index"]:
        plot_average_branching_metrics(
            ts_df=metrics_df,
            sample_groups=sample_groups_condition_2,
            domains=domains,
            title_map=title_map,
            palette=palette,
            metric=metric,
            save_prefix="4_MitoTopology_Analysis_IRSp53KO_WT_1SecBin"
        )