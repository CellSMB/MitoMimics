


#### -------
#### Imports
#### -------


# Standard library imports
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd




#### --------------------------------------------------
#### Sample Loading and Fragmentation Index Calculation
#### --------------------------------------------------


def collect_fragmentation_metrics_over_time(base_dir: str = "processed_outputs_60secbin", sample_groups: dict[str, list[int]] = None, domains: list[str] = None) -> pd.DataFrame:
    """
    Computes normalized mitochondrial count per total area (Fragmentation Index) per time point, optionally per region (Global, Telenuclear, Perinuclear).

    Parameters:
        base_dir (str): Root path containing sample folders.
        sample_groups (dict): dict mapping condition names to sample ID lists.
        domains (list[str]): List of strings of cellular spatial domain names used in the fission-fusion rate calculations.

    Returns:
        pd.DataFrame: DataFrame with columns ["Time (Minutes)", "Domain", "SampleID", "Condition", "Fragmentation Index"]
    """

    all_dfs = []

    for condition, sample_ids in sample_groups.items():
        for sid in sample_ids:
            path = Path(base_dir) / str(sid) / "processed_data_analysis" / "general_morphological_features_per_object.csv"
            if not path.exists():
                print(f"Missing data for sample {sid}")
                continue

            df = pd.read_csv(path)
            df["Time (Minutes)"] = df["Time (Seconds)"] / 60.0
            df = df.replace(0, pd.NA)

            group_list = []

            # Global
            global_group = df.groupby("Time (Minutes)")
            global_frag = (global_group["Label"].nunique() / global_group["Area"].sum()).reset_index(name="Fragmentation Index")
            global_frag["Domain"] = "Global"
            group_list.append(global_frag)

            # Perinuclear (Combined)
            perinuc_mask = df["Region"].isin(["Perinuclear", "Transition"])
            perinuc_df = df[perinuc_mask]
            perinuc_group = perinuc_df.groupby("Time (Minutes)")
            perinuc_frag = (perinuc_group["Label"].nunique() / perinuc_group["Area"].sum()).reset_index(name="Fragmentation Index")
            perinuc_frag["Domain"] = "Perinuclear (Combined)"
            group_list.append(perinuc_frag)

            # Individual Regions
            reg_df = df[df["Region"].isin(["Telenuclear", "Perinuclear", "Transition"])]
            reg_group = reg_df.groupby(["Time (Minutes)", "Region"])
            reg_frag = (reg_group["Label"].nunique() / reg_group["Area"].sum()).reset_index(name="Fragmentation Index")
            reg_frag = reg_frag.rename(columns={"Region": "Domain"})
            group_list.append(reg_frag)

            merged = pd.concat(group_list, ignore_index=True)
            merged["SampleID"] = sid
            merged["Condition"] = condition
            all_dfs.append(merged)

    return pd.concat(all_dfs, ignore_index=True)




#### ----------------------------
#### Fragmentation Index Plotting
#### ----------------------------


def plot_average_fragmentation_index(ts_df: pd.DataFrame, sample_groups: dict[str, list[int]], domains: list[str], title_map: dict[str, str], palette: dict[str, str], metric: str = "Fragmentation Index", save_prefix: str | None = None, plot_horizontal: bool = True) -> None:
    """
    Plots separate figures for Fusion and Fission rates over time.

    Parameters:
        ts_df (pd.DataFrame): Long-form DataFrame generated using the collect_fragmentation_metrics_over_time function.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.
        domains (list): List of domain strings.
        title_map (dict): Dictionary mapping region labels with desired labels.
        palette (dict): Dictionary mapping condition names to colors.
        metric (str): String of metric column heading.
        save_prefix (str, optional): Optional string to save plots and save them as "{save_prefix}_{metric.replace(' ', '')}.png".
        plot_horizontal (bool): Plot subplots horizontally or vertically.

    Returns:
        None
    """
        
    stats = (
        ts_df
        .groupby(["Time (Minutes)", "Condition", "Domain"])[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )

    n_dom = len(domains)

    if plot_horizontal:
        figsize = (4 * n_dom, 4)
        fig, axes = plt.subplots(1, n_dom, sharey=True, figsize=figsize)
    else:
        figsize = (5, 5*n_dom)
        fig, axes = plt.subplots(n_dom, 1, sharex=True, sharey=True, figsize=figsize)

    if n_dom == 1:
        axes = [axes]

    for j, dom in enumerate(domains):
        ax = axes[j]
        sub = stats[stats['Domain'] == dom]

        for cond in sample_groups:
            dcond = sub[sub['Condition'] == cond].copy()
            color = palette.get(cond, None)

            dcond["Time (Minutes)"] = pd.to_numeric(dcond["Time (Minutes)"], errors="coerce")
            dcond["mean"] = pd.to_numeric(dcond["mean"], errors="coerce")
            dcond["sem"] = pd.to_numeric(dcond["sem"], errors="coerce")

            dcond = dcond.dropna(subset=["Time (Minutes)", "mean", "sem"])

            if not dcond.empty:
                ax.plot(dcond['Time (Minutes)'], dcond['mean'], label=cond, color=color, lw=1.5)
                ax.fill_between(dcond['Time (Minutes)'], dcond['mean'] - dcond['sem'], dcond['mean'] + dcond['sem'], alpha=0.4, color=color, edgecolor="face", linewidth=0)

        # ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, auto=True)
        # ax.set_title(title_map.get(dom, dom), fontsize=11)
        # ax.set_xlabel("Time (Minutes)", fontsize=10)
        # if j == 0:
        #     ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)
        # ax.tick_params(labelsize=14)
        # if j == n_dom - 1:
        #     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

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

    # Region Mappings and Ordering
    domains = ["Global", "Telenuclear", "Perinuclear (Combined)"]

    title_map = {
        "Global": "Global",
        "Telenuclear": "Telenuclear",
        "Perinuclear (Combined)": "Perinuclear"
    }
    



    #### ----------------------------------------------------
    #### Fragmentation Index (Condition 1 - Photoirradiation)
    #### ----------------------------------------------------

    # Condition Name: Replicate Folder IDs
    sample_groups_condition_1 = {"9.6mW": [17, 18, 20],
                                 "27mW": [25, 26, 28],
                                 "128.7mW": [1, 2, 4],
                                 "252.3mW": [21, 22, 23]
                                 }


    frag_df_condition_1 = collect_fragmentation_metrics_over_time(
        base_dir="processed_outputs_60secbin",
        sample_groups=sample_groups_condition_1,
        domains=domains
    )

    plot_average_fragmentation_index(
        ts_df=frag_df_condition_1,
        sample_groups=sample_groups_condition_1,
        domains=domains,
        title_map=title_map,
        palette=palette,
        metric="Fragmentation Index",
        save_prefix="3B_MitoFragmentation_Timecourse_Photoirradiations",
        plot_horizontal=True
    )





    #### ----------------------------------------------------
    #### Fragmentation Index (Condition 2 - Genetic KO)
    #### ----------------------------------------------------

    # Condition Name: Replicate Folder IDs
    sample_groups_condition_2 = {"WT": [5, 7, 10], 
                                 "IRSp53 KO": [33, 34, 35]
                                 }

    frag_df_condition_2 = collect_fragmentation_metrics_over_time(
        base_dir="processed_outputs_60secbin",
        sample_groups=sample_groups_condition_2,
        domains=domains
    )

    plot_average_fragmentation_index(
        ts_df=frag_df_condition_2,
        sample_groups=sample_groups_condition_2,
        domains=domains,
        title_map=title_map,
        palette=palette,
        metric="Fragmentation Index",
        save_prefix="3B_MitoFragmentation_Timecourse_KO",
        plot_horizontal=True
    )