


#### -------
#### Imports
#### -------


# Standard library imports
from pathlib import Path
import pandas as pd

# Third-party imports
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns




#### ------------------------
#### Sample Loading
#### ------------------------


def collect_grouped_sample_data(base_dir: str = "processed_outputs_60secbin", sample_groups: dict[str, list[int]] = None) -> pd.DataFrame:
    """
    Aggregate Fission-Fusion CSV data collection

    Parameters:
        base_dir (str): Root path containing sample folders.
        sample_groups (dict): dict mapping condition names to sample ID lists.

    Returns:
        pd.DataFrame: DataFrame containing condition labels (e.g. "WT"), sample IDs (e.g. 5), sample group index for creating colours downstream, and associated sample colour.
    """

    all_data = []
    
    for condition, sample_ids in sample_groups.items():
        for idx, sample_id in enumerate(sample_ids):
            csv_path = Path(base_dir) / str(sample_id) / "processed_data_analysis" / "fission_fusion_summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['Condition'] = condition
                df['SampleID'] = sample_id
                df['SampleGroupIndex'] = idx
                df['SampleColor'] = sample_colors[idx]
                all_data.append(df)
            else:
                print(f"[WARNING] Missing file: {csv_path}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def collect_time_series_data(base_dir: str = "processed_outputs", sample_groups: dict[str, list[int]] = None, domains: list[str] = None, rate_cols: list[str] = None) -> pd.DataFrame:
    """
    Temporal Fission-Fusion CSV data collection

    Parameters:
        base_dir (str): Root path containing sample folders.
        sample_groups (dict): dict mapping condition names to sample ID lists.
        domains (list): list of strings of cellular spatial domain names used in the fission-fusion rate calculations.
        rate_cols (list): list of strings containing fission/fusion normalisation rate column names.

    Returns:
        pd.DataFrame: DataFrame containing time in minutes, the rate type, rate values, sample IDs (e.g. 5), condition labels (e.g. "WT"), and associated cellular domain.
    """

    all_dfs = []

    for condition, sample_ids in sample_groups.items():
        for sid in sample_ids:

            
            sample_dir = Path(base_dir) / str(sid) / "processed_data_analysis" # Change this if saved elsewhere

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
        return pd.DataFrame(columns=["Time (Minutes)", "RateType", "RateValue", "SampleID", "Condition", "Domain"])
    



#### -----------------------------------------
#### Aggregate Fission-Fusion Rates Plotting
#### -----------------------------------------


def plot_aggregate_fission_fusion_rates(df: pd.DataFrame, palette: dict[str, str], regions_order: list[str], rate_col: str = 'Normalized Fusion Rate (Mean Total Mass)', save_name: str = None) -> None:
    """
    Plots separate figures for aggregate Fusion and Fission rates.

    Parameters:
        df (pd.DataFrame): long-form DataFrame with RateType and RateValue columns.
        palette (dict): dict mapping condition names to colors.
        regions_order (list): list of domain strings for subplot ordering.
        rate_col (str): string of normalisation method (column heading) in "fission_fusion_summary.csv".
        save_name (str): optional string to save plots and name the output as "{save_name}.png".

    Returns:
        None
    """
        
    # Create a combined identifier for sample coloring
    df['SampleIdentifier'] = df['Condition'] + '_Sample' + df['SampleGroupIndex'].astype(str)
    
    # Create sample color palette
    sample_palette = {}
    for condition in df['Condition'].unique():
        for i in range(3):
            sample_palette[f'{condition}_Sample{i}'] = sample_colors[i]
    

    plt.figure(figsize=(12, 7))
    
    ax = sns.boxplot(
        data=df,
        x='Domain',
        y=rate_col,
        hue='Condition',
        order=regions_order,
        palette=palette,
        showfliers=False,
        linewidth=1.5,
        gap=0.3,
        medianprops=dict(color='red', linewidth=2.0)
    )
    
    # Set transparency for boxplot patches
    for patch in ax.patches:
        facecolor = patch.get_facecolor()
        patch.set_facecolor((*facecolor[:3], 0.6))
    
    # Add dotted vertical lines between categories
    for i in range(len(regions_order) - 1):
        ax.axvline(x=i + 0.5, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # Sample Points
    sns.stripplot(
        data=df,
        x='Domain',
        y=rate_col,
        hue='SampleIdentifier',
        dodge=True,
        order=regions_order,
        palette=sample_palette,
        edgecolor='black',
        linewidth=0.5,
        size=12,
        alpha=0.9,
        ax=ax
    )


    # Remove default legend
    ax.get_legend().remove()
    
    # Create custom legends
    conditions = df['Condition'].unique()
    condition_handles = [plt.Rectangle((0,0),1,1, facecolor=palette[condition], alpha=0.6, edgecolor='black') 
                        for condition in conditions]
    sample_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                markersize=8, markeredgecolor='black', markeredgewidth=0.5) 
                     for color in sample_colors]
    
    # legend1 = ax.legend(condition_handles, conditions, title='Conditions', loc='upper left', bbox_to_anchor=(0, 1))
    # legend2 = ax.legend(sample_handles, [f'Sample {i+1}' for i in range(3)], title='Sample Groups', loc='upper left', bbox_to_anchor=(0, 0.7))
    
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)
    
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_name:
        plt.savefig(f'{save_name}.png', dpi=600, transparent=True, bbox_inches='tight')
    
    plt.show()




#### -----------------------------------------
#### Temporal Fission-Fusion Rates Plotting
#### -----------------------------------------


def plot_average_time_series_rates(ts_df: pd.DataFrame, sample_groups: dict[str, list[int]], domains: list[str], palette: dict[str, str], figsize: tuple[int, int] = None, save_prefix: str = None) -> None:
    """
    Plots separate figures for Fusion and Fission rates over time.

    Parameters:
        ts_df (pd.DataFrame): long-form DataFrame generated using the collect_time_series_data function.
        sample_groups (dict): dict mapping condition names to sample ID lists.
        domains (list): list of domain strings.
        palette (dict): dict mapping condition names to colors.
        figsize (tuple): tuple for figure size.
        save_prefix (str): optional string to save plots and save them as "{save_prefix}_{RateType}_Rates_Timecourse.png".

    Returns:
        None
    """

    rate_types = ['Fusion', 'Fission']
    for rate_str in rate_types:
        n_dom = len(domains)
        fig, axes = plt.subplots(1, n_dom, sharey=True, figsize=figsize or (5 * n_dom, 4))
        if n_dom == 1:
            axes = [axes]

        stats = (
            ts_df[ts_df['RateType'].str.contains(rate_str)]
            .groupby(['Time (Minutes)', 'Condition', 'Domain'])['RateValue']
            .agg(['mean', 'sem'])
            .reset_index()
        )

        for j, dom in enumerate(domains):
            ax = axes[j]
            sub = stats[stats['Domain'] == dom]

            for cond in sample_groups:
                dcond = sub[sub['Condition'] == cond]
                color = palette.get(cond)
                ax.plot(
                    dcond['Time (Minutes)'], dcond['mean'],
                    label=cond,
                    linewidth=1.5,
                    color=color
                )
                ax.fill_between(
                    dcond['Time (Minutes)'],
                    dcond['mean'] - dcond['sem'],
                    dcond['mean'] + dcond['sem'],
                    alpha=0.3,
                    color=color,
                    linewidth=0.0
                )

            # Set limits explicitly to start at 0
            ax.set_xlim(left=0)  
            ax.set_ylim(bottom=0)  
            
            # Set ticks with MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(5))
            # ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.autoscale(axis='y')
            ax.set_ylim(bottom=0)

            # Manual tick setting (if you want more control)
            # x_min, x_max = ax.get_xlim()
            # y_min, y_max = ax.get_ylim()
            # x_ticks = np.arange(0, x_max + 5, 5)  # Start from 0, step by 5
            # y_ticks = np.arange(0, y_max + 50, 50)  # Start from 0, step by 50
            # ax.set_xticks(x_ticks)
            # ax.set_yticks(y_ticks)
            # ax.set_xlabel('Time (minutes)')
            # ax.set_ylabel(f'{rate_str} Rate (μm$^{{-2}}$ s$^{{-1}}$)')

            ax.tick_params(axis='both', direction='out', labelsize=18, width=1, length=6)
        plt.tight_layout()

        if save_prefix:
            plt.savefig(f"{save_prefix}_{rate_str}_Rates_Timecourse.png", dpi=600, bbox_inches="tight", transparent=True)
            
        plt.show()



if __name__=="__main__":
    
    # Region Mappings
    regions_order = ["Global", "Telenuclear", "Perinuclear"]
    
    domain_mapping = {
            "Global": "Global",
            "Telenuclear": "Telenuclear",
            "Perinuclear (Combined)": "Perinuclear"
        }
    
    sample_colors = ['#66CCCC', '#3399CC', '#004C99']  # For the 3 samples in each group

    # Colour Palettes
    palette = {
            'WT': '#FF6000',
            'IRSp53 KO': '#2E58FF',
            '9.6mW': '#FAA3FA',
            '27mW': '#E15EF9',
            '128.7mW': '#8303C8',
            '252.3mW': '#330084'
        }




    #### --------------------------------------------
    #### Aggregate Fission-Fusion Rates (Condition 1)
    #### --------------------------------------------


    # Figure 1: Irradiation conditions 
    sample_groups_condition_1 = {"9.6mW": [17, 18, 20],
                                 "27mW": [25, 26, 28],
                                 "128.7mW": [1, 2, 4],
                                 "252.3mW": [21, 22, 23]
                                 }
    
    # Load Condition 1 Data
    df_all = collect_grouped_sample_data(base_dir="processed_outputs_1secbin", sample_groups=sample_groups_condition_1)

    # Filter Data for Correct Region Data to Plot
    df_filtered = df_all[df_all['Domain'].isin(domain_mapping.keys())].copy()
    df_filtered['Domain'] = df_filtered['Domain'].map(domain_mapping)

    # Plot Aggregate Fission Fusion Rates for Condition 1
    # Fusion
    plot_aggregate_fission_fusion_rates(df_filtered, 
                                        palette=palette, 
                                        regions_order=regions_order, 
                                        rate_col='Normalized Fusion Rate (Mean Total Mass)', 
                                        save_name='3F_Irradiance_AverageAggregateFusionRates_BoxPlot')
    
    # Fission
    plot_aggregate_fission_fusion_rates(df_filtered, 
                                        palette=palette, 
                                        regions_order=regions_order, 
                                        rate_col='Normalized Fission Rate (Mean Total Mass)', 
                                        save_name='3E_Irradiance_AverageAggregateFissionRates_BoxPlot')




    #### --------------------------------------------
    #### Aggregate Fission-Fusion Rates (Condition 2)
    #### --------------------------------------------


    # Figure 2: Genetic pertubation conditions
    # Condition Name: Replicate Folder IDs
    sample_groups_condition_2 = {"WT": [5, 7, 10], 
                                 "IRSp53 KO": [33, 34, 35]
                                 }

    # Load Condition 2 Data
    df_all_condition_2 = collect_grouped_sample_data(sample_groups=sample_groups_condition_2)

    df_filtered_condition_2 = df_all_condition_2[df_all_condition_2['Domain'].isin(domain_mapping.keys())].copy()
    df_filtered_condition_2['Domain'] = df_filtered_condition_2['Domain'].map(domain_mapping)

    # Fusion
    plot_aggregate_fission_fusion_rates(df_filtered_condition_2, 
                                        palette=palette, 
                                        regions_order=regions_order, 
                                        rate_col='Normalized Fusion Rate (Mean Total Mass)', 
                                        save_name='3J_WTvsIRSp53KO_AverageAggregateFusionRates_BoxPlot')
    
    # Fission
    plot_aggregate_fission_fusion_rates(df_filtered_condition_2, 
                                        palette=palette, 
                                        regions_order=regions_order, 
                                        rate_col='Normalized Fission Rate (Mean Total Mass)', 
                                        save_name='3I_WTvsIRSp53KO_AverageAggregateFissionRates_BoxPlot')
    



    #### --------------------------------------------
    #### Temporal Fission-Fusion Rates (Condition 1)
    #### --------------------------------------------

    # Normalisation Strategy to Plot Temporal Fission-Fusion Rates for
    rate_cols = [
        "Normalized Fusion Rate (Initial Total Mass)",
        "Normalized Fission Rate (Initial Total Mass)"
    ]

    
    # Temporal Fission-Fusion Rates for Condition 1 (Photoirradiation)
    # Condition Name: Replicate Folder IDs
    sample_groups_temporal_condition_1 = {"9.6mW": [17, 18, 20],
                                        #  "27mW": [25, 26, 28],
                                        #  "128.7mW": [1, 2, 4],
                                        "252.3mW": [21, 22, 23]
                                        }

    domains = ["Global", "Telenuclear", "Perinuclear (Combined)"]

    ts_df_condition_1 = collect_time_series_data(
        base_dir="processed_outputs_20secbin/",
        sample_groups=sample_groups_temporal_condition_1,
        domains=domains,
        rate_cols=rate_cols
    )

    plot_average_time_series_rates(
        ts_df_condition_1,
        sample_groups=sample_groups_temporal_condition_1,
        domains=domains,
        palette=palette,
        figsize=(5 * len(domains), 5),
        save_prefix="3E_Irradiance_Domain_Fission_Fusion_Rates_TimeCourse_LinePlot_20SecBinning"
    )



    #### --------------------------------------------
    #### Temporal Fission-Fusion Rates (Condition 2)
    #### --------------------------------------------

    # Condition Name: Replicate Folder IDs
    sample_groups_temporal_condition_2 = {"WT": [5, 7, 10], 
                                 "IRSp53 KO": [33, 34, 35]
                                 }

    domains = ["Global", "Telenuclear", "Perinuclear (Combined)"]


    # Temporal Fission-Fusion Rates for Condition 2 (Genetic Perturbation)
    ts_df_condition_2 = collect_time_series_data(
        base_dir="processed_outputs_20secbin/",
        sample_groups=sample_groups_temporal_condition_2,
        domains=domains,
        rate_cols=rate_cols
    )

    plot_average_time_series_rates(
        ts_df_condition_2,
        sample_groups=sample_groups_temporal_condition_2,
        domains=domains,
        palette=palette,
        figsize=(5 * len(domains), 5),
        save_prefix="3I_WTvsIRSp53KO_Domain_Fission_Fusion_Rates_TimeCourse_LinePlot_20SecBinning"
    )