


#### -------
#### Imports
#### -------


# Standard library imports
import os
import pickle
from collections import defaultdict

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt




#### ------------------------
#### Sample Loading
#### ------------------------


def load_all_sample_results(sample_groups: dict[str, list[int]], cache_dir: str = "epidemic_curve_cache", normalization: str = "global_bidirectional") -> tuple:
    """
    Load pre-computed epidemic-like spread curve results for multiple samples and regions.

    Parameters:
        sample_groups (dict): Dictionary mapping condition names to lists of sample ID integers.
        cache_dir (str): Directory containing cached epidemic-like spread curve results from Main_Spread_Ratio_Calculation.py.
        normalization (str): Normalization scheme used during spread curve calculation (not necessary as it just changes file name suffix).

    Returns:
        spread_data, times_scaled (tuple): 
            spread_data (defaultdict):
                Nested dictionary indexed as `spread_data[sample_id][region]`, where each entry
                contains the stored spread curve metrics for that region.
            times_scaled (np.ndarray or None):
                Shared time vector corresponding to the spread curves. This is read from the
                first successfully loaded sample and assumed to be consistent across all samples.
                Returns None if no cache files are found.
    """
    
    spread_data = defaultdict(lambda: defaultdict(dict))
    times_scaled = None
    
    for group, samples in sample_groups.items():
        for sample_number in samples:
            cache_file = os.path.join(cache_dir, f"sample_{sample_number}_{normalization}.pkl")
            
            if not os.path.exists(cache_file):
                print(f"Warning: Missing cache for sample {sample_number}")
                continue
            
            print(f"Loading sample {sample_number}...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            if times_scaled is None:
                times_scaled = data['times']
            
            for region, region_data in data['regions'].items():
                spread_data[sample_number][region] = region_data
    
    return spread_data, times_scaled




#### --------------------------------
#### Spread Ratio Plotting
#### --------------------------------


def plot_inidividual_sample_epidemic_curves(spread_data: dict[int, dict[str, dict]], times_scaled: np.ndarray, sample_groups: dict[str, list[int]], palette: dict[str, str], regions_order: list[str], out_path: str | None = None) -> None:
    """
    Plot epidemic-like spread curves for individual samples, separated by region.

    Parameters:
        spread_data (dict): Nested dictionary indexed as `spread_data[sample_id][region]`, where each entry contains the stored spread curve metrics for that region.
        times_scaled (np.ndarray): 1D array of time points corresponding to the x-axis of the spread curves. Assumed to be shared across all samples and regions.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.
        palette (dict): Dictionary mapping condition names to color strings.
        regions_order (list): Ordered list of region names to plot.
        out_path (str, optional): Base output directory where, if provided, figures are saved under.

    Returns:
        None
    """

    for group, samples in sample_groups.items():
        if out_path:
            sample_output_folder = os.path.join(out_path, group)
            os.makedirs(sample_output_folder, exist_ok=True)
        for sample_number in samples:
            fig, axes = plt.subplots(1, len(regions_order), sharey=True, figsize=(4 * len(regions_order), 4))
            for i, region in enumerate(regions_order):
                ax = axes[i]
                curves = spread_data[sample_number][region]['curves']
                
                if curves is None or len(curves) == 0:
                    continue
                    
                curves_arr = np.array(curves)
                mean_curve = np.mean(curves_arr, axis=0)
                std_curve = np.std(curves_arr, axis=0)
                
                ax.plot(times_scaled, mean_curve, color=palette.get(group, 'blue'), linewidth=1.5)
                ax.fill_between(times_scaled, np.clip(mean_curve - std_curve, 0, 1), np.clip(mean_curve + std_curve, 0, 1), color=palette.get(group, 'blue'), alpha=0.4)
    
                ax.set_xlabel("")

                if i == 0:
                    ax.set_ylabel("")

                ax.tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)
            plt.tight_layout()
            if out_path:
                fig.savefig(os.path.join(sample_output_folder, f"{sample_number}_epidemic_spread_curves.png"), dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
            else:
                plt.show()
        

def plot_average_epidemic_curves(spread_data: dict[int, dict[str, dict]], times_scaled: np.ndarray, sample_groups: dict[str, list[int]], palette: dict[str, str], regions_order: list[str], out_path: str | None = None) -> None:
    """
    Plot group-averaged epidemic-like spread curves across samples for each region.
    
    Parameters:
        spread_data (dict): Nested dictionary indexed as `spread_data[sample_id][region]`, where each entry contains the stored spread curve metrics for that region.
        times_scaled (np.ndarray): 1D array of time points corresponding to the x-axis of the spread curves. Assumed to be shared across all samples and regions.
        sample_groups (dict): Dictionary mapping condition names to sample ID lists.
        palette (dict): Dictionary mapping condition names to color strings.
        regions_order (list): Ordered list of region names to plot.
        out_path (str, optional): Base output directory where, if provided, figures are saved under.

    Returns:
        None
    """

    region_group_avg = defaultdict(dict)
    
    for group, samples in sample_groups.items():
        if out_path:
            sample_output_folder = os.path.join(out_path, group)
            os.makedirs(sample_output_folder, exist_ok=True)        
        for region in regions_order:
            all_avg_curves = np.array([
                spread_data[sample_number][region]['avg']
                for sample_number in samples
                if len(spread_data[sample_number][region].get('avg', [])) > 0
            ])
            
            if len(all_avg_curves) > 0:
                mean_curve = np.mean(all_avg_curves, axis=0)
                sem_curve = np.std(all_avg_curves, axis=0) / np.sqrt(len(all_avg_curves))
                region_group_avg[group][region] = (mean_curve, sem_curve)
    
    fig, axes = plt.subplots(1, len(regions_order), sharey=True, figsize=(4 * len(regions_order), 4))
    for i, region in enumerate(regions_order):
        ax = axes[i]
        for group in sample_groups.keys():
            if group in region_group_avg and region in region_group_avg[group]:
                mean_curve, sem_curve = region_group_avg[group][region]
                ax.plot(times_scaled, mean_curve, label=group, color=palette.get(group, 'blue'), alpha=0.8)
                ax.fill_between(times_scaled, np.clip(mean_curve - sem_curve, 0, 1), np.clip(mean_curve + sem_curve, 0, 1), color=palette.get(group, 'blue'), alpha=0.2)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("")
        ax.tick_params(axis='both', direction='out', labelsize=14, width=1, length=6)

    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout()
    if out_path:
        fig.savefig(os.path.join(sample_output_folder, f"average_epidemic_spread_curves.png"), dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


if __name__ == "__main__":
    # Colour Palettes
    palette = {
            'WT': '#FF6000',
            'IRSp53 KO': '#2E58FF',
            '9.6mW': '#FAA3FA',
            '27mW': '#E15EF9',
            '128.7mW': '#8303C8',
            '252.3mW': '#330084'
        }
    
    # Region Mappings
    regions_order = ["Global", "Telenuclear", "Perinuclear"]
    
    normalization = "global_bidirectional" # Normalization used during calculation (not necessary as it just changes file name suffix)
    cache_dir = "spread_ratio_data_cache"
    output_folder = f"epidemic_spread_curves_{normalization}"
    
    os.makedirs(output_folder, exist_ok=True)

    
    # Condition 1: Irradiation conditions
    sample_groups_condition_1 = {
                                "9.6mW": [17, 18, 20],
                                "27mW": [25, 26, 28],
                                "128.7mW": [1, 2, 4],
                                "252.3mW": [21, 22, 23]
                                }

    # Load all pre-computed results
    print("Loading pre-computed results...")
    spread_data, times_scaled = load_all_sample_results(
        sample_groups_condition_1, 
        cache_dir=cache_dir,
        normalization=normalization
    )
    
    # Generate plots
    print("Generating plots...")
    plot_inidividual_sample_epidemic_curves(
        spread_data, times_scaled, sample_groups_condition_1, palette, 
        regions_order, out_path=output_folder
    )
    
    plot_average_epidemic_curves(
        spread_data, times_scaled, sample_groups_condition_1, palette, 
        regions_order, out_path=output_folder
    )


    # # Condition 2: WT vs IRSp53 KO
    # sample_groups_condition_2 = {
    #                             "WT": [5, 7, 10],
    #                             "IRSp53 KO": [33, 34, 35]
    #                             }

    # # Load all pre-computed results
    # print("Loading pre-computed results...")
    # spread_data, times_scaled = load_all_sample_results(
    #     sample_groups_condition_1, 
    #     cache_dir=cache_dir,
    #     normalization=normalization
    # )
    
    # # Generate plots
    # print("Generating plots...")
    # plot_inidividual_sample_epidemic_curves(
    #     spread_data, times_scaled, sample_groups_condition_1, palette, 
    #     regions_order, out_path=output_folder
    # )
    
    # plot_average_epidemic_curves(
    #     spread_data, times_scaled, sample_groups_condition_1, palette, 
    #     regions_order, out_path=output_folder
    # )