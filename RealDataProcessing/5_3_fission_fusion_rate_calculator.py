


#### -------
#### Imports
#### -------


# Standard library imports
import argparse
import os
import pickle
import zipfile
from pathlib import Path
import time
import gc

# Third-party imports
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm
from scipy.stats import binned_statistic




#### ------------------------
#### Sample Loading
#### ------------------------


def setup_data_path(sample_number, root_data_directory="", output_base_directory="processed_outputs"):
    """
    Set up the sample folder and output analysis directory.

    Args:
        sample_number (int): Identifier for the sample to analyze.
        root_data_directory (str): Optional root path containing sample folders.

    Returns:
        tuple: Paths to the sample folder and output directory.
    """

    # Input data path
    sample_folder = os.path.join(os.getcwd(), root_data_directory, str(sample_number))

    # Output path in "processed_outputs/sample_number/processed_data_analysis"
    output_root = os.path.join(os.getcwd(), output_base_directory, str(sample_number))
    analysis_output_folder = os.path.join(output_root, "processed_data_analysis")
    os.makedirs(analysis_output_folder, exist_ok=True)

    return sample_folder, analysis_output_folder


def load_data(sample_folder, load_telenuclear=False, load_perinuclear=False, load_transition=False, load_combined=False, in_memory=False):
    """
    Load instance mask data and optional sub-region datasets for a given sample.

    Args:
        sample_folder (str): Path to the sample folder.
        load_telenuclear (bool): Whether to load the telenuclear dataset.
        load_perinuclear (bool): Whether to load the perinuclear dataset.
        load_transition (bool): Whether to load the transition dataset.
        load_combined (bool): Whether to load the combined perinuclear and transition (complete perinuclear region) dataset.
        in_memory (bool): Load all data into memory (useful for HPC data processing, True) or extract data and memory-map to reduce memory overhead (useful for local data processing with limitewd RAM, False)

    Returns:
        tuple: Main instance mask, individual graph of object skeletons, and optionally loaded sub-region datasets.
    """

    # LOAD INSTANCE MASKS
    # Path to instance masks .npz file
    instance_mask_npz_file = os.path.join(os.getcwd(), sample_folder, 'instance_masks.npz')
    if in_memory:
        with np.load(instance_mask_npz_file, allow_pickle=False) as npz:
            instance_data = npz['arr_0']
    else:
        if not Path(os.path.join(sample_folder, 'instance_mask_extracted_npz/arr_0.npy')).exists():
            
            # Extract the .npz file manually
            with zipfile.ZipFile(instance_mask_npz_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(os.getcwd(), sample_folder, 'instance_mask_extracted_npz'))

        # memory-map the individual .npy files in the extracted folder
        instance_data = np.load(os.path.join(os.getcwd(), sample_folder, 'instance_mask_extracted_npz', 'arr_0.npy'), mmap_mode='r')



    # Load per Structure Sub-Node Graphs (i.e. Skeletons)
    with open(os.path.join(os.getcwd(), sample_folder, 'pos_matching_graph.pickle'), 'rb') as f:
        pos_matching_graph = pickle.load(f)


    
    # LOAD TELENUCLEAR, PERINUCLEAR, AND TRANSITION DATASETS IF SPECIFIED (OPTIONAL)
    telenuclear_instance_data = None
    perinuclear_instance_data = None
    transition_instance_data = None
    combined_perinuclear_instance_data = None

    if load_telenuclear:
        try:
            zip_path = os.path.join(sample_folder, 'perinuclear_segmentations/telenuclear_instance_masks.npz')
            if in_memory:
                with np.load(zip_path, allow_pickle=False) as npz:
                    telenuclear_instance_data = npz['data']
            else:
                path = os.path.join(sample_folder, 'perinuclear_segmentations/telenuclear_instance_mask_extracted_npz/data.npy')
                if not Path(path).exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(path))
                telenuclear_instance_data = np.load(path, mmap_mode='r')
        except Exception as e:
            print(f"Warning: Failed to load telenuclear data. Error: {e}")

    if load_perinuclear:
        try:
            zip_path = os.path.join(sample_folder, 'perinuclear_segmentations/true_perinuclear_instance_masks.npz')
            if in_memory:
                with np.load(zip_path, allow_pickle=False) as npz:
                    perinuclear_instance_data = npz['data']
            else:
                path = os.path.join(sample_folder, 'perinuclear_segmentations/true_perinuclear_instance_mask_extracted_npz/data.npy')
                if not Path(path).exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(path))
                perinuclear_instance_data = np.load(path, mmap_mode='r')
        except Exception as e:
            print(f"Warning: Failed to load perinuclear data. Error: {e}")

    if load_transition:
        try:
            zip_path = os.path.join(sample_folder, 'perinuclear_segmentations/perinuclear_transition_instance_masks.npz')
            if in_memory:
                with np.load(zip_path, allow_pickle=False) as npz:
                    transition_instance_data = npz['data']
            else:
                path = os.path.join(sample_folder, 'perinuclear_segmentations/perinuclear_transition_instance_mask_extracted_npz/data.npy')
                if not Path(path).exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(path))
                transition_instance_data = np.load(path, mmap_mode='r')
        except Exception as e:
            print(f"Warning: Failed to load transition data. Error: {e}")
    if load_combined:
        try:
            zip_path = os.path.join(sample_folder, 'perinuclear_segmentations/full_combined_perinuclear_instance_masks.npz')
            if in_memory:
                with np.load(zip_path, allow_pickle=False) as npz:
                    combined_perinuclear_instance_data = npz['data']
            else:
                path = os.path.join(sample_folder, 'perinuclear_segmentations/full_combined_perinuclear_instance_mask_extracted_npz/data.npy')
                if not Path(path).exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(path))
                combined_perinuclear_instance_data = np.load(path, mmap_mode='r')
        except Exception as e:
            print(f"Warning: Failed to load combined perinuclear/transition data. Error: {e}")

    return instance_data, pos_matching_graph, telenuclear_instance_data, perinuclear_instance_data, transition_instance_data, combined_perinuclear_instance_data





#### --------------------------------
#### Properties Calculation Functions
#### --------------------------------


def general_morphological_metrics(instance_mask_data,  micron_per_pixel, time_step, telenuclear_mask_data=None, perinuclear_mask_data=None, transition_mask_data=None, combined_perinuclear_mask_data=None):
    """
    Compute object-level morphological features per frame and optionally assign spatial regions.

    Args:
        instance_mask_data (np.ndarray): 3D array of instance labels over time.
        micron_per_pixel (float): Spatial resolution for conversion.
        time_step (float): Time between each frame in seconds.
        telenuclear_mask_data (np.ndarray): Optional telenuclear region masks.
        perinuclear_mask_data (np.ndarray): Optional perinuclear region masks.
        transition_mask_data (np.ndarray): Optional transition region masks.
        combined_perinuclear_mask_data (np.ndarray): Optional perinuclear/transition region combined masks.

    Returns:
        pd.DataFrame: Combined DataFrame of per-object features over time.
    """

    # Properties to Analyse
    analyse_properties = ['label', 'area', 'perimeter', 'convex_area', 'bbox_area', 'extent', 'solidity', 'eccentricity', 'orientation', 'axis_major_length', 'axis_minor_length', 'centroid']
    
    # Store to Store Per Frame Calculated Properties
    slice_properties = []
    
    # Iterate over each time frame (i.e., each 2D slice)
    for t in tqdm(range(instance_mask_data.shape[0]), desc='Loading'):
        # Extract the slice for the current time point
        instance_slice_2d = instance_mask_data[t, :, :]
        
        # Compute region properties for the current slice
        props = regionprops_table(instance_slice_2d, properties = analyse_properties)
        
        # Convert to DataFrame and add the time index
        df = pd.DataFrame(props)
    
        df['Time (Seconds)'] = t * time_step  # Add a time column to track which frame the properties come from
        
        # Assign Region
        region_labels = []
        if telenuclear_mask_data is not None or perinuclear_mask_data is not None or transition_mask_data is not None or combined_perinuclear_mask_data is not None:
            for label in df['label']:
                if np.any(perinuclear_mask_data[t] == label):
                    region_labels.append("Perinuclear")
                elif np.any(telenuclear_mask_data[t] == label):
                    region_labels.append("Telenuclear")
                elif np.any(transition_mask_data[t] == label):
                    region_labels.append("Transition")
                elif np.any(combined_perinuclear_mask_data[t] == label):
                    region_labels.append("Perinuclear (Combined)")
                else:
                    region_labels.append("Unassigned")  # optional fallback

            df['Region'] = region_labels
        
        # Append to the list of all properties
        slice_properties.append(df)
    
    # Combine all DataFrames into one
    result_df = pd.concat(slice_properties, ignore_index=True)
    # Convert Column Titles to Title Case
    result_df.columns = result_df.columns.str.title()
    # Removes _ from Column Titles
    result_df = result_df.rename(columns=lambda name: name.replace('_', ' '))
    # Rename Column Titles for Clarity
    result_df = result_df.rename(
        columns={"Bbox Area": "Bounding Box Area", 
                 "Centroid-0": "Centroid (Y)", 
                 "Centroid-1": "Centroid (X)"})
    
    # Move "Time (Seconds)" to the First Column
    time_column = result_df.pop('Time (Seconds)')
    result_df.insert(0, 'Time (Seconds)', time_column)
    
    # Find the Initial Area of Each Object for Normalisation Calculation
    first_time_step_area = result_df.groupby('Label', group_keys=False).apply(lambda x: x.loc[x['Time (Seconds)'].idxmin(), 'Area']).to_dict()
    # Calculate Normalised Area to Determine Mass Conservation of Mitochondria
    result_df['Area (Normalised)'] = result_df.apply(lambda row: row['Area']/first_time_step_area[row['Label']], axis=1)

    # Move "Area (Normalised)" to the Fourth Column Next to "Area" Column
    area_normalised_column = result_df.pop('Area (Normalised)')
    result_df.insert(3, 'Area (Normalised)', area_normalised_column)
    
    if telenuclear_mask_data is not None or perinuclear_mask_data is not None or transition_mask_data is not None or combined_perinuclear_mask_data is not None:
        region_column = result_df.pop('Region')
        result_df.insert(2, 'Region', region_column)

        # Move "Centroid (X)" to the 13th Column Before "Centroid (Y)" Column
        centroid_x_column = result_df.pop('Centroid (X)')
        result_df.insert(14, 'Centroid (X)', centroid_x_column)

    else:
        # Move "Centroid (X)" to the 13th Column Before "Centroid (Y)" Column
        centroid_x_column = result_df.pop('Centroid (X)')
        result_df.insert(13, 'Centroid (X)', centroid_x_column)
        
    # Convert area-related features (squared units)
    result_df['Area'] *= micron_per_pixel**2
    result_df['Bounding Box Area'] *= micron_per_pixel**2
    result_df['Convex Area'] *= micron_per_pixel**2
    
    # Convert length-related features (linear units)
    result_df['Perimeter'] *= micron_per_pixel
    result_df['Axis Major Length'] *= micron_per_pixel
    result_df['Axis Minor Length'] *= micron_per_pixel


    # Compute displacement and velocity
    result_df = result_df.sort_values(by=['Label', 'Time (Seconds)'])
    result_df[['Velocity (X)', 'Velocity (Y)', 'Velocity (Magnitude)', 'Displacement (X)', 'Displacement (Y)', 'Displacement (Magnitude)', 'Direction (Radians)', 'Direction (Degrees)']] = np.nan

    for label, group in result_df.groupby('Label'):
        group = group.sort_values('Time (Seconds)')
        dx = group['Centroid (X)'].diff()
        dy = group['Centroid (Y)'].diff()
        dt = group['Time (Seconds)'].diff()

        # Velocity (microns/second)
        vx = dx / dt * micron_per_pixel
        vy = dy / dt * micron_per_pixel
        displacement = np.sqrt(dx**2 + dy**2)
        velocity_mag = displacement / dt

        direction_rad = np.arctan2(vy, vx)
        direction_deg = np.degrees(direction_rad)

        result_df.loc[group.index, 'Velocity (X)'] = vx
        result_df.loc[group.index, 'Velocity (Y)'] = vy
        result_df.loc[group.index, 'Velocity (Magnitude)'] = velocity_mag
        result_df.loc[group.index, 'Displacement (X)'] = dx * micron_per_pixel
        result_df.loc[group.index, 'Displacement (Y)'] = dy * micron_per_pixel
        result_df.loc[group.index, 'Displacement (Magnitude)'] = displacement
        result_df.loc[group.index, 'Direction (Radians)'] = direction_rad
        result_df.loc[group.index, 'Direction (Degrees)'] = direction_deg

    return result_df



def total_area_per_frame_metrics(result_df, micron_per_pixel=1.0):
    """
    Calculate total and average object area per frame, and normalize to initial values.

    Args:
        result_df (pd.DataFrame): DataFrame with per-object features over time.
        micron_per_pixel (float): Spatial resolution for area scaling.

    Returns:
        pd.DataFrame: Per-frame area statistics including normalization.
    """

    # Calculate Total Mitochondrial Area per Frame
    total_area_per_frame = result_df.groupby('Time (Seconds)')['Area'].agg(
        Total_Area='sum',
        Average_Area='mean',
        Object_Count='count'
    ).reset_index()
    
    # Rename the columns for clarity
    total_area_per_frame.columns = ['Time (Seconds)', 'Total Mitochondrial Area', 'Average Mitochondrial Area', 'Mitochondria Count']

    total_area_per_frame = total_area_per_frame.sort_values("Time (Seconds)").reset_index(drop=True)
    nonzero_area_df = total_area_per_frame[total_area_per_frame['Total Mitochondrial Area'] > 0.0]

    if not nonzero_area_df.empty:
        total_area_at_start = nonzero_area_df.iloc[0]['Total Mitochondrial Area']
        average_area_at_start = nonzero_area_df.iloc[0]['Average Mitochondrial Area']
        initial_time = nonzero_area_df.iloc[0]['Time (Seconds)']
    else:
        total_area_at_start = np.nan
        average_area_at_start = np.nan
        initial_time = np.nan

    if not np.isnan(total_area_at_start) and total_area_at_start > 0.0:
        total_area_per_frame['Total Mitochondrial Area (Normalised)'] = total_area_per_frame['Total Mitochondrial Area']/total_area_at_start
    else:
        total_area_per_frame['Total Mitochondrial Area (Normalised)'] = np.nan

    if not np.isnan(average_area_at_start) and average_area_at_start > 0.0:
        total_area_per_frame['Average Mitochondrial Area (Normalised)'] = total_area_per_frame['Average Mitochondrial Area']/average_area_at_start
    else:
        total_area_per_frame['Average Mitochondrial Area (Normalised)'] = np.isnan

    # # Find Total Mitochondrial Area at Time=0 Seconds
    # total_area_at_start = total_area_per_frame.loc[total_area_per_frame['Time (Seconds)'] == 0.0, 'Total Mitochondrial Area'].values[0]
    
    # # Calculate Total Mitochondrial Area Normalised to Initital Total Mitochondrial Area
    # total_area_per_frame['Total Mitochondrial Area (Normalised)'] = total_area_per_frame['Total Mitochondrial Area']/total_area_at_start
    
    # # Find Average Mitochondrial Area at Time=0 Seconds
    # average_area_at_start = total_area_per_frame.loc[total_area_per_frame['Time (Seconds)'] == 0.0, 'Average Mitochondrial Area'].values[0]
    
    # # Calculate Total Mitochondrial Area Normalised to Initital Total Mitochondrial Area
    # total_area_per_frame['Average Mitochondrial Area (Normalised)'] = total_area_per_frame['Average Mitochondrial Area']/average_area_at_start

    # Convert area-related features (squared units)
    total_area_per_frame['Total Mitochondrial Area'] *= micron_per_pixel**2
    total_area_per_frame['Average Mitochondrial Area'] *= micron_per_pixel**2

    print(f"Normalization reference taken from time = {initial_time} seconds")

    return total_area_per_frame




#### ------------------------------------
#### Fission and Fusion Rate Calculations
#### ------------------------------------


def fission_fusion_rate_over_time(instance_mask_data, mechanism_dataframe, total_area_per_frame, time_step, bin_size_seconds=60, time_units="seconds", domain_specific=False, output_csv_path=None):
    """
    Calculate fission and fusion event rates over time using binning.

    Args:
        instance_mask_data (np.ndarray): Labeled instances across frames.
        mechanism_df (pd.DataFrame): DataFrame containing mechanism events.
        total_area_per_frame_dataframe (pd.DataFrame): Total area per frame.
        time_step (float): Time between frames in seconds.
        total_frames (float): Total frame count.
        bin_size_seconds (float): Bin size in seconds (default: 60s).
        time_units (str): "seconds" or "minutes".
        domain_specific (bool): If True, only include validated domain-specific events.
        output_csv_path (str): Path to the output CSV file for accumulation.

    Returns:
        pd.DataFrame: Binned rates per time interval.
    """

    if time_units == "minutes":
        time_unit_scaling = 1.0 / 60.0
        label = "Time (Minutes)"
    else:
        time_unit_scaling = 1.0
        label = "Time (Seconds)"

    fusion_mask = mechanism_dataframe['Generation Mechanism'].str.contains('Fusion', na=False)
    fission_mask = mechanism_dataframe['Termination Mechanism'].str.contains('Fission', na=False)

    if domain_specific:
        fusion_mask &= mechanism_dataframe['Generation Mechanism Valid']
        fission_mask &= mechanism_dataframe['Termination Mechanism Valid']

    fusion_df = mechanism_dataframe[fusion_mask]
    fission_df = mechanism_dataframe[fission_mask]

    fusion_times = fusion_df['Time of Generation (Seconds)']
    fission_times = fission_df['Time of Termination (Seconds)']

    total_time = instance_mask_data.shape[0] * time_step

    bins = np.arange(0, total_time + bin_size_seconds, bin_size_seconds)
    bin_labels = bins[1:] * time_unit_scaling

    fusion_counts, _ = np.histogram(fusion_times, bins=bins)
    fission_counts, _ = np.histogram(fission_times, bins=bins)

    bin_duration = bin_size_seconds * time_unit_scaling
    fusion_rates = fusion_counts / bin_duration
    fission_rates = fission_counts / bin_duration

    total_area_per_frame = total_area_per_frame.sort_values("Time (Seconds)")
    frame_times = total_area_per_frame['Time (Seconds)'].values
    areas = total_area_per_frame['Total Mitochondrial Area'].values
    total_area_per_frame = total_area_per_frame.sort_values('Time (Seconds)').reset_index(drop=True)

    mean_area_per_bin, _, _ = binned_statistic(frame_times, areas, statistic='mean', bins=bins)
    std_area_per_bin, _, _ = binned_statistic(frame_times, areas, statistic='std', bins=bins)
    # area_bin_indices = np.digitize(frame_times, bins) - 1
    # mean_area_per_bin = np.zeros(len(bin_labels))
    # counts_per_bin = np.zeros(len(bin_labels))
    # area_per_bin = np.zeros(len(bin_labels))

    # for i, idx in enumerate(area_bin_indices):
    #     if 0 <= idx < len(area_per_bin):
    #         area_per_bin[idx] += areas[i]
    #         counts_per_bin[idx] += 1

    # mean_area_per_bin = np.divide(area_per_bin, counts_per_bin)

    # global_mean_mass = areas.mean()
    # global_std_mass = areas.std()
    # global_initial_mass = total_area_per_frame.loc[total_area_per_frame['Time (Seconds)'] == 0.0, 'Total Mitochondrial Area'].values[0]

    nonzero_mass_df = total_area_per_frame[total_area_per_frame['Total Mitochondrial Area'] > 0.0]
    if not nonzero_mass_df.empty:
        global_initial_mass = nonzero_mass_df.iloc[0]['Total Mitochondrial Area']
        initial_mass_time = nonzero_mass_df.iloc[0]['Time (Seconds)']
    else:
        global_initial_mass = np.nan
        initial_mass_time = np.nan

    # Normalised Rates
    normalised_fusion_rate_mean_mass = np.divide(fusion_rates, mean_area_per_bin)
    normalised_fission_rate_mean_mass = np.divide(fission_rates, mean_area_per_bin)

    normalised_fusion_rate_initial_mass = fusion_rates / global_initial_mass
    normalised_fission_rate_initial_mass = fission_rates / global_initial_mass

    binned_df = pd.DataFrame({label: bin_labels,
                            'Fusion Events': fusion_counts,
                            'Fission Events': fission_counts,
                            'Mean Mass': mean_area_per_bin,
                            'Std Mass': std_area_per_bin,
                            'Initial Mass': global_initial_mass,
                            'Initial Mass Timpoint (Seconds)': initial_mass_time,
                            'Normalized Fusion Rate (Mean Total Mass)': normalised_fusion_rate_mean_mass,
                            'Normalized Fission Rate (Mean Total Mass)': normalised_fission_rate_mean_mass,
                            'Normalized Fusion Rate (Initial Total Mass)': normalised_fusion_rate_initial_mass,
                            'Normalized Fission Rate (Initial Total Mass)': normalised_fission_rate_initial_mass
                            })

    if os.path.exists(output_csv_path):
        binned_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        binned_df.to_csv(output_csv_path, index=False)

    print(f"Saved binned fission and Fusion Rates to {output_csv_path}")




def fission_fusion_rate_calculation(instance_mask_data, mechanism_dataframe, total_area_per_frame_dataframe, time_step, domain_specific=False, time_units="seconds"):
    """
    Calculate global or domain-specific average fission and fusion rates.

    Parameters:
    - instance_mask_data: np.ndarray of labeled instances across frames.
    - mechanism_dataframe: pd.DataFrame with event metadata.
    - total_area_per_frame_dataframe: pd.DataFrame with total area per frame.
    - time_step: float, time between frames in seconds.
    - domain_specific: bool, if True, only include validated domain-specific events.
    """

    # Fusion filter
    fusion_mask = mechanism_dataframe['Generation Mechanism'].str.contains('Fusion', na=False)
    if domain_specific:
        fusion_mask &= mechanism_dataframe['Generation Mechanism Valid']

    fusion_df = mechanism_dataframe[fusion_mask]
    fusion_times = sorted(fusion_df['Time of Generation (Seconds)'])
    unique_fusion_events = len(fusion_df)
    print(f'Number of Fusion Events: {unique_fusion_events}')

    # Fission filter
    fission_mask = mechanism_dataframe['Termination Mechanism'].str.contains('Fission', na=False)
    if domain_specific:
        fission_mask &= mechanism_dataframe['Termination Mechanism Valid']

    fission_df = mechanism_dataframe[fission_mask]
    fission_times = sorted(fission_df['Time of Termination (Seconds)'])
    unique_fission_events = len(fission_df)
    print(f'Number of Fission Events: {unique_fission_events} \n')

    # Total Imaging Time
    if time_units == "minutes":
        time_unit_scaling = 1.0 / 60.0
    else:
        time_unit_scaling = 1.0 # Default: Seconds
    total_time = instance_mask_data.shape[0] * time_step * time_unit_scaling

    # Average Rates
    average_fusion_rate = unique_fusion_events / total_time
    average_fission_rate = unique_fission_events / total_time
    print(f'Average Fusion Rate: {average_fusion_rate:.6f} Fusion Events per {time_units[:-1].capitalize()}')
    print(f'Average Fission Rate: {average_fission_rate:.6f} Fission Events per {time_units[:-1].capitalize()} \n')

    # First Nonzero mass for normalisation (Fallback as some domains/regions e.g. transition don't have mass at time 0)
    nonzero_mass_df = total_area_per_frame_dataframe[total_area_per_frame_dataframe['Total Mitochondrial Area'] > 0.0]
    if not nonzero_mass_df.empty:
        initial_mass_per_sample = nonzero_mass_df.iloc[0]['Total Mitochondrial Area']
        initial_mass_time = nonzero_mass_df.iloc[0]['Time (Seconds)']
    else:
        initial_mass_per_sample = np.nan
        initial_mass_time = np.nan

    # Mean and Std of Mitochondrial Area
    mean_mass_per_sample = total_area_per_frame_dataframe['Total Mitochondrial Area'].mean()
    std_mass_per_sample = total_area_per_frame_dataframe['Total Mitochondrial Area'].std()
    # initial_mass_per_sample = total_area_per_frame_dataframe.loc[total_area_per_frame_dataframe['Time (Seconds)'] == 0.0, 'Total Mitochondrial Area'].values[0]
    print(f"Mean Total Mitochondrial Mass per Frame: {mean_mass_per_sample:.2f}")
    print(f"Std Deviation for Mean Total Mitochondrial Mass per Frame: {std_mass_per_sample:.2f} \n")
    print(f"Total Mitochondrial Mass at Start: {initial_mass_per_sample:.2f} \n")


    if np.isnan(initial_mass_per_sample) or initial_mass_per_sample == 0.0:
        normalised_fusion_rate_initial_mass = np.nan
        normalised_fission_rate_initial_mass = np.nan
    else:
        normalised_fusion_rate_initial_mass = average_fusion_rate / initial_mass_per_sample
        normalised_fission_rate_initial_mass = average_fission_rate / initial_mass_per_sample

    # Normalised Rates
    normalised_fusion_rate_mean_mass = average_fusion_rate / mean_mass_per_sample
    normalised_fission_rate_mean_mass = average_fission_rate / mean_mass_per_sample
    # normalised_fusion_rate_initial_mass = average_fusion_rate / initial_mass_per_sample
    # normalised_fission_rate_initial_mass = average_fission_rate / initial_mass_per_sample
    print(f'Normalised Fusion Rate (Mean Total Mass): {normalised_fusion_rate_mean_mass:.6f} Fusion Events per μm² per {time_units[:-1].capitalize()}')
    print(f'Normalised Fission Rate (Mean Total Mass): {normalised_fission_rate_mean_mass:.6f} Fission Events per μm² per {time_units[:-1].capitalize()} \n')
    print(f'Normalised Fusion Rate (Initial Total Mass): {normalised_fusion_rate_initial_mass:.6f} Fusion Events per μm² per {time_units[:-1].capitalize()}')
    print(f'Normalised Fission Rate (Initial Total Mass): {normalised_fission_rate_initial_mass:.6f} Fission Events per μm² per {time_units[:-1].capitalize()} \n')
    print(f"Total Mitochondrial Mass at Start (first non-zero): {initial_mass_per_sample:.2f} at {initial_mass_time:.2f} seconds\n")

    return {
        'fusion_events': unique_fusion_events,
        'fission_events': unique_fission_events,
        'average_fusion_rate': average_fusion_rate,
        'average_fission_rate': average_fission_rate,
        'mean_mass': mean_mass_per_sample,
        'std_mass': std_mass_per_sample,
        'initial_mass': initial_mass_per_sample,
        'normalised_fusion_rate_mean_mass': normalised_fusion_rate_mean_mass,
        'normalised_fission_rate_mean_mass': normalised_fission_rate_mean_mass,
        'normalised_fusion_rate_initial_mass': normalised_fusion_rate_initial_mass,
        'normalised_fission_rate_initial_mass': normalised_fission_rate_initial_mass
    }



def extract_microdomain_fission_fusion_mechanisms(instance_mask_data, mechanism_dataframe):
    """
    For each mechanism, determine whether the object was present in the region
    at its generation and termination frames. Add flags instead of modifying frame values.
    """
    
    # Pre-compute unique labels per frame
    unique_labels_per_frame = {
        frame_idx: set(np.unique(instance_mask_data[frame_idx]))
        for frame_idx in tqdm(range(instance_mask_data.shape[0]), desc='Pre-computing')
    }
    
    # Vectorized operations
    df = mechanism_dataframe.copy()
    
    # Apply checks vectorized
    df['Generation Mechanism Valid'] = df.apply(
        lambda row: row['Label'] in unique_labels_per_frame.get(row['Generation Frame'], set()),
        axis=1
    )
    df['Termination Mechanism Valid'] = df.apply(
        lambda row: row['Label'] in unique_labels_per_frame.get(row['Termination Frame'], set()),
        axis=1
    )
    
    # Filter rows where at least one endpoint is valid
    valid_mask = df['Generation Mechanism Valid'] | df['Termination Mechanism Valid']

    return df[valid_mask].reset_index(drop=True)



def collect_and_save_fission_fusion_rates(sample_id, condition_name, domain_name, rate_dict, output_csv_path):
    """
    Save the calculated fission/fusion data from one sample and domain into a cumulative CSV.

    Parameters:
        sample_id (int): Sample number (e.g., 5, 7, 10).
        condition_name (str): String label for the condition (e.g., 'RPE1 WT').
        domain_name (str): Domain label (e.g., 'Global', 'Telenuclear', etc.).
        rate_dict (dict): Dictionary returned by fission_fusion_rate_calculation.
        output_csv_path (str): Path to the output CSV file for accumulation.

    This function will append to an existing file or create a new one.
    """

    row = {
        'Sample': sample_id,
        'Condition': condition_name,
        'Domain': domain_name,
        'Fusion Events': rate_dict['fusion_events'],
        'Fission Events': rate_dict['fission_events'],
        'Mean Mass': rate_dict.get('mean_mass', np.nan),
        'Std Mass': rate_dict.get('std_mass', np.nan),
        'Initial Mass': rate_dict.get('initial_mass', np.nan),
        'Normalized Fusion Rate (Mean Total Mass)': rate_dict['normalised_fusion_rate_mean_mass'],
        'Normalized Fission Rate (Mean Total Mass)': rate_dict['normalised_fission_rate_mean_mass'],
        'Normalized Fusion Rate (Initial Total Mass)': rate_dict['normalised_fusion_rate_initial_mass'],
        'Normalized Fission Rate (Initial Total Mass)': rate_dict['normalised_fission_rate_initial_mass']
    }

    df = pd.DataFrame([row])

    if os.path.exists(output_csv_path):
        df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_csv_path, index=False)



def process_domain(sample_number, condition, domain_name, domain_mask, mechanism_df, time_step, micron_per_pixel, output_dir, bin_size_seconds, time_units):
    result_df = general_morphological_metrics(domain_mask, micron_per_pixel, time_step)
    total_area = total_area_per_frame_metrics(result_df, micron_per_pixel=micron_per_pixel)
    domain_mechanism_df = extract_microdomain_fission_fusion_mechanisms(domain_mask, mechanism_df)
    rate_dict = fission_fusion_rate_calculation(domain_mask, domain_mechanism_df, total_area, time_step, domain_specific=True, time_units=time_units)
    collect_and_save_fission_fusion_rates(sample_number, condition, domain_name, rate_dict, f"{output_dir}/fission_fusion_summary.csv")
    fission_fusion_rate_over_time(domain_mask, domain_mechanism_df, total_area, time_step, bin_size_seconds=bin_size_seconds, time_units=time_units, domain_specific=True, output_csv_path=f"{output_dir}/{domain_name}_fission_fusion_rates_over_time.csv")
    
    del result_df, total_area, domain_mechanism_df, rate_dict
    gc.collect()




#### ------------------------
#### Main to Process Data
#### ------------------------


def main(sample_number=1, condition_name="WT", time_step=0.5, pixel_res=0.108, root_data_directory="", output_base_directory="processed_outputs", domains=["global"], bin_size_seconds=60, time_units="seconds", in_memory=False):
    """
    Main processing pipeline to extract features from sample data and save outputs.

    Args:
        sample_number (int): Sample ID number to process.
        time_step (float): Time step in seconds between frames.
        pixel_res (float): Pixel resolution in microns.
        root_data_directory (str): Optional base directory for sample folders (default current directory of script).
        domains (list): List of domains to analyze from ["global", "telenuclear", "perinuclear", "transition"]
        time_units (str): 
        in_memory (bool): Load all data into memory (useful for HPC data processing, True) or extract data and memory-map to reduce memory overhead (useful for local data processing with limitewd RAM, False)

    Returns:
        None
    """

    # Determine what data to load based on selected domains
    load_telenuclear = "telenuclear" in domains
    load_perinuclear = "perinuclear" in domains  
    load_transition = "transition" in domains
    load_combined = "combined_perinuclear" in domains

    # Assign Sample Folders
    sample_folder, analysis_output_folder = setup_data_path(sample_number=sample_number, root_data_directory=root_data_directory, output_base_directory=output_base_directory)

    # Remove existing file and start fresh
    csv_output_path = f"{analysis_output_folder}/fission_fusion_summary.csv"
    if os.path.exists(csv_output_path):
        os.remove(csv_output_path)

    # Load Data
    instance_data, pos_matching_graph, telenuclear_instance_data, perinuclear_instance_data, transition_instance_data, combined_perinuclear_instance_data = load_data(
        sample_folder,
        load_telenuclear=load_telenuclear,
        load_perinuclear=load_perinuclear,
        load_transition=load_transition,
        load_combined=load_combined,
        in_memory=in_memory
        )

    # Load Fission and Fusion Mechanism Dataframe
    mechanism_path = os.path.join(analysis_output_folder, "fission_fusion_mechanisms.csv")
    mechanism_df = pd.read_csv(mechanism_path)

    global_total_area_path = os.path.join(analysis_output_folder, "total_area_per_frame_metrics.csv")
    total_area_per_frame = pd.read_csv(global_total_area_path)

    # Pixel Scaling in Microns Per Pixel
    micron_per_pixel = pixel_res/2 

    

    # Fission and Fusion Rate Calculations
    # Process selected domains
    if "global" in domains:
        print("Processing Global domain...")
        global_rate_dict = fission_fusion_rate_calculation(instance_data, mechanism_df, total_area_per_frame, time_step=time_step, domain_specific=False, time_units=time_units)
        collect_and_save_fission_fusion_rates(sample_number, condition_name, "Global", global_rate_dict, f"{analysis_output_folder}/fission_fusion_summary.csv")
        fission_fusion_rate_over_time(instance_data, mechanism_df, total_area_per_frame, time_step, bin_size_seconds=bin_size_seconds, time_units=time_units, domain_specific=False, output_csv_path=f"{analysis_output_folder}/Global_fission_fusion_rates_over_time.csv")
    
    if "telenuclear" in domains and telenuclear_instance_data is not None:
        print("Processing Telenuclear domain...")
        time1 = time.time()
        process_domain(sample_number, condition_name, "Telenuclear", telenuclear_instance_data, mechanism_df, time_step, micron_per_pixel, analysis_output_folder, bin_size_seconds, time_units)
        elapsed_time1 = time.time()-time1

        del telenuclear_instance_data
        gc.collect()

        print(f"Processed Telenuclear Domain in {elapsed_time1} Seconds")
    
    if "perinuclear" in domains and perinuclear_instance_data is not None:
        print("Processing Perinuclear domain...")
        time2 = time.time()
        process_domain(sample_number, condition_name, "Perinuclear", perinuclear_instance_data, mechanism_df, time_step, micron_per_pixel, analysis_output_folder, bin_size_seconds, time_units)
        elapsed_time2 = time.time()-time2

        del perinuclear_instance_data
        gc.collect()

        print(f"Processed Perinuclear Domain in {elapsed_time2} Seconds")
    
    if "transition" in domains and transition_instance_data is not None:
        print("Processing Transition domain...")
        time3 = time.time()
        process_domain(sample_number, condition_name, "Transition", transition_instance_data, mechanism_df, time_step, micron_per_pixel, analysis_output_folder, bin_size_seconds, time_units)
        elapsed_time3 = time.time()-time3

        del transition_instance_data
        gc.collect()

        print(f"Processed Transition Domain in {elapsed_time3} Seconds")

    if "combined_perinuclear" in domains and combined_perinuclear_instance_data is not None:
        print("Processing Perinuclear (Combined) domain...")
        time4 = time.time()
        process_domain(sample_number, condition_name, "Perinuclear (Combined)", combined_perinuclear_instance_data, mechanism_df, time_step, micron_per_pixel, analysis_output_folder, bin_size_seconds, time_units)
        elapsed_time4 = time.time()-time4

        del combined_perinuclear_instance_data
        gc.collect()

        print(f"Processed Perinuclear (Combined) Domain in {elapsed_time4} Seconds")

    del instance_data, pos_matching_graph, mechanism_df, total_area_per_frame
    gc.collect()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Script to compute fission/fusion rates across domains for mitochondrial segmentation data.")
    parser.add_argument("--samples", nargs="+", type=int, help="List of sample numbers to analyse")
    parser.add_argument("--conditions", nargs="+", type=str, help="List of condition names aligned with the samples")

    parser.add_argument("--time_step", type=float, default=0.5, help="Time step between frames in Seconds")
    parser.add_argument("--pixel_res",type=float, default=0.108, help="xy resolution in Microns per Pixel")
    parser.add_argument("--root_data_directory", type=str, help="Parent/root directory where all sample folders are located")
    parser.add_argument("--output_base_directory", type=str, default="processed_outputs", help="Base directory to store all processed outputs (default: processed_outputs)")
    parser.add_argument("--domains", nargs="+", choices=["global", "telenuclear", "perinuclear", "transition", "combined_perinuclear"], default=["global"], help="Select which domains to analyze (default: global only)")
    parser.add_argument("--bin_size_seconds", type=float, default=60.0, help="Bin size for calculating over time fission and fusion rates")
    parser.add_argument("--time_units", choices=["minutes", "seconds"], default="seconds", help="Time units for rate calculations")
    parser.add_argument("--in_memory", action="store_true", help="Load all data into memory (useful for HPC data processing, True) or extract data and memory-map to reduce memory overhead (useful for local data processing with limitewd RAM, False)")

    args=parser.parse_args()

    # Process Sample
    if args.samples and args.conditions:
        if len(args.samples) != len(args.conditions):
            raise ValueError("The number of samples must match the number of conditions")

        for sample, condition in zip(args.samples, args.conditions):
            print(f"Running Sample {sample} ({condition}) for domains: {args.domains}")
            main(
                sample_number=sample,
                condition_name=condition,
                time_step=args.time_step,
                pixel_res=args.pixel_res,
                root_data_directory=args.root_data_directory,
                output_base_directory=args.output_base_directory,
                domains=args.domains,
                bin_size_seconds=args.bin_size_seconds,
                time_units=args.time_units,
                in_memory=args.in_memory
            )
    else:
        raise ValueError("Both --samples and --conditions must be specified.")
    
    # python 5_3_fission_fusion_rate_calculator.py --samples 33 --conditions "WT" --time_step 0.5 --pixel_res 0.108 --root_data_directory "input_data/" --output_base_directory "processed_outputs/" --domains global telenuclear perinuclear combined_perinuclear transition --bin_size_seconds 60 --time_units minutes

