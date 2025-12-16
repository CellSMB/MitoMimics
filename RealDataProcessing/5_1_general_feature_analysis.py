


#### -------
#### Imports
#### -------


# Standard library imports
import argparse
import os
import pickle
import zipfile
from pathlib import Path
import gc

# Third-party imports
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm



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




def total_area_per_frame_metrics(result_df):#, micron_per_pixel=1.0):
    """
    Calculate total and average object area per frame, and normalize to initial values.

    Args:
        result_df (pd.DataFrame): DataFrame with per-object features over time. Assumes result_df['Area'] is already in physical units (e.g. μm²).
        # micron_per_pixel (float): Spatial resolution for area scaling.

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
    
    # Find Total Mitochondrial Area at Time=0 Seconds
    total_area_at_start = total_area_per_frame.loc[total_area_per_frame['Time (Seconds)'] == 0.0, 'Total Mitochondrial Area'].values[0]
    
    # Calculate Total Mitochondrial Area Normalised to Initital Total Mitochondrial Area
    total_area_per_frame['Total Mitochondrial Area (Normalised)'] = total_area_per_frame['Total Mitochondrial Area']/total_area_at_start
    
    # Find Average Mitochondrial Area at Time=0 Seconds
    average_area_at_start = total_area_per_frame.loc[total_area_per_frame['Time (Seconds)'] == 0.0, 'Average Mitochondrial Area'].values[0]
    
    # Calculate Total Mitochondrial Area Normalised to Initital Total Mitochondrial Area
    total_area_per_frame['Average Mitochondrial Area (Normalised)'] = total_area_per_frame['Average Mitochondrial Area']/average_area_at_start

    # Convert area-related features (squared units)
    # total_area_per_frame['Total Mitochondrial Area'] *= micron_per_pixel**2
    # total_area_per_frame['Average Mitochondrial Area'] *= micron_per_pixel**2

    return total_area_per_frame



def endpoint_branching_properties(pos_matching_graph, time_step=1.0):
    """
    Calculate endpoint and branching counts per object and per frame.

    Args:
        pos_matching_graph (networkx.Graph): Graph of object skeletons over time.
        time_step (float): Time between frames in seconds.

    Returns:
        tuple: 
            - Per-object endpoint/branching metrics per frame.
            - Total endpoint/branching summary metrics per frame.
    """

    # Split Up Individual Skeleton Graph into Per Frame Connection Graph i.e. Remove Connections between multiple frames to make calculations of individual frames easier
    edges_per_frame = [(node1, node2)
                    for node1, node2 in pos_matching_graph.edges() 
                    if node1[2]==node2[2]]
    edges_per_frame_subgraph = pos_matching_graph.edge_subgraph(edges_per_frame).copy()

    # Get all Per Frame Node Degrees in One Step
    node_degrees = dict(edges_per_frame_subgraph.degree())

    # Determine Mitochondrial Endpoint Locations per Frame
    endpoints_per_frame = [node 
                        for node, degree in node_degrees.items() 
                        if degree==1]
    endpoints_per_frame_subgraph = edges_per_frame_subgraph.subgraph(endpoints_per_frame).copy()

    # Determine Mitochondrial Branch Point Locations per Frame
    branches_per_frame = [node 
                        for node, degree in node_degrees.items() 
                        if degree>2]
    branches_per_frame_subgraph = edges_per_frame_subgraph.subgraph(branches_per_frame).copy()

    # Determine Mitochondrial Branch Point Locations per Frame
    branches_and_endpoints_per_frame = [node 
                                        for node, degree in node_degrees.items() 
                                        if degree>2 or degree==1]

    branches_endpoints_per_frame_subgraph = edges_per_frame_subgraph.subgraph(branches_and_endpoints_per_frame).copy()

    # Dictionary to Store Per Frame Calculated Enpoint and Branching Properties
    endpoint_branch_properties = {'Time (Seconds)': [], 'Label': [], 'Endpoint Count': [], 'Branch Count': []}

    # Calculate Endpoint Counts per Object per Frame
    for endpoint_node in endpoints_per_frame:
        label = edges_per_frame_subgraph.nodes[endpoint_node].get('label', None)
        frame = endpoint_node[2]

        if label is not None:
            endpoint_branch_properties['Time (Seconds)'].append(frame*time_step)
            endpoint_branch_properties['Label'].append(label)
            endpoint_branch_properties['Endpoint Count'].append(1)
            endpoint_branch_properties['Branch Count'].append(0)

    # Calculate Branch Counts per Object per Frame
    for branch_node in branches_per_frame:
        label = edges_per_frame_subgraph.nodes[branch_node].get('label', None)
        frame = branch_node[2]

        if label is not None:
            endpoint_branch_properties['Time (Seconds)'].append(frame*time_step)
            endpoint_branch_properties['Label'].append(label)
            endpoint_branch_properties['Endpoint Count'].append(0)
            endpoint_branch_properties['Branch Count'].append(1)

    # Create DataFrame and Reorganise
    endpoint_branch_df = pd.DataFrame(endpoint_branch_properties)
    endpoint_branch_df_grouped = endpoint_branch_df.groupby(['Label', 'Time (Seconds)'], as_index=False).sum()

    # Calculate Total Branching Data per Frame
    total_branches_per_frame = endpoint_branch_df.groupby('Time (Seconds)')['Branch Count'].agg(
        Total_Branches='sum',
        Average_Branches='mean'
    ).reset_index()

    # Calculate Total Endpoint Data per Frame
    total_endpoints_per_frame = endpoint_branch_df.groupby('Time (Seconds)')['Endpoint Count'].agg(
        Total_Endpoints='sum',
        Average_Endpoints='mean',
    ).reset_index()

    # Merge Both Total Branching and Endpoint Calculation Data Frames
    total_endpoint_branch_per_frame_df = total_endpoints_per_frame.merge(total_branches_per_frame, on='Time (Seconds)', how='outer')
    # Rename Column Titles for Clarity
    total_endpoint_branch_per_frame_df = total_endpoint_branch_per_frame_df.rename(
        columns={'Total_Endpoints': 'Total Endpoint Count', 
                'Average_Endpoints': 'Average Endpoints', 
                'Total_Branches': 'Total Branch Count', 
                'Average_Branches': 'Average Branches'})
    
    return endpoint_branch_df_grouped, total_endpoint_branch_per_frame_df




#### ------------------------
#### Main to Process Data
#### ------------------------


def main(sample_number=1, time_step=0.5, pixel_res=0.108, root_data_directory="", output_base_directory="processed_outputs", load_telenuclear=False, load_perinuclear=False, load_transition=False, load_combined=False, in_memory=False):
    """
    Main processing pipeline to extract features from sample data and save outputs.

    Args:
        sample_number (int): Sample ID number to process.
        time_step (float): Time step in seconds between frames.
        pixel_res (float): Pixel resolution in microns.
        root_data_directory (str): Optional base directory for sample folders (default current directory of script).
        load_telenuclear (bool): Flag to load telenuclear mask.
        load_perinuclear (bool): Flag to load perinuclear mask.
        load_transition (bool): Flag to load transition mask.
        load_combined (bool): Flag to load combined perinuclear/transition mask.
        in_memory (bool): Load all data into memory (useful for HPC data processing, True) or extract data and memory-map to reduce memory overhead (useful for local data processing with limitewd RAM, False)

    Returns:
        None
    """

    # Assign Sample Folders
    sample_folder, analysis_output_folder = setup_data_path(sample_number=sample_number, root_data_directory=root_data_directory, output_base_directory=output_base_directory)

    # Load Data
    instance_data, pos_matching_graph, telenuclear_instance_data, perinuclear_instance_data, transition_instance_data, combined_perinuclear_instance_data = load_data(
        sample_folder,
        load_telenuclear=load_telenuclear,
        load_perinuclear=load_perinuclear,
        load_transition=load_transition,
        load_combined=load_combined,
        in_memory=in_memory
        )

    # Pixel Scaling in Microns Per Pixel
    micron_per_pixel = pixel_res/2 


    # General morphological calculations
    morphological_features_per_object_df = general_morphological_metrics(
        instance_data,
        micron_per_pixel=micron_per_pixel,
        time_step=time_step,
        telenuclear_mask_data=telenuclear_instance_data,
        perinuclear_mask_data=perinuclear_instance_data,
        transition_mask_data=transition_instance_data,
        combined_perinuclear_mask_data=combined_perinuclear_instance_data
        )
    
    # Save general morphological calculations (regionprops) in .csv spreadsheet
    output_file = f"{analysis_output_folder}/general_morphological_features_per_object.csv"
    morphological_features_per_object_df.to_csv(output_file, index=False)
    print("Saved morphological_features_per_object_df to CSV.")


    # Total and average areas per frame of object mass
    total_area_per_frame_df = total_area_per_frame_metrics(result_df=morphological_features_per_object_df)#, micron_per_pixel=micron_per_pixel)
    
    # Save total and average area metrics per frame in .csv spreadsheet
    output_file = f"{analysis_output_folder}/total_area_per_frame_metrics.csv"
    total_area_per_frame_df.to_csv(output_file, index=False)
    print("Saved total_area_per_frame_df to CSV.")


    # Endpoint and Branching Calculation
    endpoint_branch_per_object_df, total_endpoint_branch_per_frame_df = endpoint_branching_properties(pos_matching_graph=pos_matching_graph, time_step=time_step)

    # Save endpoint and branching counts in .csv spreadsheet
    output_file = f"{analysis_output_folder}/endpoint_branching_per_object_metrics.csv"
    endpoint_branch_per_object_df.to_csv(output_file, index=False)
    print("Saved endpoint_branch_per_object_df to CSV.")

    # Save endpoint and branching metrics at a per frame level in .csv spreadsheet
    output_file = f"{analysis_output_folder}/total_endpoint_branching_per_frame_metrics.csv"
    total_endpoint_branch_per_frame_df.to_csv(output_file, index=False)
    print("Saved total_endpoint_branch_per_frame_df to CSV.")

    del instance_data, pos_matching_graph, telenuclear_instance_data, perinuclear_instance_data, transition_instance_data, combined_perinuclear_instance_data
    del morphological_features_per_object_df, total_area_per_frame_df, endpoint_branch_per_object_df, total_endpoint_branch_per_frame_df

    gc.collect()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="")
    parser.add_argument("--sample", type=int, default=1, help="Sample number to analyse")
    parser.add_argument("--time_step", type=float, default=0.5, help="Time step between frames in Seconds")
    parser.add_argument("--pixel_res",type=float, default=0.108, help="xy resolution in Microns per Pixel")
    parser.add_argument("--root_data_directory", type=str, help="Parent/root directory where all sample folders are located")
    parser.add_argument("--output_base_directory", type=str, default="processed_outputs", help="Base directory to store all processed outputs (default: processed_outputs)")
    parser.add_argument("--load_telenuclear", action="store_true", help="Load telenuclear mask if available")
    parser.add_argument("--load_perinuclear", action="store_true", help="Load perinuclear mask if available")
    parser.add_argument("--load_transition", action="store_true", help="Load transition mask if available")
    parser.add_argument("--load_combined", action="store_true", help="Load combined perinuclear/transition mask if available")
    parser.add_argument("--in_memory", action="store_true", help="Load all data into memory (useful for HPC data processing, True) or extract data and memory-map to reduce memory overhead (useful for local data processing with limitewd RAM, False)")


    args=parser.parse_args()

    # Process Sample
    main(sample_number=args.sample, 
         time_step=args.time_step, 
         pixel_res=args.pixel_res,
         root_data_directory=args.root_data_directory,
         output_base_directory=args.output_base_directory,
         load_telenuclear=args.load_telenuclear,
         load_perinuclear=args.load_perinuclear,
         load_transition=args.load_transition,
         load_combined=args.load_combined,
         in_memory=args.in_memory)
    
    # python 5_1_general_feature_analysis.py --sample 33 --time_step 0.5 --pixel_res 0.108 --root_data_directory "raw/" --output_base_directory "processed_outputs/" --load_telenuclear --load_perinuclear --load_transition --load_combined