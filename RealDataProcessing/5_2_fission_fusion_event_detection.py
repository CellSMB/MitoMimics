


#### -------
#### Imports
#### -------


# Standard libraries
import argparse
import concurrent.futures
import os
import pickle
import zipfile
from pathlib import Path
import gc

# Third-party libraries
import numpy as np
import pandas as pd
import scipy.ndimage
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


def load_data(sample_folder, in_memory=False):
    """
    Load instance mask data and optional sub-region datasets for a given sample.

    Args:
        sample_folder (str): Path to the sample folder.

    Returns:
        tuple: Main instance mask, individual object skeleton masks, and temporal object connection graph.
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


    # LOAD INSTANCE SKELETONS
    # Path to instance skeleton .npz file
    instance_skeleton_npz_file = os.path.join(os.getcwd(), sample_folder, 'instance_skel.npz')
    if in_memory:
        with np.load(instance_skeleton_npz_file, allow_pickle=False) as npz:
            instance_skeleton = npz['arr_0']
    else:
        if not Path(os.path.join(sample_folder, 'instance_skeleton_extracted_npz/arr_0.npy')).exists():
            
            
            # Extract the .npz file manually
            with zipfile.ZipFile(instance_skeleton_npz_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(os.getcwd(), sample_folder, 'instance_skeleton_extracted_npz'))

        # memory-map the individual .npy files in the extracted folder
        instance_skeleton = np.load(os.path.join(os.getcwd(), sample_folder, 'instance_skeleton_extracted_npz', 'arr_0.npy'), mmap_mode='r')



    # Load Graph Structure
    with open(os.path.join(os.getcwd(), sample_folder, 'matching_nx_graph.pickle'), 'rb') as f:
        matching_graph = pickle.load(f)

    
    return instance_data, instance_skeleton, matching_graph




#### ------------------------
#### Detect Fission/Fusion
#### ------------------------


def event_mechanism_detection(matching_graph, instance_data, time_step=0.5):
    """
    Analyze a temporal object matching graph to identify generation and termination mechanisms
    (e.g., Fission, Fusion, Spontaneous) for each tracked object.

    Args:
        matching_graph (networkx.DiGraph): A directed graph where nodes are (frame, object_id) 
                                           and edges represent temporal correspondence.
        time_step (float): Time between frames, used to convert frame indices to seconds.

    Returns:
        pd.DataFrame: DataFrame containing object labels and their generation/termination 
                      frames, times, persistence, and mechanism classifications.
    """
    

    # Dictionary to store the generation and termination mechanism of each object
    generation_termination_mechanisms = {
        'Label': [],
        'Generation Frame': [],
        'Time of Generation (Seconds)': [],
        'Termination Frame': [],
        'Time of Termination (Seconds)': [],
        'Persistence Time (Seconds)': [],
        'Generation Mechanism': [],
        'Termination Mechanism': []
    }

    # Loop over unique object IDs
    object_ids = {node[1] for node in matching_graph.nodes()}

    for object_id in object_ids:
        # Find all nodes associated with this object_id
        nodes_with_same_id = [node for node in matching_graph.nodes() if node[1] == object_id]
        
        # Step 1: Determine Generation Mechanism (First node)
        first_time_step_node = min(nodes_with_same_id, key=lambda node: node[0])
        first_predecessor_nodes = list(matching_graph.predecessors(first_time_step_node))

        if len(first_predecessor_nodes) == 0:
            generation_mechanism = 'Spontaneous'
        elif len(first_predecessor_nodes) == 1:
            generation_mechanism = 'Fission'  # A single predecessor from a different object
        else:
            generation_mechanism = 'Fission/Fusion' if any(len(list(matching_graph.successors(predecessor))) > 1 for predecessor in matching_graph.predecessors(first_time_step_node)) else 'Fusion'
        
        # Step 2: Determine Termination Mechanism (Last node)
        last_time_step_node = max(nodes_with_same_id, key=lambda node: node[0])
        last_successor_nodes = list(matching_graph.successors(last_time_step_node))

        if len(last_successor_nodes) == 0:
            termination_mechanism = 'Spontaneous'
        elif len(last_successor_nodes) == 1:
            termination_mechanism = 'Fusion'  # A single successor with a different object ID
        else:
            termination_mechanism = 'Fission/Fusion' if any(len(list(matching_graph.predecessors(successor))) > 1 for successor in matching_graph.successors(last_time_step_node)) else 'Fission'
        
        # Step 3: Check if object exists at generation and termination frames in the cropped region (if cropped, else nothing changes)
        generation_frame = first_time_step_node[0]
        termination_frame = last_time_step_node[0]

        generation_exists = (0 <= generation_frame < instance_data.shape[0] and np.any(instance_data[generation_frame] == object_id))
        termination_exists = (0 <= termination_frame < instance_data.shape[0] and np.any(instance_data[termination_frame] == object_id))

        if not generation_exists and generation_mechanism in ['Fission', 'Fusion', 'Fission/Fusion']:
            generation_mechanism = 'Cropped'

        if not termination_exists and termination_mechanism in ['Fission', 'Fusion', 'Fission/Fusion']:
            termination_mechanism = 'Cropped'

        # Store the results for this object ID
        generation_termination_mechanisms['Label'].append(object_id)
        generation_termination_mechanisms['Generation Frame'].append(first_time_step_node[0])
        generation_termination_mechanisms['Time of Generation (Seconds)'].append(first_time_step_node[0]*time_step)
        generation_termination_mechanisms['Termination Frame'].append(last_time_step_node[0])
        generation_termination_mechanisms['Time of Termination (Seconds)'].append(last_time_step_node[0]*time_step)
        generation_termination_mechanisms['Persistence Time (Seconds)'].append((last_time_step_node[0]-first_time_step_node[0])*time_step)
        generation_termination_mechanisms['Generation Mechanism'].append(generation_mechanism)
        generation_termination_mechanisms['Termination Mechanism'].append(termination_mechanism)

    mechanism_df = pd.DataFrame(generation_termination_mechanisms)

    return mechanism_df




#### -----------------------------------
#### Find Fission and Fusion Coordinates
#### -----------------------------------


# Function to push points to the nearest boundary while avoiding overshooting
def push_points_to_boundary(fission_point, interior_mask, dist_transform, max_distance=7, step_size=0.5):
    """
    Function to push the fission/fusion point to the nearest boundary while handling potential gaps.
    It will push the point towards the object boundary until a max distance or a boundary is reached.
    
    Parameters:
    - fission_point: The initial point to be pushed towards the boundary.
    - interior_mask: The binary mask representing the interior of the object.
    - dist_transform: The distance transform of the object mask.
    - max_distance: Maximum distance to push the point.
    - step_size: Step size for pushing the point.
    
    Returns:
    - The new position of the fission point after being pushed.
    """
    
    # Extract coordinates of the object’s interior (where dist_transform > 0)
    interior_points = np.argwhere(interior_mask)
    
    # Get the signed distance of each interior point
    dist_vals = dist_transform[interior_points[:, 0], interior_points[:, 1]]
    
    # Find the point that is closest to the fission point
    distances_to_fission = np.linalg.norm(interior_points - fission_point, axis=1)
    closest_point_idx = np.argmin(distances_to_fission)
    closest_point = interior_points[closest_point_idx]
    
    # Initialize the current position and distance
    current_position = closest_point
    distance_travelled = 0
    
    # Keep moving the point until the maximum distance is reached or we hit the gap
    while distance_travelled < max_distance:
        # Check the direction towards the boundary (sign of the distance transform)
        direction_to_boundary = np.sign(dist_transform[current_position[0], current_position[1]])
        
        # Move the point a small step towards the boundary
        new_position = current_position + direction_to_boundary * step_size  # Adjusted step size
        
        # Ensure new_position is valid as an integer index and within the mask boundary
        new_position = np.round(new_position).astype(int)  # Ensure it's an integer for indexing
        new_position = np.clip(new_position, 0, np.array(interior_mask.shape) - 1)  # Prevent out-of-bound indices
        
        # Check if the new position is still within the object boundary
        if dist_transform[new_position[0], new_position[1]] <= 0:
            # If we go outside the object (hit the boundary), stop
            break
        
        # Update current position and distance travelled
        current_position = new_position
        distance_travelled += step_size
    
    return current_position




def fission_fusion_event_coordinate_detection(split_instance_masks, split_skel_masks):
    """
    Function to detect the approximate coordinates of fission/fusion between two or more objects in pairs.
    For objects which fission/fusion into two objects, this algorithm is pretty precise. However, for 
    objects which have more than 1 fission location, this algorithm gets close but needs an additional 
    push to converge coordinates at the real point of fission/fusion.
    
    Parameters:
    - split_instance_masks (list): List of binary instance masks of the split objects.
    - split_skel_masks (list): List of binary skeleton masks of the split objects.
    
    Returns:
    - dict: A dictionary containing the fission/fusion coordinates for the successor/predecessor objects as well as the parent/child object.
    """
    
    # Compute distance transforms for each successor object
    distance_transforms = [scipy.ndimage.distance_transform_edt(mask == 0) for mask in split_skel_masks]
        
    # Compute pairwise combined transforms
    pairwise_transforms = []
    num_splits = len(split_skel_masks)
    
    # Compute the pairwise combined masks and their distance transforms
    for i in range(num_splits):
        for j in range(i + 1, num_splits):
            # Combine the masks for the two objects with logical OR to avoid overlap issues
            comb_fmask = np.logical_or(split_instance_masks[i], split_instance_masks[j])  # Use logical OR to combine masks
            
            # Compute signed distance transform of the combined object
            dist_inside = scipy.ndimage.distance_transform_edt(comb_fmask)
            dist_outside = scipy.ndimage.distance_transform_edt(1 - comb_fmask)
            signed_dist = (dist_inside - dist_outside) * -1
    
            # Compute combined transform (ensure the transforms are distinct)
            combined_transform = distance_transforms[i] + distance_transforms[j] + signed_dist * 2
    
            # Find the minimum value and store the location
            min_value = np.min(combined_transform)
            locations = np.argwhere(combined_transform == min_value)  # Get all min locations
            midpoint = locations[0]
            pairwise_transforms.append((locations, min_value, (i, j)))
            
    # Sort pairwise transforms by min_value to get the smallest ones
    pairwise_transforms.sort(key=lambda x: x[1])  # Sort by the minimum value of each pair
    
    fission_coordinates = {}
    
    # Append the first num_splits-1 minimum locations
    for locations, min_value, (idx1, idx2) in pairwise_transforms[:num_splits-1]:  # Take the first n-1 entries from the sorted list
        fission_point = locations[0]  # Get the first fission point with the lowest min_value
        
        # Check which mask the fission point is contained in then use the opposing mask to push towards
        if split_instance_masks[idx1][fission_point[0], fission_point[1]]:
            # fission point is inside the mask corresponding to idx1
            dist_transform = distance_transforms[idx2]
            interior_mask = split_instance_masks[idx2]  # Mask of the other non-current object
        else:
            # fission point is inside the mask corresponding to idx2
            dist_transform = distance_transforms[idx1]
            interior_mask = split_instance_masks[idx1]  # Mask of the other non-current object
            
        # Push the fission point to the nearest boundary while considering the gap
        pushed_fission_point = push_points_to_boundary(fission_point, interior_mask, dist_transform)
        fission_point_tuple = (int(pushed_fission_point[0]), int(pushed_fission_point[1]))

        # Store the fission point with idx1, idx2 as the key
        # Make sure we consistently store tuples
        if fission_point_tuple in fission_coordinates:
            # If value is not already a list of tuples, convert it to one
            if isinstance(fission_coordinates[fission_point_tuple][0], int):
                # Convert the existing list [idx1, idx2] to a list of tuples [(idx1, idx2)]
                existing_indices = fission_coordinates[fission_point_tuple]
                fission_coordinates[fission_point_tuple] = [(existing_indices[0], existing_indices[1])]
            
            # Now append the new tuple
            fission_coordinates[fission_point_tuple].append((idx1, idx2))
        else:
            # Create a new list with (idx1, idx2) as a tuple
            fission_coordinates[fission_point_tuple] = [(idx1, idx2)]
    return fission_coordinates



def process_row(row, mechanism_type, part_type, matching_graph, instance_data, instance_skeleton):
    """
    Process a single row for calculating fission or fusion coordinates.
    
    Args:
        row: Single row of the dataframe
        mechanism_type (str): Type of mechanism ('Fission' or 'Fusion').
        part_type (str): Either 'Generation' or 'Termination'.
        matching_graph: The matching graph object.
        instance_data: The instance data for image segmentation.
        instance_skeleton: The skeletonized instance data.
    
    Returns:
        dict: A dictionary containing the fission/fusion coordinates for this row.
    """
    
    object_id = row['Label']
    
    # Calculate frame and successor/predecessor IDs based on mechanism type
    if mechanism_type == 'Fission':
        frame = int(2 * row['Time of Termination (Seconds)'])  # For Fission, use Termination Time
        node_ids = list(matching_graph.successors((frame, object_id)))
    elif mechanism_type == 'Fusion':  
        frame = int(2 * row['Time of Generation (Seconds)'])  # For Fusion, use Generation Time
        node_ids = list(matching_graph.predecessors((frame, object_id)))
        
    # Coordinate detection for Fission/Fusion events are non-trivial (not implemented)
    elif mechanism_type == 'Fission/Fusion' and part_type == 'Generation':
        frame = int(2 * row['Time of Generation (Seconds)'])
        node_ids = list(matching_graph.predecessors((frame, object_id)))
    elif mechanism_type == 'Fission/Fusion' and part_type == 'Termination':
        frame = int(2 * row['Time of Termination (Seconds)'])
        node_ids = list(matching_graph.successors((frame, object_id)))
    
    # Filter masks which have been cropped or edited from instance_data
    node_ids = [(frame_time, obj_id) for (frame_time, obj_id) in node_ids if 0 <= frame_time < instance_data.shape[0] and np.any(instance_data[frame_time] == obj_id)]

    # Generate masks for the split instances and skeletons
    instance_masks = [instance_data[node[0], :, :] == node[1] for node in node_ids]
    skeleton_masks = [instance_skeleton[node[0], :, :] == node[1] for node in node_ids]
    
    # Detect the fission/fusion event coordinates
    event_coordinates = fission_fusion_event_coordinate_detection(instance_masks, skeleton_masks)
    
    # Replace idx1, idx2 with the node[1] values for each fission/fusion point
    event_data = {}
    for event_point, idx_tuples in event_coordinates.items():
        try:
            # Handle each tuple of indices properly
            object_ids = []
            for idx_tuple in idx_tuples:
                # If it's already a tuple, extract both indices
                if isinstance(idx_tuple, tuple):
                    idx1, idx2 = idx_tuple
                    object_ids.append(node_ids[idx1][1])
                    object_ids.append(node_ids[idx2][1])
                # If it's a single index, extract that one
                else:
                    object_ids.append(node_ids[idx_tuple][1])
                    
            # Add the original object_id
            object_ids.append(object_id)
            
            # Remove duplicates
            event_data[event_point] = list(set(object_ids))
        except TypeError as e:
            print(f"Failed at event_point={event_point}")
            print(f"node_ids = {node_ids}")
            raise
            
    return event_data



def calculate_event_coordinates_in_chunks(mechanism_type, part_type, matching_graph, instance_data, instance_skeleton, mechanism_df, chunk_size=200):
    """
    General function to calculate fission or fusion coordinates in parallel on the CPU.
    
    Args:
        mechanism_type (str): Type of mechanism ('Fission' or 'Fusion').
        part_type (str): Type of event ('Generation' or 'Termination').
        matching_graph: The matching graph object.
        instance_data: The instance data for image segmentation.
        instance_skeleton: The skeletonized instance data.
        mechanism_df: DataFrame with mechanism information.
        chunk_size (int): Number of chunks to process in parallel (change depending on RAM and CPU Core/Thread availability.
    
    Returns:
        dict: A dictionary containing the fission/fusion coordinates.
    """
    
    mechanism_rows = mechanism_df[mechanism_df[f'{part_type} Mechanism'] == mechanism_type]
    
    all_coordinates = {}
    
    # Split the dataframe into smaller chunks and process each chunk
    for i in tqdm(range(0, len(mechanism_rows), chunk_size), desc=f'Processing {mechanism_type} in Chunks'):
        chunk = mechanism_rows.iloc[i:i + chunk_size]
        
        # Use ThreadPoolExecutor to process each chunk in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _, row in chunk.iterrows():
                futures.append(executor.submit(process_row, row, mechanism_type, part_type, matching_graph, instance_data, instance_skeleton))
            
            # Wait for all futures to complete and collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for event_point, event_data in result.items():
                    if event_point in all_coordinates:
                        all_coordinates[event_point].extend(event_data)  # Append to existing
                    else:
                        all_coordinates[event_point] = event_data  # Create new entry
    
    return all_coordinates



def add_event_locations_mechanism_df(mechanism_df, matching_graph, all_fission_coordinates, all_fusion_coordinates, time_step=0.5):
    """
    Augment the mechanism DataFrame with spatial coordinates of fission and fusion events.

    For each object row, the function assigns spatial coordinates to the generation and/or 
    termination events if they correspond to a fission or fusion.

    Args:
        mechanism_df (pd.DataFrame): DataFrame containing object lifecycle and mechanism metadata.
        matching_graph (networkx.DiGraph): Temporal graph of object correspondences.
        all_fission_coordinates (dict): Dictionary mapping (Y, X) coordinate tuples to object ID sets for fission.
        all_fusion_coordinates (dict): Dictionary mapping (Y, X) coordinate tuples to object ID sets for fusion.
        time_step (float): Time between frames, used to resolve time indexing.

    Returns:
        pd.DataFrame: Updated mechanism_df with four new columns for event X and Y coordinates 
                      at generation and termination points.
    """
    
    # Initialize empty columns for both generation and termination fission locations
    mechanism_df['Generation Event Locations (X)'] = [[] for _ in range(len(mechanism_df))]
    mechanism_df['Generation Event Locations (Y)'] = [[] for _ in range(len(mechanism_df))]
    mechanism_df['Termination Event Locations (X)'] = [[] for _ in range(len(mechanism_df))]
    mechanism_df['Termination Event Locations (Y)'] = [[] for _ in range(len(mechanism_df))]

    # Loop through the rows of the dataframe
    for idx, row in tqdm(mechanism_df.iterrows(), total=len(mechanism_df), desc='Matching Event Locations'):
        object_id = row['Label']
        generation_mechanism = row['Generation Mechanism']
        termination_mechanism = row['Termination Mechanism']

        # Check if the row corresponds to a fission event
        if generation_mechanism == 'Fission' or termination_mechanism == 'Fission':
            # Get the frame and parent object if applicable
            generation_frame = int(2 * row['Time of Generation (Seconds)']) if generation_mechanism == 'Fission' else None
            termination_frame = int(2 * row['Time of Termination (Seconds)']) if termination_mechanism == 'Fission' else None

            # Handle pre-fission or post-fission cases
            # Case 1: Handle pre-fission object
            if generation_mechanism == 'Fission' and termination_mechanism != 'Fission':
                
                predecessor_node = list(matching_graph.predecessors((generation_frame, object_id)))
                parent_object_id = predecessor_node[0][1] if predecessor_node else None
                for fission_point, associated_object_ids in all_fission_coordinates.items():
                    if object_id in associated_object_ids and parent_object_id in associated_object_ids:
                        mechanism_df.at[idx, 'Generation Event Locations (Y)'].append(fission_point[0])
                        mechanism_df.at[idx, 'Generation Event Locations (X)'].append(fission_point[1])
            
            # Case 2: Handle post-fission object
            elif termination_mechanism == 'Fission' and generation_mechanism != 'Fission':
                
                for fission_point, associated_object_ids in all_fission_coordinates.items():
                    if object_id in associated_object_ids:
                        mechanism_df.at[idx, 'Termination Event Locations (Y)'].append(fission_point[0])
                        mechanism_df.at[idx, 'Termination Event Locations (X)'].append(fission_point[1])

            # Case 3: Handle the case when both generation and termination are fission
            if generation_mechanism == 'Fission' and termination_mechanism == 'Fission':
                predecessor_node = list(matching_graph.predecessors((generation_frame, object_id)))
                parent_object_id = predecessor_node[0][1] if predecessor_node else None
                for fission_point, associated_object_ids in all_fission_coordinates.items():
                    if object_id in associated_object_ids and parent_object_id in associated_object_ids:
                        mechanism_df.at[idx, 'Generation Event Locations (Y)'].append(fission_point[0])
                        mechanism_df.at[idx, 'Generation Event Locations (X)'].append(fission_point[1])
                    elif object_id in associated_object_ids and parent_object_id not in associated_object_ids:
                        mechanism_df.at[idx, 'Termination Event Locations (Y)'].append(fission_point[0])
                        mechanism_df.at[idx, 'Termination Event Locations (X)'].append(fission_point[1])
        
        # Check if the row corresponds to a fusion event
        if generation_mechanism == 'Fusion' or termination_mechanism == 'Fusion':
            # Get the frame and parent object if applicable
            generation_frame = int(2 * row['Time of Generation (Seconds)']) if generation_mechanism == 'Fusion' else None
            termination_frame = int(2 * row['Time of Termination (Seconds)']) if termination_mechanism == 'Fusion' else None
            # Handle pre-fusion or post-fusion cases
            # Case 1: Handle pre-fusion object
            if generation_mechanism == 'Fusion' and termination_mechanism != 'Fusion':
                for fusion_point, associated_object_ids in all_fusion_coordinates.items():
                    if object_id in associated_object_ids:
                        mechanism_df.at[idx, 'Generation Event Locations (Y)'].append(fusion_point[0])
                        mechanism_df.at[idx, 'Generation Event Locations (X)'].append(fusion_point[1])
            
            # Case 2: Handle post-fusion object
            elif termination_mechanism == 'Fusion' and generation_mechanism != 'Fusion':
                successor_node = list(matching_graph.successors((termination_frame, object_id)))
                child_object_id = successor_node[0][1] if successor_node else None
                for fusion_point, associated_object_ids in all_fusion_coordinates.items():
                    if object_id in associated_object_ids and child_object_id in associated_object_ids:
                        mechanism_df.at[idx, 'Termination Event Locations (Y)'].append(fusion_point[0])
                        mechanism_df.at[idx, 'Termination Event Locations (X)'].append(fusion_point[1])

            # Case 3: Handle the case when both generation and termination are fusion
            if generation_mechanism == 'Fusion' and termination_mechanism == 'Fusion':
                successor_node = list(matching_graph.successors((termination_frame, object_id)))
                child_object_id = successor_node[0][1] if successor_node else None
                for fusion_point, associated_object_ids in all_fusion_coordinates.items():
                    if object_id in associated_object_ids and child_object_id in associated_object_ids:
                        mechanism_df.at[idx, 'Termination Event Locations (Y)'].append(fusion_point[0])
                        mechanism_df.at[idx, 'Termination Event Locations (X)'].append(fusion_point[1])
                    elif object_id in associated_object_ids and child_object_id not in associated_object_ids:
                        mechanism_df.at[idx, 'Generation Event Locations (Y)'].append(fusion_point[0])
                        mechanism_df.at[idx, 'Generation Event Locations (X)'].append(fusion_point[1])
        
    return mechanism_df




#### ------------------------
#### Main to Process Data
#### ------------------------


def main(sample_number=1, time_step=0.5, pixel_res=0.108, root_data_directory="", output_base_directory="processed_outputs", in_memory=False):
    """
    Main processing function to load data, detect event mechanisms, localize event coordinates,
    and save output CSV summarizing mitochondrial dynamics.

    Args:
        sample_number (int): ID of the sample to process (used to construct paths).
        time_step (float): Time interval between frames in seconds.
        pixel_res (float): Pixel resolution in microns/pixel (used for scaling).
        root_data_directory (str): Base directory containing sample folders (default: current working directory).
        in_memory (bool): Whether to load data into memory (True) or memory-map extracted arrays (False).

    Returns:
        None. Saves the analysis output to CSV at the corresponding sample's processed data directory.
    """

    # Assign Sample Folders
    sample_folder, analysis_output_folder = setup_data_path(sample_number=sample_number, root_data_directory=root_data_directory, output_base_directory=output_base_directory)

    # Pixel Scaling in Microns Per Pixel
    micron_per_pixel = pixel_res/2 

    # Load Data
    instance_data, instance_skeleton, matching_graph = load_data(sample_folder=sample_folder, in_memory=in_memory)

    # Align objects with generation and termination event mechanisms
    mechanism_df = event_mechanism_detection(matching_graph=matching_graph, instance_data=instance_data, time_step=time_step)

    # Detect Fission and Fusion Event Coordinates
    all_fission_coordinates = calculate_event_coordinates_in_chunks('Fission', 'Termination', matching_graph, instance_data, instance_skeleton, mechanism_df, chunk_size=300)
    all_fusion_coordinates = calculate_event_coordinates_in_chunks('Fusion', 'Generation', matching_graph, instance_data, instance_skeleton, mechanism_df, chunk_size=300)

    # Add Fission and Fusion Event Coordinates to mechanism_df
    mechanism_df = add_event_locations_mechanism_df(mechanism_df, matching_graph, all_fission_coordinates, all_fusion_coordinates, time_step=0.5)

    # Save fission and fusion events in .csv spreadsheet
    output_file = f"{analysis_output_folder}/fission_fusion_mechanisms.csv"
    mechanism_df.to_csv(output_file, index=False)
    print("Saved mechanism_df to CSV.")

    del instance_data, instance_skeleton, matching_graph, mechanism_df, all_fission_coordinates, all_fusion_coordinates
    gc.collect()



if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Process a single sample to detect fission and fusion coordinates')
    parser.add_argument("--sample", type=int, default=1, help="Sample number to analyse")
    parser.add_argument("--time_step", type=float, default=0.5, help="Time step between frames in Seconds")
    parser.add_argument("--pixel_res",type=float, default=0.108, help="xy resolution in Microns per Pixel")
    parser.add_argument("--root_data_directory", type=str, help="Parent/root directory where all sample folders are located")
    parser.add_argument("--output_base_directory", type=str, default="processed_outputs", help="Base directory to store all processed outputs (default: processed_outputs)")
    parser.add_argument("--in_memory", action="store_true", help="Load all data into memory (useful for HPC data processing, True) or extract data and memory-map to reduce memory overhead (useful for local data processing with limitewd RAM, False)")

    args=parser.parse_args()

    # Process the sample
    main(sample_number=args.sample, time_step=args.time_step, pixel_res=args.pixel_res, root_data_directory=args.root_data_directory, output_base_directory=args.output_base_directory, in_memory=args.in_memory)

    # python 5_2_fission_fusion_event_detection.py --sample 33 --time_step 0.5 --pixel_res 0.108 --root_data_directory "input_data_crop/" --output_base_directory "processed_outputs/"