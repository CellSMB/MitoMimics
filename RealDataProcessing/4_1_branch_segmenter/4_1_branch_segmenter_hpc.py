


#### -------
#### Imports
#### -------


import os
import time
import gc
import numpy as np
import concurrent.futures
from tqdm import tqdm
import argparse
import multiprocessing
import pickle
import h5py
from collections import defaultdict
import networkx as nx



#### ------------------------
#### Branch Finding Algorithm
#### ------------------------


def find_direct_paths(graph):
    """
    Extracts direct paths between terminal nodes efficiently.
    
    Terminal nodes (Branch or Enpoint Nodes) are defined as: degree != 2
    Intermediate nodes are skipped: degree == 2
    
    Returns:
      list: A list of paths (each path is a list of nodes) connecting two terminal nodes.
            Each direct path is the shortest uninterrupted path along a chain between two 
            terminal nodes.
    """
    
    # Pre-compute degrees for all nodes for faster lookups
    node_degrees = dict(graph.degree())
    terminal_nodes = {node 
                      for node, degree in node_degrees.items() 
                      if degree != 2}
    
    # Track visited edges
    visited_edges = set()
    direct_paths = []
    direct_edges = []

    # Process all terminal nodes at once
    for terminal in terminal_nodes:

        # Process each neighbor
        for neighbor in graph.neighbors(terminal):
            # Canonicalize edge representation
            edge = tuple(sorted([terminal, neighbor]))
            
            if edge in visited_edges:
                continue
                
            visited_edges.add(edge)
            path = [terminal, neighbor]
            edges = [edge]

            # If neighbor is a terminal node, complete path
            if neighbor in terminal_nodes:
                direct_paths.append(path)
                direct_edges.append(edges)
                continue
            
            # Follow the path until we hit another terminal node
            current = neighbor
            prev = terminal
            
            while True:
                # Get all neighbors except the previous node
                next_nodes = [n 
                              for n in graph.neighbors(current) 
                              if n != prev]
                
                # If there are no next nodes or more than one, something's wrong
                if len(next_nodes) != 1:
                    break
                    
                next_node = next_nodes[0]
                edge = tuple(sorted([current, next_node]))
                visited_edges.add(edge)

                edges.append(edge)
                path.append(next_node)
                prev, current = current, next_node
                
                # If Terminal Node Reached Then Complete Path
                if current in terminal_nodes:
                    direct_paths.append(path)
                    direct_edges.append(edges)
                    break
    return direct_paths, direct_edges



def find_valid_pairs_single_frame(graph, instance_data, frame_num):
    """
    Find valid pairs of nodes within object subgraphs for a single frame.
    """

    valid_pairs = []
    
    # Pre-compute all node degrees once for the entire graph
    node_degrees = dict(graph.degree())
    
    # Get unique object IDs for the current frame
    current_frame_object_ids = np.unique(instance_data[frame_num])
    
    # Group nodes by object_id for the current frame
    nodes_by_object = defaultdict(list)
    
    # Filter nodes only for the current frame
    frame_nodes = [(n, d) 
                   for n, d in graph.nodes(data=True) 
                   if isinstance(n, tuple) and len(n) > 2 and n[2] == frame_num]
    
    for node, attributes in frame_nodes:
        object_id = attributes.get('label')
        if object_id in current_frame_object_ids:
            nodes_by_object[object_id].append(node)
    
    # Process each object's subgraph in the current frame
    for object_id, nodes_in_object in nodes_by_object.items():
        if not nodes_in_object:
            continue
            
        # Create subgraph only once per object_id
        subgraph = graph.subgraph(nodes_in_object).copy()
        subgraph_degrees = dict(subgraph.degree())
        
        # Get direct paths within the subgraph
        direct_paths, direct_edges = find_direct_paths(subgraph)
        
        # Process all paths for the current object_id
        for path_idx, path in enumerate(direct_paths):
            source_node = path[0]
            target_node = path[-1]
            
            # Use pre-computed degrees
            degree_source = subgraph_degrees.get(source_node, 0)
            degree_target = subgraph_degrees.get(target_node, 0)

            # Determine pair type using fewer conditional checks
            if degree_source == 1 and degree_target == 1:
                pair_type = "endpoint_to_endpoint"
            elif (degree_source > 2 and degree_target == 1) or (degree_source == 1 and degree_target > 2):
                pair_type = "branch_to_endpoint"
            elif degree_source > 2 and degree_target > 2 and source_node == target_node:
                pair_type = "branch_to_branch_loop"
            elif degree_source > 2 and degree_target > 2 and source_node != target_node:
                pair_type = "branch_to_branch_straight"
            else:
                continue
            
            # Store the valid path
            valid_pairs.append({
                'frame_num': frame_num,
                'object_id': object_id,
                'pair_type': pair_type,
                'source_node': source_node,
                'target_node': target_node,
                'path': path,
                'edges': direct_edges[path_idx]
            })
    
    return valid_pairs



#### --------------------------------------
#### Mapping Branch Paths to Instance Masks
#### --------------------------------------


def map_paths_to_instance_data(valid_pairs_by_frame, instance_data, hdf5_filename, batch_id=None):
    """
    Memory-efficient implementation to map graph paths to instance data using 
    distance-based assignment and store in a single HDF5 file.

    Parameters:
        valid_pairs_by_frame: Dictionary of valid pairs by frame
        instance_data: The original instance segmentation data
        hdf5_filename: Path to the single HDF5 file
        batch_id: Optional batch identifier for logging
        
    Returns:
        None (results are saved directly to HDF5)
    """ 

    # Open the HDF5 file in append mode
    with h5py.File(hdf5_filename, "a", libver="latest") as f:
        # Process each frame
        for frame_num, valid_pairs in valid_pairs_by_frame.items():
            frame_group = f.require_group(f"frames/{frame_num}")
            
            # Get frame mask - only access the specific frame we need
            frame_mask = instance_data[frame_num]
            
            # Group paths by object_id
            paths_by_object = defaultdict(list)
            for pair_idx, pair in enumerate(valid_pairs):
                object_id = pair["object_id"]
                paths_by_object[object_id].append((pair_idx, pair))
            
            # Process each object
            for object_id, paths in paths_by_object.items():
                object_group = frame_group.require_group(str(object_id))
                
                # Create binary mask for this object
                object_mask = (frame_mask == object_id).astype(np.uint8)
                
                # Get all coordinates of pixels in this object
                y_coords, x_coords = np.where(object_mask > 0)
                pixel_coords = np.column_stack((y_coords, x_coords))
                
                if len(pixel_coords) == 0:
                    continue  # Skip if no pixels for this object
                
                # Create a single assignment mask for the object
                assignment_mask = np.zeros_like(object_mask, dtype=np.int32)
                min_distances = np.ones(object_mask.shape) * np.inf
                
                # Process each path one at a time
                for path_idx, (pair_idx, pair) in enumerate(paths):
                    path_nodes = pair["path"]
                    
                    # Extract the pixel coordinates from the nodes
                    path_coords = np.array([(node[0], node[1]) 
                                            for node in path_nodes 
                                            if isinstance(node, tuple) and len(node) > 2])
                    
                    if len(path_coords) < 2:
                        continue  # Skip paths with too few nodes
                    
                    # Calculate distances for this path only
                    for y, x in path_coords:
                        y_diff = pixel_coords[:, 0] - y
                        x_diff = pixel_coords[:, 1] - x
                        distances = np.sqrt(y_diff**2 + x_diff**2)
                        
                        # Update minimum distances where this path is closer
                        mask_update_indices = np.where(distances < min_distances[pixel_coords[:, 0], pixel_coords[:, 1]])[0]
                        
                        if len(mask_update_indices) > 0:
                            update_coords = pixel_coords[mask_update_indices]
                            min_distances[update_coords[:, 0], update_coords[:, 1]] = distances[mask_update_indices]
                            assignment_mask[update_coords[:, 0], update_coords[:, 1]] = path_idx + 1
                
                # Store path masks in HDF5
                for path_idx, (pair_idx, pair) in enumerate(paths):
                    path_mask = (assignment_mask == path_idx + 1)
                    if np.any(path_mask):  # Only store if there are pixels assigned to this path
                        dataset_name = f"{pair_idx}"
                        # if dataset_name in object_group:
                        #     del object_group[dataset_name]  # Remove if it already exists
                            
                        object_group.create_dataset(dataset_name, data=path_mask, compression="gzip", dtype=np.uint8)

                        # Store Path Node Data in HDF5 Dataset MetaData (Access with .attrs)
                        path_nodes = [tuple(node) for node in pair["path"]]
                        object_group[dataset_name].attrs["path_nodes"] = path_nodes
                        edge_nodes = [(tuple(edge[0]), tuple(edge[1])) for edge in pair["edges"]]
                        object_group[dataset_name].attrs["path_edges"] = edge_nodes
                        object_group[dataset_name].attrs["pair_type"] = pair["pair_type"]
                        object_group[dataset_name].attrs["source_node"] = pair["source_node"]
                        object_group[dataset_name].attrs["target_node"] = pair["target_node"]
            
            gc.collect()
    
    return None



def parallel_process_frame_batch(batch_id, frame_batch, graph, instance_data, output_dir):
    """
    Process a batch of frames in parallel and store results in a temporary HDF5 file.

    Parameters:
        batch_id: ID of the batch
        frame_batch: List of frame numbers to process
        graph: The main graph
        instance_data: The instance data dictionary/array
        output_dir: Directory to store temporary HDF5 files
        
    Returns:
        Path to the temporary HDF5 file with batch results
    """

    # Create a temporary HDF5 file for this batch
    temp_hdf5_file = os.path.join(output_dir, f"temp_batch_{batch_id}.h5")

    print(f"[Batch {batch_id}] STARTED. Writing to {temp_hdf5_file}")

    # Store frame paths for storage object
    frame_results = {}
    try:
        with tqdm(total=len(frame_batch), desc=f"Processing batch {batch_id}") as pbar:
            for frame_num in frame_batch:
                print(f"[Batch {batch_id}] Processing frame {frame_num}")
                valid_pairs = find_valid_pairs_single_frame(graph, instance_data, frame_num)
                frame_results[frame_num] = valid_pairs
                
                # Save results to temporary HDF5
                map_paths_to_instance_data(
                    {frame_num: valid_pairs}, 
                    {frame_num: instance_data[frame_num]},
                    temp_hdf5_file,
                    batch_id
                )
                
                pbar.update(1)
        print(f"[Batch {batch_id}] COMPLETED. File written: {temp_hdf5_file}")

    except Exception as e:
        print(f"[Batch {batch_id}] ERROR: {str(e)}")
        raise e
            
    gc.collect()
    
    return temp_hdf5_file, frame_results



def merge_hdf5_files(temp_files, final_output_file):
    """
    Merge temporary HDF5 files into a single output file.
    
    Parameters:
        temp_files: List of paths to temporary HDF5 files
        final_output_file: Path to the final merged HDF5 file
    """

    # Create a new empty output file
    with h5py.File(final_output_file, "a") as f_out:
        for temp_file in tqdm(temp_files, desc="Merging temporary files"):
            if os.path.exists(temp_file):
                with h5py.File(temp_file, "r") as f_in:

                    # Create frames group in merged file
                    if 'frames' not in f_out:
                        f_out.create_group(f'frames')
                        
                    # Iterate through the groups under the 'frames' group
                    src_frames_group = f_in['frames']
                    for frame_num in src_frames_group.keys():
                        # Ensure the group exists in the output file
                        if frame_num not in f_out['frames']:
                            f_out.create_group(f'frames/{frame_num}')
                        
                        # Iterate over the object IDs (subgroups) in each frame
                        src_frame = f_in['frames'][frame_num]
                        for object_id in src_frame.keys():
                            # Ensure the object_id exists in the output file
                            if object_id not in f_out['frames'][frame_num]:
                                f_out.create_group(f'frames/{frame_num}/{object_id}')
                            
                            # Now, copy datasets from each object_id
                            src_instance_object = f_in['frames'][frame_num][object_id]
                            for dataset_name in src_instance_object.keys():
                                dataset_path = f'frames/{frame_num}/{object_id}/{dataset_name}'
                                
                                # Check if the dataset exists in the output file
                                tgt_instance_object = f_out['frames'][frame_num][object_id]
                                if dataset_name not in tgt_instance_object:
                                    src_instance_object.copy(dataset_name, tgt_instance_object)

                                    # Copy Path Metadata Attributes to the merged target HDF5 File
                                    src_branch_object = f_in['frames'][frame_num][object_id][dataset_name]
                                    tgt_branch_object = f_out['frames'][frame_num][object_id][dataset_name]
                                    for attr_key, attr_val in src_branch_object.attrs.items():
                                        tgt_branch_object.attrs[attr_key] = attr_val
                                else:
                                    print(f"Skipping {dataset_path} because it already exists.")

                # Remove the temporary file after merging
                os.remove(temp_file)



def extended_process_and_store_all_frames(graph, instance_data, output_path, output_dir=None, batch_size=5, max_workers=None):
    """
    Optimized version that processes frames in batches using ProcessPoolExecutor with
    separate HDF5 files for each batch to avoid locking issues.
    
    Parameters:
        graph: The main pos_matching_graph NetworkX graph
        instance_data: The instance data array/dictionary
        output_file: Path to the final HDF5 output file
        output_dir: Directory to store temporary files (if None, uses same directory as output_file)
        batch_size: Number of frames to process in a single batch
        max_workers: Maximum number of worker processes (defaults to CPU count // 2 if None)
        
    Returns:
        PathStorage object with the processed graph data
    """

    print("Processing all frames with optimized parallelization...")

    start_time = time.time()

    # Create output directory if not provided
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    segmentation_filename = os.path.join(output_path, 'branch_instance_masks.h5')
    print(f"Final Segmentation File Location: {segmentation_filename}")
    
    # Determine total frames and create batches
    total_frames = len(instance_data)
    frame_batches = [list(range(i, min(i + batch_size, total_frames))) 
                     for i in range(0, total_frames, batch_size)]
    
    if max_workers is None:
        max_workers = os.cpu_count() // 2
    
    print(f"Using {max_workers} worker processes with batch size {batch_size}")
    
    temp_files = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Submit all batch jobs
        for batch_idx, batch in enumerate(frame_batches):
            future = executor.submit(
                parallel_process_frame_batch, 
                batch_idx, 
                batch, 
                graph, 
                {frame: instance_data[frame] for frame in batch}, 
                output_path
            )
            
            futures.append(future)
        
        # Process results as they complete
        with tqdm(total=len(futures), desc="Processing frame batches") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    temp_file, frame_results = future.result()
                    temp_files.append(temp_file)
    
                    pbar.update(1)
    
                    # Clear frame_results to save memory
                    del frame_results 
                    
                    gc.collect()
                        
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
    
    # Merge all temporary HDF5 files into the final output file
    merge_hdf5_files(temp_files, segmentation_filename)

    # Final garbage collection
    gc.collect()

    total_time = time.time() - start_time

    print(f"\n Total Processing Time: {total_time} seconds")




def process_single_sample(sample_num, data_dir, output_root_dir, batch_size=10, max_workers=None):
    
    """
    Process a single sample
    """
    
    print(f"\n ==== Processing Sample {sample_num} ====")
    
    # Paths for this sample
    sample_dir = os.path.join(data_dir, str(sample_num))
    raw_path = os.path.join(sample_dir, "raw.npz")
    instance_path = os.path.join(sample_dir, "instance_masks.npz")
    pos_matching_graph_path = os.path.join(sample_dir, "pos_matching_graph.pickle")
    
    # Check if files exist
    if not (os.path.exists(raw_path) and os.path.exists(instance_path)):
        print(f"\n Error: Required files not found for sample {sample_num}.")
        return
    
    # Load data
    print(f"\n Loading data for sample {sample_num}...")

    raw_data = np.load(raw_path, mmap_mode='r')['arr_0']
    instance_data = np.load(instance_path, mmap_mode='r')['arr_0']
    with open(pos_matching_graph_path, 'rb') as f:
        # Load per Structure Sub-Node Graphs (i.e. Skeletons)
        pos_matching_graph_data = pickle.load(f)

    # Create output directory for this sample
    sample_output_dir = os.path.join(output_root_dir, str(sample_num))
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create output directory for this parameter set
    param_dir_name = f"branch_segmentations"
    param_output_dir = os.path.join(sample_output_dir, param_dir_name)
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Output path for branch segmentations
    output_path = param_output_dir#os.path.join(param_output_dir, f".h5")

    # Run Branch Segmentation in parallel
    extended_process_and_store_all_frames(
        graph=pos_matching_graph_data,
        instance_data=instance_data,
        output_path=output_path,
        batch_size=batch_size,  # Smaller batch size for less memory usage
        max_workers=max_workers  # Limit workers to control memory
    )
    
    print(f"\n Completed processing for sample {sample_num}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single sample')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing sample folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for output results')
    parser.add_argument('--sample', type=int, required=True, help='Sample number to process')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for job processing')
    parser.add_argument('--max_workers', type=int, default=5, help='Maximum number of worker processes')
    
    args = parser.parse_args()
    
    # Process the sample
    process_single_sample(
        sample_num=args.sample,
        data_dir=args.data_dir,
        output_root_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
