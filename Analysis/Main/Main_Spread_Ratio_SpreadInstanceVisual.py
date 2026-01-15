


#### -------
#### Imports
#### -------


# Standard library imports
from collections import deque
import os
from pathlib import Path
import zipfile

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable




#### ---------------------------
#### Compute BFS Ancestor Search
#### ---------------------------


def network_spread_memory_efficient(sample_folder: str, sample_no: int, graph_path: str, mask_path: str, seed_node: tuple[int, int], target_frame: int | None = None, show_matplotlib: bool = True) -> dict[tuple[int, int], int]:
    """
    Compute temporal connectivity between seeds and other objects on-demand using a memory-efficient BFS without storing full subgraph. 

    Parameters:
        sample_folder (str): Path to the sample folder containing segmentation outputs.
        sample_no (int): Sample ID identifier (for output naming).
        graph_path (str): Path to NetworkX directed graph pickle file with nodes as (frame, object_id).
        mask_path (str): Path to instance mask .npz file.
        seed_node (tuple): Seed node represented as (frame, object_id).
        target_frame (int, optional): Final frame index to propagate to.
        show_matplotlib (bool): Choose to generate matplotlib visualisation.

    Returns:
        connection_times (dict): Mapping from seed-frame nodes to the earliest frame at which they become connected to the seed.
    """

    # Load matching graph
    G = np.load(graph_path, allow_pickle=True)
    G = G.item() if isinstance(G, np.ndarray) else G
    
    if not Path(os.path.join(sample_folder, 'instance_mask_extracted_npz/arr_0.npy')).exists():
        instance_mask_npz_file = os.path.join(sample_folder, 'instance_masks.npz')
        with zipfile.ZipFile(instance_mask_npz_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(sample_folder, 'instance_mask_extracted_npz'))
    
    inst_arr = np.load(os.path.join(sample_folder, 'instance_mask_extracted_npz', 'arr_0.npy'), mmap_mode='r')

    seed_frame = seed_node[0]
    seed_mask = inst_arr[seed_frame]
    total_frame_count = len(inst_arr)
    
    print(f"Using memory-efficient approach for frames {seed_frame} to {target_frame}")
    
    # Find reachable nodes frame by frame to control memory
    connection_times = {}
    current_reachable = {seed_node}
    all_reachable = {seed_node}
    
    # Add seed to connection times
    connection_times[seed_node] = seed_frame
    if target_frame is None:
        target_frame = len(inst_arr) - 1
        
    for frame in range(seed_frame + 1, target_frame + 1):
        if frame % 50 == 0:
            print(f"Processing frame {frame}, currently tracking {len(current_reachable)} active nodes")
        
        next_reachable = set()
        
        # Find what current reachable nodes connect to in this frame
        for current_node in current_reachable:
            if current_node in G:
                for successor in G.successors(current_node):
                    if successor[0] == frame:
                        next_reachable.add(successor)
                        all_reachable.add(successor)
        
        # For each new reachable node, trace back to seed frame
        for new_node in next_reachable:
            # Trace back to find seed frame ancestors
            seed_ancestors = find_seed_ancestors_efficient(G, new_node, seed_frame)
            
            # Update connection times
            for ancestor in seed_ancestors:
                if ancestor not in connection_times or frame < connection_times[ancestor]:
                    connection_times[ancestor] = frame
        
        current_reachable = next_reachable
        
        # Early termination if nothing new is reachable
        if not current_reachable:
            print(f"No more reachable nodes found at frame {frame}, terminating early")
            break
    
    print(f"Found connections for {len(connection_times)} seed frame objects")
    
    if show_matplotlib:
        visualize_connections_fast(sample_no, seed_mask, seed_node, connection_times, target_frame, seed_frame, total_frame_count)
    
    return connection_times


def find_seed_ancestors_efficient(G, node: tuple[int, int], seed_frame: int) -> set[tuple[int, int]]:
    """
    Efficiently find all ancestor nodes of a given node that exist at a given seed frame using BFS with early termination.

    Parameters:
        G (nx.Graph): NetworkX directed graph with nodes as (frame, object_id).
        node (tuple): Node represented as (frame, object_id) from which to traverse backwards from.
        seed_frame (int): Frame index on which the seed node is located (ancestor frame).

    Returns:
        seed_ancestors (set): Set of ancestor nodes represented as (frame, object_id) traversed from a given node.
    """

    seed_ancestors = set()
    queue = deque([node])
    visited = {node}
    
    while queue:
        current = queue.popleft()
        
        # If we've reached seed frame, record and don't go further
        if current[0] == seed_frame:
            seed_ancestors.add(current)
            continue
        
        # If we've gone past seed frame, skip
        if current[0] < seed_frame:
            continue
        
        # Add predecessors
        if current in G:
            for pred in G.predecessors(current):
                if pred not in visited and pred[0] >= seed_frame:
                    visited.add(pred)
                    queue.append(pred)
    
    return seed_ancestors





#### ------------------------
#### Plotting
#### ------------------------


def visualize_connections_fast(sample_no: int, seed_mask: np.ndarray, seed_node: tuple[int, int], connection_times: dict[tuple[int, int], int], 
                               target_frame: int, seed_frame: int, total_frame_count: int | None = None, 
                               snapshot_percentiles: list[int] = [0,10,50,100], cmap_name: str = "Reds_r", 
                               out_path: str | None = None, scale: str | None = None) -> None:
    """
    Visualise temporal connectivity between a seed object and other objects.
    
    Parameters:
        sample_no (int): Sample ID identifier (for output naming).
        seed_mask (np.ndarray): 2D labeled instance mask for all objects on the seed frame.
        seed_node (tuple): Seed node represented as (frame, object_id).
        connection_times (dict): Mapping from other nodes/objects on the seed frame to the earliest frame at which they become connected to the seed.
        target_frame (int): Final frame index (for visualisation).
        seed_frame (int): Frame index of the seed node.
        total_frame_count (int, optional): Total number of frames in the dataset (used for colour normalisation).
        snapshot_percentiles (list): Percentiles of the total frame range at which to generate snapshot images of spread.
        cmap_name (str): Matplotlib colormap string to represent connection times.
        out_path (str, optional): Directory to save figures.
        scale (str, optional): Time unit scaling for plotting (None, 'seconds', or 'minutes').
    
    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12, 12))

    unique_objects = np.unique(seed_mask)
    unique_objects = unique_objects[unique_objects != 0]
    
    if connection_times:
        min_f = min(connection_times.values())
        max_f = max(connection_times.values())
    else:
        min_f, max_f = seed_frame, target_frame
    
    cmap = plt.cm.get_cmap(cmap_name)
    seed_color = [0.0, 1.0, 0.0, 1.0]  # black
    disconnected_color = [0.6, 0.6, 0.6, 1.0]  # grey
    
    print(f"Visualizing {len(unique_objects)} objects")
    print(f"Connection times: {len(connection_times)} objects connected")
    print(f"Frame range: {min_f} to {max_f}")
    
    # Create composite image
    h, w = seed_mask.shape
    
    connected_count = 0
    disconnected_count = 0

    if scale.lower() == "seconds":
        time_scale = 0.5
    elif scale.lower() == "minutes":
        time_scale = 0.5/60
    else:
        time_scale = 1
    
    if total_frame_count is None:
        norm = plt.Normalize(vmin=seed_frame*time_scale, vmax=target_frame*time_scale)
    else:
        norm = plt.Normalize(vmin=seed_frame*time_scale, vmax=total_frame_count*time_scale)
    
    snapshot_frames = [int(total_frame_count*pct/100) for pct in snapshot_percentiles]
    
    for frame in snapshot_frames:
        final_image = np.zeros((h, w, 4))
        for obj_id in unique_objects:
            mask = seed_mask == obj_id
            node = (seed_frame, obj_id)
            
            if node == seed_node:
                final_image[mask] = seed_color
            elif node in connection_times:
                frame_connected = connection_times[node]
                if frame_connected <= frame:
                    final_image[mask] = cmap(norm(frame_connected * time_scale))
                else:
                    final_image[mask] = disconnected_color
                connected_count += 1
            else:
                final_image[mask] = disconnected_color
                disconnected_count += 1
    
        print(f"Connected: {connected_count}, Disconnected: {disconnected_count}")
        
        im = ax.imshow(final_image, alpha=1.0, origin='lower')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(direction='out', labelsize=14, width=1, length=6)
        
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
        
        if out_path:
            fig.savefig(os.path.join(out_path, f"{sample_no}_spread_snapshpot_frame{frame}_object{seed_node[1]}.png"), dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
        else:
            plt.show()



if __name__=="__main__":
    # (Sample Name, Sample ID, Frame, Instance ID)
    perinuclear_spread_combinations = [
        ('WT', 10, 0, 1673), # Front Peri Big Network
        ('WT', 10, 0, 1887), # Mid Peri Smaller Mitochondria
        ('IRSp53_KO', 34, 0, 8116), # Big Network
        ('IRSp53_KO', 34, 0, 8105), # Small
        # ('9.6mW_cm2', 20, 0, 7219), # Central
        # ('50W_cm2', 25, 0, 4842), # Big Network
        # ('50W_cm2', 25, 0, 4884), # Smaller
        # ('200W_cm2', 1, 0, 6729), # Big Network
        # ('200W_cm2', 1, 0, 6880), # Small Mitochondria
        # ('400W_cm2', 21, 0, 9734), # Big Network
        # ('400W_cm2', 21, 0, 10083) # Small Mitochondria Close to Nucleus
    ]

    telenuclear_spread_combinations = [
        ('WT', 10, 0, 1), # Front
        ('IRSp53_KO', 34, 0, 52), # Front in a Group, Small
        ('IRSp53_KO', 34, 0, 4512), # Front Side in a Group closer to Perinucleus, Small
        # ('9.6mW_cm2', 20, 0, 6285), # Leading Edge Front
        # ('50mW_cm2', 25, 0, 1), # Leading Edge Front Isolated
        # ('200mW_cm2', 1, 0, 6946),
        # ('400mW_cm2', 21, 0, 10424)
    ]

    all_samples = perinuclear_spread_combinations + telenuclear_spread_combinations
    output_folder = "spread_ratio_network_spread_snapshots"
    os.makedirs(output_folder, exist_ok=True)

    for sample_name, sample_id_folder, start_frame, object_id in all_samples:
        print(f"Processing {sample_name}, seed {object_id}")
        
        sample_no = sample_id_folder
        parent_path = os.path.join(os.getcwd(), f'segmentation_data/{sample_no}')
        graph_path = os.path.join(parent_path, 'matching_nx_graph.pickle')
        mask_path = os.path.join(parent_path, 'instance_masks.npz')
        sample_folder = parent_path
        os.makedirs(os.path.join(output_folder, sample_name), exist_ok=True)
        os.makedirs(os.path.join(output_folder, sample_name, str(object_id)), exist_ok=True)
        seed_node = (start_frame, object_id)
        
        connection_times = network_spread_memory_efficient(sample_folder, sample_no, graph_path, mask_path, seed_node, target_frame=None, show_matplotlib=False)
        

        # Extract and load mitochondrial instance masks (memory-efficient loading npz, high disk usage)
        if not Path(os.path.join(sample_folder, 'instance_mask_extracted_npz/arr_0.npy')).exists():
            instance_mask_npz_file = os.path.join(sample_folder, 'instance_masks.npz')
            with zipfile.ZipFile(instance_mask_npz_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(sample_folder, 'instance_mask_extracted_npz'))
        
        inst_arr = np.load(os.path.join(sample_folder, 'instance_mask_extracted_npz', 'arr_0.npy'), mmap_mode='r')
        seed_mask = inst_arr[start_frame]
        total_frame_count = len(inst_arr)
        

        # Generate snapshots and save figures
        visualize_connections_fast(
            sample_no, seed_mask, seed_node, connection_times,
            seed_frame=start_frame,
            target_frame=len(inst_arr) - 1,
            total_frame_count=total_frame_count,
            snapshot_percentiles=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
            out_path=os.path.join(output_folder, sample_name, str(object_id)),
            cmap_name="plasma",
            scale="minutes"
        )