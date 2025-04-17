# Standard library
import argparse
import gc
import itertools
import math
import os
import pickle
import sys
import timeit
from collections import defaultdict, deque
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint

# Third-party libraries
import matplotlib.pyplot as plt
import nibabel as nib
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed

sys.setrecursionlimit(30000)

###### Functions


# SORTS NODE GRAPH
def assign_positions(G):

    """
    Assigns 2D layout positions (which could be used for vis) to nodes in the directed temporal graph that represents inter frame matchings `G`.
    Used to keep consistent labels across frames if a mitochondrion instance DOESNT fision or fusion
    Each node is a (t, x), where `t` indicates the frame.
    The function organizes nodes such that time is on the x-axis and rows (y-axis) 
    are assigned to avoid overlaps.
    Nodes are grouped by time and placed in rows. Paths that split or merge are 
    assigned new rows to avoid visual ambiguity.
    """
    
    pos = {}  # Position dictionary: node -> (x, y)
    assigned_rows = {}  # Mapping from nodes to rows
    current_row = 0     # Row counter

    nodes_by_time = {}
    for node in G.nodes():
        t, x = node
        nodes_by_time.setdefault(t, []).append(node)

    # Sort time steps for consistent processing
    time_steps = sorted(nodes_by_time.keys())

    def assign_row(node, row, t):
        # Assign the node to the given row
        assigned_rows[node] = row
        pos[node] = (t, row)

        # Get successors of the node
        successors = list(G.successors(node))

        for successor in successors:
            if successor in assigned_rows:
                continue  # Skip if already assigned

            preds = list(G.predecessors(successor))

            if len(preds) == 1 and len(successors) == 1:
                # 1-to-1 matching, continue on the same row
                assign_row(successor, row, t + 1)
            else:
                # Split or merge, assign new row
                nonlocal current_row
                current_row += 1
                assign_row(successor, current_row, t + 1)

    for t in time_steps:
        for node in nodes_by_time[t]:
            if node not in assigned_rows:
                # Assign a new row to the node
                assign_row(node, current_row, t)
                current_row += 1

    return pos


def label_unique_across_frames(array_3d):

    """
    Labels connected components in each timestep of an array (time, y, x), ensuring unique labels for each instance at a given frame,
    and that instances labels are unique also across all time slices. 
    """

    t = array_3d.shape[0]
    # We'll store our labeled slices in a new array of the same shape.
    labeled_3d = np.zeros_like(array_3d, dtype=int)
    
    # 'offset' will keep track of how many labels we've used so far
    offset = 0
    
    for i in range(t):
        # Label the slice (assuming > 0 is foreground; adjust as needed).
        # If your data is already boolean, you can pass array_3d[i] directly.
        labeled_slice = label(array_3d[i] > 0, background=0)
        
        # Any non-zero label in 'labeled_slice' should be incremented by 'offset'.
        # This ensures uniqueness across frames.
        if labeled_slice.max() > 0:  # Only shift if there are labeled regions
            labeled_slice[labeled_slice > 0] += offset
            
            # Update our global offset to the new maximum
            offset = labeled_slice.max()
        
        # Store the slice
        labeled_3d[i] = labeled_slice
    
    # After processing all time steps, the overall max label is in `offset`.
    max_label = offset
    
    return max_label, labeled_3d




def create_overlap_digraph_pandas(single_frame_skel_stack):
    """
    Create a directed graph (DiGraph) where each node is (t, label) for
    non-zero labels in a stack of frames. Edges indicate overlap between labels
    in consecutive frames. 

    This is the baslineof fission and fusion matching, subsequent algo post processing steps
    remove what we consider eroneous fis/fus after this graph is created.

    Parameters
    ----------
    single_frame_skel_stack : np.ndarray
        3D array of shape (T, X, Y), where each 2D slice (t, :, :) contains
        integer labels. Zero is considered background.

    Returns
    -------
    G : networkx.DiGraph
        Directed graph where each node is (t, label), and edges connect labels
        that overlap from frame t to t+1.
    """
    G = nx.DiGraph()

    # 1. Add nodes: (t, label) for each frame and each non-zero label
    n_frames = single_frame_skel_stack.shape[0]
    for t in range(n_frames):
        unique_id_list = list(set(single_frame_skel_stack[t, :, :].flatten()))
        # Remove background label if it exists
        if 0 in unique_id_list:
            unique_id_list.remove(0)

        for label_val in unique_id_list:
            G.add_node((t, label_val))

    # 2. Add edges: compare consecutive frames
    for t in range(n_frames - 1):
        next_t = t + 1

        # Flatten the current and next frame
        low_frame_arr = single_frame_skel_stack[t, :, :]
        high_frame_arr = single_frame_skel_stack[next_t, :, :]

        # Build a DataFrame (one row per pixel) showing label in t and t+1
        df = pd.DataFrame({
            'A': low_frame_arr.flatten(),
            'B': high_frame_arr.flatten()
        })

        # Drop duplicates and filter out rows with zeros
        df_unique = df.drop_duplicates()
        df_filtered = df_unique[(df_unique['A'] != 0) & (df_unique['B'] != 0)]

        # Convert pairs to numpy array for iteration
        unique_pairs = df_filtered.to_numpy()

        # Each unique (A, B) pair indicates an overlap => edge
        for label_low, label_high in unique_pairs:
            G.add_edge((t, label_low), (next_t, label_high))

    return G


# The following function, and the functions that it calls after are used to "efficiently" find
# paths between two nodes that have no edge or node overlaps
# This is used to correct code un then remerge of masks, that are unlikely to be a fission->fusion event
# but instead a drop in signal or related issue

def find_non_overlapping_path_groups(G, max_path_length=5, num_processes=None):

    """
    Finds and groups all sets of non-overlapping paths between diverging and converging 
    nodes in a directed graph `G` using flow-based decomposition. Delegates per-source 
    processing to multiprocessing workers and applies post-processing to filter and 
    merge maximal non-overlapping path sets.
    
    Args:
        G (networkx.DiGraph): A directed graph where each node is a (time, id) tuple.
        max_path_length (int): Maximum number of time steps allowed in a path.
        num_processes (int or None): Number of parallel processes to use.
    
    Returns:
        List[Dict]: A list of path group dictionaries, each with 'source', 'meeting',
        and 'paths' keys. Each path is a list of node tuples.
    """

    potential_sources = [n for n in G.nodes() if G.out_degree(n) > 1]
    potential_meetings = [n for n in G.nodes() if G.in_degree(n) > 1]
    potential_sources.sort()
    potential_meetings.sort(reverse=True)

    total_sources = len(potential_sources)
    print(f"Total sources to check: {total_sources}")

    # Precompute limited-depth descendants for each source node
    descendant_dict = {
        source: set(nx.single_source_shortest_path_length(G, source, cutoff=max_path_length).keys())
        for source in potential_sources
    }

    with Pool(processes=num_processes) as pool:
        args = [
            (source, G, max_path_length, descendant_dict[source], potential_meetings, i + 1, total_sources)
            for i, source in enumerate(potential_sources)
        ]
        results = pool.starmap(process_source, args)

    all_path_groups = []
    for result in results:
        all_path_groups.extend(result)

    maximal_path_groups = filter_non_maximal_path_groups(all_path_groups)
    merged_path_groups = merge_groups_by_source_meeting(maximal_path_groups)

    return merged_path_groups


def process_source(source, G, max_path_length, valid_descendants, potential_meetings, current_source_index, total_sources):


    """
    Processes a single source node by checking for valid meeting nodes, constructing
    flow networks, and extracting non-overlapping path groups if valid flow exists.
    
    Args:
        source (Tuple): Source node tuple (time, id).
        G (networkx.DiGraph): The graph.
        max_path_length (int): Maximum allowed length of any path.
        valid_descendants (Set): Descendant nodes reachable from the source.
        potential_meetings (List): List of candidate meeting nodes.
        current_source_index (int): Index of the current source for logging.
        total_sources (int): Total number of source nodes.
    
    Returns:
        List[Dict]: A list of dictionaries, each with 'source', 'meeting', and 'paths'.
    """

    path_groups = []
    source_time = source[0]

    # Only keep meetings within max_path_length and within descendant set
    valid_meetings = [
        m for m in potential_meetings 
        if m[0] > source_time and 
           (m[0] - source_time) <= max_path_length and 
           m in valid_descendants
    ]

    for meeting in valid_meetings:
        F = build_flow_graph(G, source, meeting, max_path_length)
        if F is None:
            continue
        source_out = (source, 'out')
        meeting_in = (meeting, 'in')
        if not F.has_node(source_out) or not F.has_node(meeting_in):
            continue
        try:
            flow_value = nx.maximum_flow_value(F, source_out, meeting_in)
        except nx.NetworkXUnbounded:
            flow_value = 0
        if flow_value >= 2:
            flow_dict = nx.maximum_flow(F, source_out, meeting_in)[1]
            paths = decompose_flow(F, flow_dict, source_out, meeting_in)
            original_paths = convert_flow_paths_to_original(paths)
            if len(original_paths) >= 2:
                path_groups.append({'source': source, 'meeting': meeting, 'paths': original_paths})

    return path_groups


def build_flow_graph(G, source, meeting, max_path_length):

    """
    Constructs a flow network from source to meeting node over a time-bounded window.
    
    Args:
        G (networkx.DiGraph): Original graph.
        source (Tuple): Source node.
        meeting (Tuple): Meeting node.
        max_path_length (int): Max allowed steps from source to meeting.
    
    Returns:
        networkx.DiGraph or None: The constructed flow network, or None if constraints are violated.
    """
    
    source_time = source[0]
    meeting_time = meeting[0]
    if (meeting_time - source_time) > max_path_length or source_time >= meeting_time:
        return None
    
    F = nx.DiGraph()
    nodes_in_range = [n for n in G.nodes() if source_time <= n[0] <= meeting_time]
    
    for node in nodes_in_range:
        in_node = (node, 'in')
        out_node = (node, 'out')
        cap = 1000 if node == source or node == meeting else 1
        F.add_edge(in_node, out_node, capacity=cap)
        
        for succ in G.successors(node):
            if succ[0] == node[0] + 1 and succ in nodes_in_range:
                F.add_edge(out_node, (succ, 'in'), capacity=1)
    
    return F

def decompose_flow(F, flow_dict, source_out, meeting_in):

    """
    Decomposes a flow dictionary into individual simple paths between source and meeting.
    
    Args:
        F (networkx.DiGraph): Flow graph.
        flow_dict (dict): Resulting flow values from max flow.
        source_out (Tuple): Source 'out' node.
        meeting_in (Tuple): Meeting 'in' node.
    
    Returns:
        List[List[Tuple]]: List of flow paths as sequences of nodes in the flow graph.
    """
    
    paths = []
    residual = nx.DiGraph()
    for u in flow_dict:
        for v in flow_dict[u]:
            if flow_dict[u][v] > 0:
                residual.add_edge(u, v, capacity=flow_dict[u][v])
    
    while True:
        try:
            path = nx.shortest_path(residual, source_out, meeting_in)
            paths.append(path)
            for u, v in zip(path[:-1], path[1:]):
                if residual[u][v]['capacity'] > 1:
                    residual[u][v]['capacity'] -= 1
                else:
                    residual.remove_edge(u, v)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
    return paths

def convert_flow_paths_to_original(flow_paths):

    """
    Converts paths from flow node notation (with 'in'/'out') back to original node tuples.
    
    Args:
        flow_paths (List[List[Tuple]]): List of flow paths using augmented nodes.
    
    Returns:
        List[List[Tuple]]: List of original graph node paths.
    """
    
    original_paths = []
    for path in flow_paths:
        original = []
        for node in path:
            if node[1] == 'out' and node[0] not in original:
                original.append(node[0])
            elif node[1] == 'in' and node[0] not in original:
                original.append(node[0])
        original_paths.append(original)
    return original_paths

def filter_non_maximal_path_groups(groups):

    """
    Removes path groups that are subsets of other path groups with the same source and meeting.
    
    Args:
        groups (List[Dict]): List of path group dictionaries.
    
    Returns:
        List[Dict]: Filtered list containing only maximal (non-subset) path groups.
    """
        
    groups_by_key = defaultdict(list)
    for group in groups:
        key = (group['source'], group['meeting'])
        groups_by_key[key].append(set(map(tuple, group['paths'])))
    
    maximal = []
    for key, group_sets in groups_by_key.items():
        sorted_groups = sorted(group_sets, key=lambda x: -len(x))
        kept = []
        for g in sorted_groups:
            if not any(g.issubset(h) for h in kept):
                kept.append(g)
        maximal.extend([{'source': key[0], 'meeting': key[1], 'paths': [list(p) for p in g]} for g in kept])
    return maximal

def merge_groups_by_source_meeting(groups):

    """
    Merges path groups with the same source and meeting into unified path sets.
    
    Args:
        groups (List[Dict]): List of non-overlapping path group dictionaries.
    
    Returns:
        List[Dict]: List of merged path groups, grouped by source and meeting node.
    """
    
    merged = defaultdict(set)
    for group in groups:
        key = (group['source'], group['meeting'])
        merged[key].update(tuple(path) for path in group['paths'])
    return [{'source': k[0], 'meeting': k[1], 'paths': [list(p) for p in v]} for k, v in merged.items()]




def is_path_subset(path1, path2):

    """
    Checks whether `path1` is a contiguous subpath of `path2`.
    
    Args:
        path1 (List): A list of nodes representing the candidate subpath.
        path2 (List): A list of nodes representing the full path.
    
    Returns:
        bool: True if `path1` is a contiguous subset of `path2`, False otherwise.
    """

    len1, len2 = len(path1), len(path2)
    if len1 > len2:
        return False
    for i in range(len2 - len1 + 1):
        if path2[i:i+len1] == path1:
            return True
    return False

def filter_path_subset_groups(path_groups):

    """
    Removes any path group from a list if all of its paths are subsets of the paths in another group.
    
    Args:
        path_groups (List[Dict]): A list of path group dictionaries, each containing 
            a 'paths' key with a list of node sequences.
    
    Returns:
        List[Dict]: A filtered list of path groups, excluding those where all paths 
        are subsets of another group's paths.
    """

    groups_to_remove = set()

    for i, group1 in enumerate(path_groups):
        if i in groups_to_remove:
            continue

        for j, group2 in enumerate(path_groups):
            if j == i or j in groups_to_remove:
                continue

            # Compare each path in group1 to each path in group2
            # If any path1 is a subset of path2, remove group2 (the superset group)
            has_subset_path = False
            for path1 in group1['paths']:
                for path2 in group2['paths']:
                    if is_path_subset(path1, path2):
                        groups_to_remove.add(j)
                        has_subset_path = True
                        break
                if has_subset_path:
                    break

    return [group for i, group in enumerate(path_groups) if i not in groups_to_remove]

def min_mask_distance(mask_A, mask_B):
    """
    Computes the minimum Euclidean distance between the true regions of two binary masks.
    
    Parameters:
    mask_A (np.array): A binary numpy array.
    mask_B (np.array): A binary numpy array.
    
    Returns:
    float: The minimum distance between the true regions in the two masks.
    """
    # Compute the distance transform on the inverse of mask_A
    dist_transform = distance_transform_edt(1 - mask_A)
    # Extract distances at positions where mask_B is True and return the minimum
    return dist_transform[mask_B.astype(bool)].min()


def pairs_from_frame_pair(t, low, high):
    
    """
    Generates triplets (t, A, B) for non-zero label pairs from two consecutive frames.
    
    Args:
        t (int): The current time/frame index.
        low (np.ndarray): 2D array of labels from frame t.
        high (np.ndarray): 2D array of labels from frame t+1.
    
    Returns:
        np.ndarray: An array of shape (N, 3), where each row is a triplet (t, label_in_low, label_in_high)
                    for each non-zero pair.
    """
    
    valid = (low != 0) & (high != 0)
    A, B = low[valid], high[valid]
    return np.column_stack((np.full(len(A), t), A, B))



def get_overlap_graph(single_frame_skel_stack):
    
    """
    Constructs a directed graph from a stack of labeled skeleton masks across time.
    
    Each node is a tuple (t, label), and a directed edge is added from label `a` at time `t`
    to label `b` at time `t+1` if they overlap spatially.
    
    Args:
        single_frame_skel_stack (np.ndarray): A 3D array of shape (T, H, W) where each frame
            contains integer-labeled skeleton components.
    
    Returns:
        networkx.DiGraph: A directed graph where nodes represent labeled components at each
        time step, and edges indicate temporal overlap between frames.
    """

    # Parallelize over frame pairs
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(pairs_from_frame_pair)(t, single_frame_skel_stack[t], single_frame_skel_stack[t+1])
        for t in range(single_frame_skel_stack.shape[0] - 1)
    )
    
    # Combine results and remove duplicates
    unique_pairs_with_timesteps = np.unique(np.vstack(results), axis=0)



    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Efficiently build nodes and edges
    # Node format: (t, id)
    t_vals, id_l_vals, id_h_vals = unique_pairs_with_timesteps.T
    
    # Add edges in bulk for efficiency
    edges = [((t, id_l), (t+1, id_h)) for t, id_l, id_h in zip(t_vals, id_l_vals, id_h_vals)]
    G.add_edges_from(edges)

    return(G)


def shortest_distance_kdtree(mask1, mask2):
    coords1 = np.column_stack(np.nonzero(mask1))
    coords2 = np.column_stack(np.nonzero(mask2))
    tree = cKDTree(coords1)
    dist, _ = tree.query(coords2, k=1)
    return dist.min()
    

def filter_dicts_by_timesteps(groups):
    """
    Filters path groups so that:
    - Groups are prioritized by minimal path length.
    - No group can overlap (share a node) with any shorter-length group's nodes.
    - Within each length group, includes as many as possible without overlap.
    
    Returns:
        List of accepted groups.
    """
    from collections import defaultdict

    # Step 1: Group all groups by their minimal path length
    length_to_groups = defaultdict(list)
    group_node_sets = []

    for group in groups:
        min_path_len = min(len(p) for p in group['paths'])
        group_nodes = set(node for path in group['paths'] for node in path)
        length_to_groups[min_path_len].append((group, group_nodes))
        group_node_sets.append((group, group_nodes, min_path_len))

    # Step 2: Process by increasing path length
    all_blocked_nodes = set()
    selected_groups = []

    for path_len in sorted(length_to_groups.keys()):
        used_nodes_in_this_round = set()
        for group, group_nodes in length_to_groups[path_len]:
            if group_nodes.isdisjoint(all_blocked_nodes):
                selected_groups.append(group)
                used_nodes_in_this_round.update(group_nodes)
        all_blocked_nodes.update(used_nodes_in_this_round)

    return selected_groups



if __name__ == "__main__":

    # Timing for checking 
    full_start_time = timeit.default_timer()

    # input arguments
    parser = argparse.ArgumentParser(description="Process an integer input.")
    parser.add_argument("in_num", type=int, help="An integer input for the program.")
    parser.add_argument("in_path_model", type=str, help="Path to folder that contains fmask and skel model outputs")
    parser.add_argument("in_path_raw", type=str, help="Path to folder that contains the raws")
    parser.add_argument("out_path", type=str, help="Path to folder to store processed outputs")

    args = parser.parse_args()

    in_num = args.in_num
    in_path_model = args.in_path_model
    in_path_raw = args.in_path_raw
    out_path = args.out_path
    
    print(f"Input number is: {in_num}")

    
    # hardcoded parameters, potentially modify these if your time interval is different
    cycle_max_length = 19
    fade_max_length = 16
    blip_max_length = 18


    # path processing
    raw_path = os.path.join(in_path_raw,f'MIMIC_Ts_{str(in_num).zfill(4)}_0000.nii.gz')
    fmask_path = os.path.join(in_path_model,f'fmask/MIMIC_Ts_{str(in_num).zfill(4)}.nii.gz')
    skel_path = os.path.join(in_path_model,f'skel/MIMIC_Ts_{str(in_num).zfill(4)}.nii.gz')
    
    print('raw_path:', raw_path)
    print('fmask_path:', fmask_path)
    print('skel_path:', skel_path)

    
    # output folders
    out_dir = os.path.join(out_path, f'{str(in_num)}')
    print(out_dir)
    os.mkdir(out_dir)
    

    # load in raw arrays
    
    skel_arr = nib.load(skel_path).get_fdata().astype(np.uint32)
    skel_arr = np.transpose(skel_arr, (2,0,1))
    
    fmask_arr = nib.load(fmask_path).get_fdata().astype(np.uint32)
    fmask_arr = np.transpose(fmask_arr, (2,0,1))
    
    raw_arr = nib.load(raw_path).get_fdata().astype(np.uint16)
    raw_arr = np.transpose(raw_arr, (2,0,1))


    # alias
    skel_pred_arr = skel_arr
    fmask_pred_arr = fmask_arr
    curr_raw_arr = raw_arr#


    # Prints for checking
    print('curr_raw_arr shape =', str(curr_raw_arr.shape))
    min_t_value = 0
    print('min_t_value =', min_t_value)
    max_t_value = curr_raw_arr.shape[0]-1
    print('max_t_value =', max_t_value)
    
    
    # Create a stack of instance masks for each frame. Each instances is given a unique id across all frames
    # Max label is the number of unique instances
    max_label, single_frame_skel_stack = label_unique_across_frames(skel_pred_arr)

    
    # new start spot for instances that are added during post processing
    new_id_counter = max_label+1


    # swap type
    single_frame_skel_stack = single_frame_skel_stack.astype(np.uint32)


    # While loop for fixing short unmerge-remerge events 
    patience_counter = 0
    last_to_fix = 100000000000
    keep_doing = True
    
    while keep_doing == True:
    
        print('--------------------------------------------------------------------------------------------')

        # get the overlap graph for the current single_frame_skel_stack
        # digraph matching instances between frames
        # See function def for more details
        G = get_overlap_graph(single_frame_skel_stack)
        

        # finds distjoint paths between nodes
        # see function for mroe details
        path_groups = find_non_overlapping_path_groups(G, max_path_length=cycle_max_length)
    

        # filters path groups found by previous function
        filtered_path_groups = filter_path_subset_groups(path_groups)


        # further filters the path groups to make sure that they arnt to far from each other
        # also splits groups that have more than 2 paths between the same two nodes for further processing
        loops_to_fix = list()
    
        for each in filtered_path_groups:
            ts_low = each['source'][0]
            ts_compare = ts_low + 1
        
            comb_list = sum(each['paths'],[])
            
            curr_uniques = [x[1] for x in comb_list if x[0] == ts_compare]
            #print(curr_uniques)
        
            if len(curr_uniques) == 2:
                mask_1 = single_frame_skel_stack[ts_compare, :, :] == curr_uniques[0]
                mask_2 = single_frame_skel_stack[ts_compare, :, :] == curr_uniques[1]
                distance = shortest_distance_kdtree(mask_1, mask_2)
                
                if distance > 30:
                    continue
                
                loops_to_fix.append(each)
            
            elif len(curr_uniques) > 2:
                # 1. Create a graph
                t_graph = nx.Graph()
                
                # 2. Add all current_uniques as nodes
                t_graph.add_nodes_from(curr_uniques)
                
                # 3. For every pair of 'curr_uniques', check distance and potentially add an edge
                for c1, c2 in itertools.combinations(curr_uniques, 2):
                    mask_1 = single_frame_skel_stack[ts_compare, :, :] == c1
                    mask_2 = single_frame_skel_stack[ts_compare, :, :] == c2
                    
                    distance = shortest_distance_kdtree(mask_1, mask_2)
                    
                    # If they are close enough, add an edge
                    if distance <= 30:
                        t_graph.add_edge(c1, c2)
                
                # 4. Collect all subgraphs with size >= 2
                connected_components = nx.connected_components(t_graph)
                subgraphs = []
                for component in connected_components:
                    if len(component) >= 2:
                        sg = t_graph.subgraph(component).copy()
                        subgraphs.append(sg)
                
                # 
                for sg in subgraphs:
                    # Get the set of skeleton IDs in this subgraph
                    subgraph_ids = set(sg.nodes)
                    
                    # Filter original paths so that each path’s ID at ts_compare is in this subgraph
                    splitted_paths = []
                    for path in each['paths']:
                        # A path is a list of tuples (timestep, skeleton_id)
                        # We want to see if there's a tuple (ts_compare, X) where X is in subgraph_ids
                        if any((ts_compare, node_id) in path for node_id in subgraph_ids):
                            splitted_paths.append(path)
                    
                    # Construct a new dictionary in the same format but only for this subgraph’s paths
                    splitted_dict = {
                        'source': each['source'],
                        'meeting': each['meeting'],
                        'paths': splitted_paths
                    }
                    
                    # Now append this sub-split entry instead of the full 'each'
                    loops_to_fix.append(splitted_dict)


        # end condition (no more double paths to fix)
        if len(loops_to_fix) == 0:
            break

        # patience break if it gets stuck on edge case unsolvable loops (hasnt hapened, but just in case)
        if len(loops_to_fix) >= last_to_fix:
            patience_counter += 1

            last_to_fix = len(loops_to_fix)
        else:
            last_to_fix = len(loops_to_fix)
            patience_counter =0

        if patience_counter > 10:
            print('failing to end')
            break


        # sorts the double paths in order of shortest to longer
        loops_to_fix.sort(key=lambda x: len(x['paths'][0]))
        
        # get the shortest in the sorted list
        loop =  sum(loops_to_fix[0]['paths'], [])


        # add the to a list for processing
        valid_list = list()
        min_bar = len(loop)
        for each in loops_to_fix:
            #if len(sum(each['paths'], []))<= min_bar+2:
            valid_list.append(each)

        # return a list of double paths to process in the iteration
        # smaller loops are prioritized. overlapping loops (those that share a node) arnt done in the same iteration
        valid_list_non_overlap = filter_dicts_by_timesteps(valid_list)
    


        # Change the single_frame_skel_stack to merge the IDs of the double paths
        # As this same arr is referenced in future iterations and post processing, the changes are permenant
        counter = 0
        
        for ploop in valid_list_non_overlap:
    
            loop =  sum(ploop['paths'], [])
    
        
            ts_range = [x[0] for x in loop]
            
            ts_low = min(ts_range)
            ts_high = max(ts_range)
        

            for ts in range(ts_low + 1, ts_high):
            
                new_id_counter+=1
            
                curr_new_val = new_id_counter
                
                #print(ts)
                vals_to_merge = [x for x in loop if x[0] == ts]
                ids_to_merge = [x[1] for x in vals_to_merge]        
            
                for idx in ids_to_merge:
                    single_frame_skel_stack[ts,...][single_frame_skel_stack[ts,...] == idx] = np.uint32(curr_new_val)




    # Get the new Graph from the corrected skel stack
    G = get_overlap_graph(single_frame_skel_stack)



    
    # Following code block fixes a similar issue to the unmerge-remerge.
    # Occasionally single loss will cause a bit of a mitos skeleton mask to disjoint from the majority, then disapear
    # this is corrected by adding them back to the main mask


    """
    Performs iterative pruning of short-lived, disjoint components in a labeled skeleton stack
    by building a temporal overlap graph and cleaning up leaf-to-hub chains of limited length.
    
    Args:
        single_frame_skel_stack (np.ndarray): A 3D array of shape (T, H, W) with labeled
            components per time frame.
        fade_max_length (int): Maximum allowed chain length (in graph hops) between a leaf and a hub 
            for removal.
        min_t_value (int): Minimum time index (inclusive) to consider for pruning.
        max_t_value (int): Maximum time index (inclusive) to consider for pruning.
    
    Returns:
        None: Modifies `single_frame_skel_stack` in-place by removing short disconnected tracks.
    """
    
    
    keep_pruning = True
    
    counter = 0
    
    while keep_pruning == True:
    
        #print('counter', str(counter))
        G = nx.DiGraph()
    
        for i in range(single_frame_skel_stack.shape[0]):
        
            # get the unique ids at each frame
            unique_id_list = list(np.unique(single_frame_skel_stack[i,:,:]))
            unique_id_list.remove(0)
        
            # add the nodes to the graph, where the node is a tuple (t,id), where t is the timestep and id is the id at that timestep
            for curr_unique_id in unique_id_list:
        
                G.add_node((i, curr_unique_id))
    
    
    
        # add edges
        for i in range(single_frame_skel_stack.shape[0]-1):
            
            curr_ts = i
            next_ts = i+1
        
            low_frame_arr = single_frame_skel_stack[curr_ts,:,:]
            high_frame_arr = single_frame_skel_stack[next_ts,:,:]
        
            # Flatten and create a DataFrame
            df = pd.DataFrame({'A': low_frame_arr.flatten(), 'B': high_frame_arr.flatten()})
            
            # Drop duplicates first
            unique_df = df.drop_duplicates()
            
            # Filter out any pairs that contain a zero
            filtered_df = unique_df[(unique_df['A'] != 0) & (unique_df['B'] != 0)]
            
            # Convert the result back to a numpy array if needed
            unique_pairs = filtered_df.to_numpy()
        
            for overlap_match in unique_pairs:
                low_node = (curr_ts, overlap_match[0])
                high_node = (next_ts, overlap_match[1])
                G.add_edge(low_node, high_node)
    
    
        # DISJOIINT TRACKING
        
        terminated_nodes = [node for node in G.nodes() if len([x for x in G.successors(node)]) == 0 and node[0] != max_t_value]
        starting_nodes = [node for node in G.nodes() if len([x for x in G.predecessors(node)]) == 0 and node[0] != 0]
        
        len(terminated_nodes)
        
        counter_terminated = 0
        for terminated_node in terminated_nodes:
            #print(counter_terminated)
            counter_terminated+=1
            #print('-------')
            #print(terminated_node)
        
            prospective_matches = list()
        
            for starting_node in starting_nodes:
                if (starting_node[0] -1) == terminated_node[0]:
                    prospective_matches.append(starting_node)
        
            #print(prospective_matches)
        
            
            if len(prospective_matches) > 0:
        
                term_node_snapshot = single_frame_skel_stack[terminated_node[0],...] == terminated_node[1]
                term_node_snapshot = term_node_snapshot* 1
        
                #plt.imshow(term_node_snapshot)
                #plt.show()
        
                indices = np.argwhere(term_node_snapshot)
                term_average_loc = np.mean(indices, axis=0)
                term_loc_tuple = (float(term_average_loc[0]), float(term_average_loc[1]))
                term_num_px = len(indices)
        
                #print(term_loc_tuple)
                #print(term_num_px)
        
                prospective_starting_node_list = list()
                
                for starting_node in prospective_matches:
        
                    start_node_snapshot = single_frame_skel_stack[starting_node[0],...] == starting_node[1]
                    start_node_snapshot = start_node_snapshot* 1
                    prospective_starting_node_list.append(start_node_snapshot)
                    #plt.imshow(start_node_snapshot)
                    #plt.show()
        
                    curr_indices = np.argwhere(start_node_snapshot)
                    curr_start_average_loc = np.mean(curr_indices, axis=0)
                    curr_start_loc_tuple = (float(curr_start_average_loc[0]), float(curr_start_average_loc[1]))
        
                    curr_start_num_px = len(curr_indices)
        
        
                    #print(curr_start_loc_tuple)
                    #print(curr_start_num_px)
        
        
                    px_diff = abs(term_num_px - curr_start_num_px)
                    #print(px_diff)
        
        
                    centroid_dist = math.dist(term_loc_tuple, curr_start_loc_tuple)
                    #print(centroid_dist)
        
        
                    if px_diff < 150 and centroid_dist < 65:
                        G.add_edge(terminated_node, starting_node)
    
    
        
        # convert to undirected
        GU = G.to_undirected()
    
        #print('starting cleanup')
        
        n = fade_max_length
        
        # Identify leaf nodes (degree == 1) and hub nodes (degree > 2)
        leaf_nodes = [node for node in GU.nodes() if GU.degree[node] == 1]
        hub_nodes = set(node for node in GU.nodes() if GU.degree[node] > 2)
        
        terminated_subsets = []
        
        # Perform BFS from each leaf node
        for leaf in leaf_nodes:
            queue = [(leaf, [leaf], 0)]  # (current_node, path, depth)
            visited = set([leaf])
            found = False
        
            while queue and not found:
                current_node, path, depth = queue.pop(0)
        
                if depth >= n:
                    continue
        
                for neighbor in GU.neighbors(current_node):
                    if neighbor in hub_nodes:
                        # Found a hub node within distance n
                        terminated_subsets.append(path)  # Exclude the hub node
                        found = True  # Stop searching from this leaf
                        break
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor], depth + 1))
    
    
        terminated_subsets = [x for x in terminated_subsets if min_t_value not in [y[0] for y in x]]
        terminated_subsets = [x for x in terminated_subsets if max_t_value not in [y[0] for y in x]]
        #print(len(terminated_subsets))
        counter+=1
        
        if len(terminated_subsets) == 0:
            break
        
        for curr_cycle in terminated_subsets:
            
            
            
            ts_range = [x[0] for x in curr_cycle]
            
            ts_low = min(ts_range)
            ts_high = max(ts_range)
        
            if min_t_value == ts_low or max_t_value == ts_high:
                continue
        
            #print(curr_cycle)
        
            for node_to_delete in curr_cycle:
                #print(node_to_delete)
        
                ts = node_to_delete[0]
                val = node_to_delete[1]
        
        
                single_frame_skel_stack[ts,...][single_frame_skel_stack[ts,...] == val] = 0
    
    
        print('ending cleanup')





    # This code blocks corrects "blips" which is small patches of signal that is picked up for less than x seconds (defined above)

    """
    Removes all connected components from the graph `G` that are smaller than a 
    specified length, and deletes corresponding labels from the input skeleton stack.
    
    Args:
        G (networkx.DiGraph): A directed graph with nodes in the format (t, label).
        single_frame_skel_stack (np.ndarray): A 3D labeled array of shape (T, H, W).
        blip_max_length (int): The maximum size threshold under which components are removed.
    
    Returns:
        None: Modifies `G` and `single_frame_skel_stack` in-place.
    """
    
    GU = G.to_undirected()
    s_n = blip_max_length
    small_subgraphs = [GU.subgraph(component).copy() for component in nx.connected_components(GU) if len(component) < s_n]
    
    for small_sg in small_subgraphs:
        
        
        ts_range = [x[0] for x in small_sg]
        
        ts_low = min(ts_range)
        ts_high = max(ts_range)
    
    
        for node_to_delete in small_sg:
            #print(node_to_delete)
    
            ts = node_to_delete[0]
            val = node_to_delete[1]
    
            single_frame_skel_stack[ts,...][single_frame_skel_stack[ts,...] == val] = 0
    
            G.remove_node(node_to_delete)
    
    
    
    
    
    # Relabels the ids so that if the same instance doesnt fission or fuse between frames it keeps the same id
    pos = assign_positions(G)
    
    new_frame_stack = np.zeros_like(single_frame_skel_stack).astype(np.uint32)
    for key, val in pos.items():
    
        ts = key[0]
        old_val = key[1]
        new_val = val[1] + 1
        new_frame_stack[ts,...][single_frame_skel_stack[ts,...] == old_val] = new_val
    mapping_dict = {key: (value[0], value[1] + 1)  for key, value in pos.items()}
    
    G_FINAL = nx.relabel_nodes(G, mapping_dict)
    



    # Uses the skeletons as instance watershed seeds to create the final output

    # create the fmask watersheds
    #fmask_dist_list = list()
    ws_instance_list = list()
    corrected_skels = list()
    
    for i in range(fmask_pred_arr.shape[0]):
        #print(i)
    
        # dist transforms for the full mask for a single frame
        curr_frame_fmask = fmask_pred_arr[i,:,:]
        curr_frame_fmask_dist = ndi.distance_transform_edt(curr_frame_fmask).astype(np.float16)
        #fmask_dist_list.append(curr_frame_fmask_dist)
    
    
        # get unique skeletons for a single frame
        curr_frame_single_skeleton = new_frame_stack[i,:,:]
        corrected_skels.append(curr_frame_single_skeleton)
        
    
        # watershed full mask
        curr_frame_ws_label = watershed(-curr_frame_fmask_dist, curr_frame_single_skeleton, mask=(curr_frame_fmask)).astype(np.uint32)
        ws_instance_list.append(curr_frame_ws_label)
    
    # stack watersheded full masks
    ws_instance_stack = np.stack(ws_instance_list)
    corrected_skell_stack = np.stack(corrected_skels)
    
    ws_instance_stack_fixed = ws_instance_stack.copy()
    
    skel_set = set()
    
    for i in range(max_t_value+1):
        curr_frame = corrected_skell_stack[i,:,:]
        for each in np.unique(curr_frame):
            if each != 0:
                skel_set.add((i,each))
    
    
    
    ws_mask_set = set()
    
    for i in range(max_t_value+1):
        curr_frame = ws_instance_stack[i,:,:]
        for each in np.unique(curr_frame):
            if each != 0:
                ws_mask_set.add((i,each))
    
    missing_set_list = list(skel_set - ws_mask_set)
    
    for each in missing_set_list:
    
        curr_skel_frame = corrected_skell_stack[each[0],:,:]
        adder_array = np.array(curr_skel_frame == each[1])*each[1].astype(np.uint32)
    
        ws_instance_stack_fixed[each[0],:,:] += adder_array
    
    ws_instance_stack_fixed.shape
    
    skel_set = set()
    
    for i in range(max_t_value+1):
        curr_frame = corrected_skell_stack[i,:,:]
        for each in np.unique(curr_frame):
            if each != 0:
                skel_set.add((i,each))
    
    
    
    ws_mask_set = set()
    
    for i in range(max_t_value+1):
        curr_frame = ws_instance_stack_fixed[i,:,:]
        for each in np.unique(curr_frame):
            if each != 0:
                ws_mask_set.add((i,each))
    
    list(skel_set - ws_mask_set)
    
    set(G_FINAL.nodes())^ws_mask_set
    




    full_end_time = timeit.default_timer() - full_start_time
    print(full_end_time)




    # Saves results
    out_folder = out_dir
    
    np.savez_compressed(os.path.join(out_folder,'instance_skel.npz'), corrected_skell_stack)
    
    # Save the graph to a file with pickle
    with open(os.path.join(out_folder,'matching_nx_graph.pickle'), 'wb') as f:
        pickle.dump(G_FINAL, f)
    
    np.savez_compressed(os.path.join(out_folder,'instance_masks.npz'), ws_instance_stack_fixed)
    
    np.savez_compressed(os.path.join(out_folder,'raw.npz'), curr_raw_arr)
    




































