


#### -------
#### Imports
#### -------


# Standard library imports
import argparse
from collections import defaultdict
import concurrent.futures
import gc
import os
import pickle
import time

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm




#### ------------------------
#### Sample Loading
#### ------------------------


def get_seeds(df: pd.DataFrame, region_name: str, region_map: dict[str, list[str] | None], frame: int = 0) -> list[tuple[int, int]]:
    """
    Extract initial seed nodes for spread curve analysis.  
    
    Parameters:
    df (pd.DataFrame): Dataframe containing per-object measurements.
    region_name (str): Name of the region to seed from.
    region_map (dict): Mapping of region names to lists of region labels.
    frame (int, optional): Frame index associated with seed node locations.

    Returns
        List of seed nodes represented as (frame, object_id)
    """

    if region_name == "Global":
        mask = df['Time (Seconds)'] == df['Time (Seconds)'].min()
    else:
        mask = (df['Time (Seconds)'] == df['Time (Seconds)'].min()) & (df['Region'].isin(region_map[region_name]))
    oids = df.loc[mask, 'Label'].unique()

    return [(frame, oid) for oid in oids]


def load_graph_and_csv(sample_number: int) -> tuple[object, pd.DataFrame]:
    """
    Load the tracking graph and corresponding per-object CSV data for a sample.

    Parameters:
        sample_number (int): Sample identifies to locate graph/CSV files.

    Returns:
        tuple:
            G (nx.Graph): NetworkX directed graph with nodes as (frame, object_id).
            df (pd.Dataframe): DataFrame containing per-object morphological and metadata features.
    """

    parent_path = os.path.join(os.getcwd(), f'segmentation_data/{sample_number}')
    graph_path = parent_path + '/matching_nx_graph.pickle'
    G = np.load(graph_path, allow_pickle=True)
    df = pd.read_csv(f"processed_outputs_60secbin/{sample_number}/processed_data_analysis/general_morphological_features_per_object.csv")

    return G, df





#### --------------------------------
#### Spread Ratio Processing
#### --------------------------------


def compute_single_seed_with_cache(seed: tuple[int, int], seed_idx: int, t0: int, t_max: int, normalization: str, objects_by_frame: dict[int, set[int]], all_objects: set[int], 
                                   successors_cache: dict[tuple[int, int], list[tuple[int, int]]], 
                                   predecessors_cache: dict[tuple[int, int], list[tuple[int, int]]] | None = None, 
                                   ancestry_cache: dict[tuple[int, int], set[int]] | None = None) -> tuple[int, list[int], list[float]]:
    """
    Compute the spread curve for a single seed node using cached graph traversal data.

    Parameters:
        seed (tuple): Seed node represented as (frame, object_id).
        seed_idx (int): Index of the seed.
        t0 (int): Initial frame index.
        t_max (int): Final frame index.
        normalization (str): Spread curve normalization method.
        objects_by_frame (dict): Mapping of frame indices to the set of object IDs present at that particular frame.
        all_objects (set): Set of all object IDs in the graph.
        successors_cache (dict): Precomputed mapping of each node to its succesor nodes.
        predecessors_cache (dict): Precomputed mapping of each node to its predecessor nodes.
        ancestry_cache (dict): Precomputed full ancestry of object IDs for each node.

    Returns:
        tuple:
            seed_idx (int): Index of the seed.
            times (list): Frame indices.
            spread_curve (list): Fraction of objects reached at each time point.
    """
    
    if normalization == "global" or normalization == "global_bidirectional":
        denom = len(all_objects)
    elif normalization == "reachable":
        # Quick BFS using cache
        reachable = set()
        queue = [seed]
        visited = {seed}
        
        while queue:
            node = queue.pop(0)
            reachable.add(node)
            for succ in successors_cache.get(node, []):
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        
        denom = len({n[1] for n in reachable})
    elif normalization == "dynamic":
        denom = None
    
    reached = {seed}
    reached_objects = {seed[1]}
    spread_curve = []
    times = range(t0, t_max + 1)

    for t in times:
        current_nodes = [n for n in reached if n[0] == t]
        for node in current_nodes:
            for succ in successors_cache.get(node, []):
                if succ not in reached:
                    reached.add(succ)
                    reached_objects.add(succ[1])
        
        # NEW: For bidirectional counting, use pre-computed ancestry
        if normalization == "global_bidirectional" and ancestry_cache:
            newly_reached_nodes = [n for n in reached if n[0] == t]
            for node in newly_reached_nodes:
                # Use pre-computed full ancestry instead of BFS each time
                if node in ancestry_cache:
                    reached_objects.update(ancestry_cache[node])

        if normalization == "global" or normalization == "global_bidirectional" or normalization == "reachable":
            frac = len(reached_objects) / max(1, denom)
        elif normalization == "dynamic":
            denom_t = len(objects_by_frame[t])
            frac = len(reached_objects & objects_by_frame[t]) / max(1, denom_t)

        spread_curve.append(frac)

    return seed_idx, list(times), spread_curve


def process_chunk(chunk_seeds: list[tuple[int, tuple[int, int]]], t0: int, t_max: int, normalization: str, objects_by_frame: dict[int, set[int]], all_objects: set[int],
                  successors_cache: dict[tuple[int, int], list[tuple[int, int]]], 
                  predecessors_cache: dict[tuple[int, int], list[tuple[int, int]]] | None = None, 
                  ancestry_cache: dict[tuple[int, int], set[int]] | None = None) -> list[tuple[int, list[int], list[float]]]:
    """
    Process a chunk of seeds and compute their spread curves.

    Parameters:
        chunk_seeds (list): List of (seed_index, seed_node) tuples.
        t0 (int): Initial frame index.
        t_max (int): Final frame index.
        normalization (str): Spread curve normalization method.
        objects_by_frame (dict): Mapping of frame indices to the set of object IDs present at that particular frame.
        all_objects (set): Set of all object IDs in the graph.
        successors_cache (dict): Precomputed mapping of each node to its succesor nodes.
        predecessors_cache (dict): Precomputed mapping of each node to its predecessor nodes.
        ancestry_cache (dict): Precomputed full ancestry of object IDs for each node.

    Returns:
        results (list[tuple]):
            seed_idx (int): Index of the seed.
            times (list): Frame indices.
            spread_curve (list): Fraction of objects reached at each time point.
    
    """
    
    results = []
    
    for seed_idx, seed in chunk_seeds:
        s_idx, times, spread_curve = compute_single_seed_with_cache(seed, seed_idx, t0, t_max, normalization, objects_by_frame, all_objects, successors_cache, predecessors_cache, ancestry_cache)
        results.append((s_idx, times, spread_curve))
    
    return results
        

def compute_multiple_curves_chunked(G, seeds: list[tuple[int, int]], t0: int, t_max: int, normalization: str = "global", 
                                    successors_cache: dict | None = None, predecessors_cache: dict | None = None, ancestry_cache: dict | None = None, 
                                    max_workers: int = 8, chunk_size: int = 50) -> tuple[list[int], np.ndarray]:
    """
    Compute spread curves for multiple seeds in parallel.

    Parameters:
        G (nx.Graph): NetworkX directed graph with nodes as (frame, object_id).
        seeds (list): List of seed node tuples represented as (frame, object_id).
        t0 (int): Initial frame index.
        t_max (int): Final frame index.
        normalization (str): Spread curve normalization method.
        successors_cache (dict): Precomputed mapping of each node to its succesor nodes.
        predecessors_cache (dict): Precomputed mapping of each node to its predecessor nodes.
        ancestry_cache (dict): Precomputed full ancestry of object IDs for each node.
        max_workers (int): Number of parallel worker threads.
        chunk_size (int): Number of seeds per parallel chunk.

    Returns:
        tuple:
            times (list): Frame indices.
            all_curves (np.ndarray): Array containing spread curve data.
    
    """
    start_time = time.time()

    t0 = 0
    t_max = max(list(nodes[0] for nodes in G.nodes())) + 1

    objects_by_frame = defaultdict(set)
    all_objects = set()
    for node in G.nodes:
        frame, obj_id = node
        if t0 <= frame <= t_max:
            objects_by_frame[frame].add(obj_id)
            all_objects.add(obj_id)
    
    if successors_cache is None:
        successors_cache = {}
        for node in tqdm(G.nodes, desc="Caching successors"):
            if t0 <= node[0] <= t_max:
                successors_cache[node] = [succ for succ in G.successors(node) if t0 <= succ[0] <= t_max]
            
    pre_comp_time = time.time() - start_time
    print(f"Pre-computation time: {pre_comp_time:.2f}s")
    
    indexed_seeds = list(enumerate(seeds))
    chunks = [indexed_seeds[i:i+chunk_size] for i in range(0, len(indexed_seeds), chunk_size)]
    
    # Pass ancestry_cache to workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_chunk, chunk, t0, t_max, normalization, 
                objects_by_frame, all_objects, successors_cache, 
                predecessors_cache if normalization == "global_bidirectional" else None,
                ancestry_cache if normalization == "global_bidirectional" else None
            ) 
            for chunk in chunks
        ]
        with tqdm(total=len(chunks)) as pbar:
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
                pbar.update(1)
                
    results.sort(key=lambda x: x[0])
    times = results[0][1]
    all_curves = np.vstack([r[2] for r in results])
    end_time = time.time() - start_time
    print(f"Total computation time: {end_time:.2f}s")
    
    return times, all_curves


def process_spread_curve_calculations(sample_number: int, regions_order: list[str], region_map: dict[str, list[str] | None], scale: str = "minutes", normalization: str = "global", max_workers: int = 8, chunk_size: int = 8, output_dir: str = "spread_ratio_data_cache") -> str:
    """
    Compute and cache epidemic-like spread curves for a single sample.

    Parameters:
        sample_number (int): Sample identifies to locate graph/CSV files.
        regions_order (list): Ordered list of region strings to comput spread curves for.
        region_map (dict): Mapping of region names to lists of region labels.
        scale (str): Time scale (seconds or minutes).
        normalization (str): Spread curve normalization method.
        max_workers (int): Number of parallel worker threads.
        chunk_size (int): Number of seeds per parallel chunk.
        output_dir (str): Directory to save spread curve results as pickle files.

    Returns:
        output_file (str): Path to the saved pickle file containing spread curve results.
    """
    os.makedirs(output_dir, exist_ok=True)

    if scale.lower() == "seconds":
        time_scale = 0.5
        x_label = "Time (secs)"
    elif scale.lower() == "minutes":
        time_scale = 0.5/60
        x_label = "Time (mins)"
    else:
        time_scale = 1
        x_label = "Frame"
    
    

    print(f"\nSample {sample_number}:")
    G, df = load_graph_and_csv(sample_number)
    
    print(f"  Graph has {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    max_frames = max(list(nodes[0] for nodes in G.nodes()))
    t0 = 0
    t_max = max_frames + 1
    
    successors_cache = {}
    predecessors_cache = {}
    ancestry_cache = {}
    sample_data = {}

    for node in tqdm(G.nodes, desc="Caching successors and predecessors"):
        if t0 <= node[0] <= t_max:
            successors_cache[node] = [succ for succ in G.successors(node) if t0 <= succ[0] <= t_max]
            if normalization == "global_bidirectional":
                predecessors_cache[node] = [pred for pred in G.predecessors(node) if t0 <= pred[0] <= t_max]
    
    # Pre-compute full ancestry for all nodes (only for bidirectional)
    if normalization == "global_bidirectional":
        print("Pre-computing ancestry for all nodes...")
        for node in tqdm(G.nodes, desc="Computing ancestry"):
            if t0 <= node[0] <= t_max:
                # BFS backwards to get all ancestor object IDs
                ancestor_objs = set()
                queue = [node]
                visited = {node}
                
                while queue:
                    current = queue.pop(0)
                    ancestor_objs.add(current[1])  # Add object ID
                    
                    if current in predecessors_cache:
                        for pred in predecessors_cache[current]:
                            if pred not in visited:
                                visited.add(pred)
                                queue.append(pred)
                
                ancestry_cache[node] = ancestor_objs

    for region in regions_order:
        print(f"    Region: {region}")
        seeds = get_seeds(df, region, region_map)

        if not seeds:
            print("    No seeds found")
            continue
        
        print(f"    Processing {len(seeds)} seeds...")

        times, curves = compute_multiple_curves_chunked(
            G, seeds, t0=0, t_max=max_frames+1, 
            normalization=normalization,  
            successors_cache=successors_cache,
            predecessors_cache=predecessors_cache if normalization == "global_bidirectional" else None,
            ancestry_cache=ancestry_cache if normalization == "global_bidirectional" else None,
            max_workers=max_workers, chunk_size=chunk_size
        )
        
        avg_curve = np.mean(curves, axis=0)
        if region not in sample_data:
            sample_data[region] = {}
        sample_data[region]['curves'] = curves
        sample_data[region]['avg'] = avg_curve
            
    times_scaled = [t * time_scale for t in times]

    output_file = os.path.join(output_dir, f"sample_{sample_number}_{normalization}.pkl")
    with open(output_file, 'wb') as file:
        pickle.dump({'sample_number': sample_number,
                     'regions': sample_data,
                     'times': times_scaled,
                     'normalization_method': normalization}, file)
    print(f"Time range: {times_scaled[0]:.2f} to {times_scaled[-1]:.2f} {x_label.lower()}")

    # Clean up large objects
    del G, df, successors_cache, predecessors_cache, ancestry_cache, sample_data
    gc.collect()

    return output_file


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Compute spread ratio curves for single samples.")
    parser.add_argument("--sample", type=int, help="Sample number")
    parser.add_argument("--regions_order", nargs="+", type=str, help="Regions to calculate spread ratio for")
    parser.add_argument("--normalization", type=str, default='global', choices=['global', 'global_bidirectional', 'reachable', 'dynamic'], help="Method to normalise spread ratio calculations by")
    parser.add_argument("--scale", type=str, default="minutes", choices=['frames', 'seconds', 'minutes'], help="Time scale/units for spread ratio analysis")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--chunk_size", type=int, default=8, help="Chunk size for parallel processing")
    parser.add_argument("--output_dir", type=str, default="spread_ratio_data_cache", help="Directory to save spread ratio results")

    args=parser.parse_args()

    region_map = {"Global": None,
                  "Telenuclear": ["Telenuclear"],
                  "Perinuclear": ["Perinuclear", "Transition"]}

    process_spread_curve_calculations(sample_number=args.sample, 
                                      regions_order=args.regions_order,
                                      region_map=region_map,
                                      scale=args.scale, 
                                      normalization=args.normalization, 
                                      max_workers=args.max_workers, 
                                      chunk_size=args.chunk_size, 
                                      output_dir=args.output_dir)
