import networkx as nx
import numpy as np
import pickle
from skimage.morphology import skeletonize
import napari
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import networkx as nx
import os
import concurrent.futures
import time
    

def skeletonize_labels(label_array):
    """
    Skeletonizes each labeled region in a 2D label array.

    Args:
        label_array (np.ndarray): A 2D array of labeled regions.

    Returns:
        np.ndarray: A 2D array of the same shape containing skeletonized labels.
    """
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels > 0]

    skeletonized = np.zeros_like(label_array, dtype=label_array.dtype)

    for label in unique_labels:
        binary_mask = (label_array == label)
        skeleton = skeletonize(binary_mask)
        skeletonized[skeleton] = label

    return skeletonized


def skeletons_to_graph(skeletonized_array):
    """
    Converts a 2D skeletonized label image into an undirected graph where nodes represent pixel coordinates.

    Args:
        skeletonized_array (np.ndarray): A 2D array with skeletonized labeled regions.

    Returns:
        networkx.Graph: A graph where nodes are (row, col) and edges represent 8-connected neighbors.
    """
    G = nx.Graph()
    rows, cols = skeletonized_array.shape

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    for row in range(rows):
        for col in range(cols):
            if skeletonized_array[row, col] > 0:
                node_id = (row, col)
                label = skeletonized_array[row, col]
                G.add_node(node_id, label=label)

                for dr, dc in neighbors:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols and skeletonized_array[nr, nc] > 0:
                        G.add_edge(node_id, (nr, nc))

    high_degree_nodes = [n for n in G.nodes if G.degree(n) > 2]
    merged_nodes = set()

    for node in high_degree_nodes:
        if node in merged_nodes:
            continue

        neighbors_to_merge = [
            neighbor for neighbor in G.neighbors(node) if G.degree(neighbor) > 2
        ]

        if neighbors_to_merge:
            new_node = random.choice([node] + neighbors_to_merge)
            for neighbor in [node] + neighbors_to_merge:
                if neighbor != new_node:
                    for nn in G.neighbors(neighbor):
                        if nn != new_node:
                            G.add_edge(new_node, nn)
                    G.remove_node(neighbor)
                    merged_nodes.add(neighbor)

    return G


def simplify_graph(G, sample_factor, L):
    """
    Simplifies a graph by retaining only key nodes and sampling intermediate paths.

    Args:
        G (networkx.Graph): Input graph.
        sample_factor (int): Number of nodes to skip between samples.
        L (int): Label to assign to all nodes in the simplified graph.

    Returns:
        networkx.Graph: A simplified version of the input graph.
    """
    key_nodes = [n for n in G.nodes if G.degree(n) == 1 or G.degree(n) > 2]
    simplified_G = nx.Graph()

    if len(G.nodes) == 1:
        single_node = list(G.nodes)[0]
        simplified_G.add_node(single_node, label=L)
        return simplified_G

    visited_edges = set()

    for key_node in key_nodes:
        for neighbor in G.neighbors(key_node):
            if (key_node, neighbor) in visited_edges or (neighbor, key_node) in visited_edges:
                continue

            path = [key_node]
            current_node = neighbor

            while G.degree(current_node) == 2:
                neighbors = list(G.neighbors(current_node))
                next_node = neighbors[0] if neighbors[0] != path[-1] else neighbors[1]
                path.append(current_node)
                current_node = next_node

            path.append(current_node)

            for i in range(len(path) - 1):
                visited_edges.add((path[i], path[i + 1]))

            if len(path) > 2:
                num_samples = len(path) // sample_factor
                indices = np.linspace(0, len(path) - 1, num=num_samples + 2).astype(int)
                sampled_path = [path[i] for i in indices]
            else:
                sampled_path = path

            for node in sampled_path:
                if node not in simplified_G:
                    simplified_G.add_node(node, label=L)

            for i in range(len(sampled_path) - 1):
                simplified_G.add_edge(sampled_path[i], sampled_path[i + 1])

    return simplified_G


def simplify_full_frame(full_pos_graph, node_list, simplify_factor):
    """
    Simplifies labeled subgraphs in a full position graph and combines the result.

    Args:
        full_pos_graph (networkx.Graph): The full position graph with labeled nodes.
        node_list (List[Tuple[int, int]]): List of node IDs to extract subgraphs by label.
        simplify_factor (int): Sampling factor to reduce intermediate nodes.

    Returns:
        networkx.Graph: The merged simplified graph.
    """
    combined_graph = nx.Graph()

    for node_id in node_list:
        subgraph_nodes = [n for n, d in full_pos_graph.nodes(data=True) if d['label'] == node_id[1]]
        subgraph = full_pos_graph.subgraph(subgraph_nodes)
        simplified_pos_graph = simplify_graph(subgraph, simplify_factor, node_id[1])
        combined_graph.add_nodes_from(simplified_pos_graph.nodes(data=True))
        combined_graph.add_edges_from(simplified_pos_graph.edges)

    return combined_graph


def cartesian_distance(node1, node2):
    """
    Computes Euclidean distance between two (x, y) coordinates.

    Args:
        node1 (Tuple[float, float]): First node position.
        node2 (Tuple[float, float]): Second node position.

    Returns:
        float: Euclidean distance.
    """
    x1, y1 = node1
    x2, y2 = node2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def edge_similarity(graph1, node1, graph2, node2):
    """
    Computes absolute difference in node degrees.

    Args:
        graph1 (networkx.Graph): First graph.
        node1: Node in the first graph.
        graph2 (networkx.Graph): Second graph.
        node2: Node in the second graph.

    Returns:
        int: Absolute difference in node degree.
    """
    return abs(graph1.degree(node1) - graph2.degree(node2))


def connectivity_similarity(graph1, node1, graph2, node2):
    """
    Computes similarity between neighborhoods using the Jaccard Index.

    Args:
        graph1 (networkx.Graph): First graph.
        node1: Node in the first graph.
        graph2 (networkx.Graph): Second graph.
        node2: Node in the second graph.

    Returns:
        float: Value between 0 and 1 representing connectivity similarity.
    """
    neighbors1 = set(graph1.neighbors(node1))
    neighbors2 = set(graph2.neighbors(node2))

    if len(neighbors1.union(neighbors2)) == 0:
        return 0
    return 1 - len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))


def clustering_coefficient_similarity(graph1, node1, graph2, node2):
    """
    Computes the absolute difference in local clustering coefficients.

    Args:
        graph1 (networkx.Graph): First graph.
        node1: Node in the first graph.
        graph2 (networkx.Graph): Second graph.
        node2: Node in the second graph.

    Returns:
        float: Absolute difference in clustering coefficients.
    """
    c1 = nx.clustering(graph1, node1)
    c2 = nx.clustering(graph2, node2)
    return abs(c1 - c2)


def cost_function(node1, node2, graph1, graph2, cartesian_weight=1.0, edge_weight=1.0, connectivity_weight=1.0, clustering_weight=1.0):
    """
    Computes a weighted cost between two nodes based on spatial and graph structural properties.

    Args:
        node1, node2: Nodes to compare.
        graph1, graph2 (networkx.Graph): Graphs containing the nodes.
        cartesian_weight, edge_weight, connectivity_weight, clustering_weight (float): Weights for different metrics.

    Returns:
        float: Total weighted cost between node1 and node2.
    """
    total_cost = 0.0

    if cartesian_weight > 0:
        total_cost += cartesian_weight * cartesian_distance(node1, node2)

    if edge_weight > 0:
        total_cost += edge_weight * edge_similarity(graph1, node1, graph2, node2)

    if connectivity_weight > 0:
        total_cost += connectivity_weight * connectivity_similarity(graph1, node1, graph2, node2)

    if clustering_weight > 0:
        total_cost += clustering_weight * clustering_coefficient_similarity(graph1, node1, graph2, node2)

    return total_cost


def greedy_node_matching(graph1, graph2, cartesian_weight=1.0, edge_weight=1.0, connectivity_weight=1.0, clustering_weight=1.0):
    """
    Performs greedy matching of nodes from graph1 to graph2 based on lowest-cost pairing.

    Args:
        graph1, graph2 (networkx.Graph): Graphs with nodes to match.
        cartesian_weight, edge_weight, connectivity_weight, clustering_weight (float): Cost weights.

    Returns:
        List[Tuple]: List of matched node pairs (node_from_graph1, best_match_from_graph2).
    """
    matches = []

    for node1 in graph1.nodes:
        best_match = None
        best_cost = float('inf')

        for node2 in graph2.nodes:
            current_cost = cost_function(node1, node2, graph1, graph2,
                                         cartesian_weight,
                                         edge_weight,
                                         connectivity_weight,
                                         clustering_weight)

            if current_cost < best_cost:
                best_cost = current_cost
                best_match = node2

        matches.append((node1, best_match))

    return matches


def get_inter_frame_graph(low_frame_idx, raw_arr, frame_matching_G, skel_dict):
    """
    Constructs an inter-frame graph by matching skeleton subgraphs from two consecutive frames 
    using greedy node matching and appending the results to a combined graph.

    Args:
        low_frame_idx (int): Index of the lower frame in the time pair.
        raw_arr (np.ndarray): Full time-series raw input array (not directly used here).
        frame_matching_G (networkx.Graph): Graph with nodes labeled as (frame, label) 
            representing matched components across frames.
        skel_dict (dict): Dictionary mapping frame indices to simplified skeleton graphs. 
            Each graph contains nodes with 2D positions and 'label' attributes.

    Returns:
        networkx.Graph: A graph combining spatial nodes from two consecutive frames, 
        with edges between matched nodes based on greedy spatial + structural similarity.
    """
    print(f'doing frame {low_frame_idx}')

    high_frame_idx = low_frame_idx + 1

    low_frame_nodes = [x for x in frame_matching_G.nodes() if x[0] == low_frame_idx]
    high_frame_nodes = [x for x in frame_matching_G.nodes() if x[0] == high_frame_idx]

    curr_matching_G = frame_matching_G.subgraph(low_frame_nodes + high_frame_nodes)

    simplified_low_G = skel_dict[low_frame_idx]
    simplified_high_G = skel_dict[high_frame_idx]

    mapping = {(x, y): (x, y, low_frame_idx) for x, y in simplified_low_G.nodes()}
    simplified_low_G_wz = nx.relabel_nodes(simplified_low_G, mapping)

    mapping = {(x, y): (x, y, high_frame_idx) for x, y in simplified_high_G.nodes()}
    simplified_high_G_wz = nx.relabel_nodes(simplified_high_G, mapping)

    combined_graph = nx.compose(simplified_low_G_wz, simplified_high_G_wz)

    curr_matching_G_U = curr_matching_G.to_undirected()
    matched_subgraph_list = [curr_matching_G.subgraph(c) for c in nx.connected_components(curr_matching_G_U)]
    matching_list = []

    for matching_subgraph in matched_subgraph_list:
        curr_dict = {}

        low_nodes = [x for x in matching_subgraph.nodes if x[0] == low_frame_idx]
        low_pos_node_dict = {}
        for curr_node in low_nodes:
            subgraph_nodes = [n for n, d in simplified_low_G.nodes(data=True) if d['label'] == curr_node[1]]
            low_pos_node_dict[curr_node] = subgraph_nodes
        curr_dict['low'] = low_pos_node_dict

        high_nodes = [x for x in matching_subgraph.nodes if x[0] == high_frame_idx]
        high_pos_node_dict = {}
        for curr_node in high_nodes:
            subgraph_nodes = [n for n, d in simplified_high_G.nodes(data=True) if d['label'] == curr_node[1]]
            high_pos_node_dict[curr_node] = subgraph_nodes
        curr_dict['high'] = high_pos_node_dict

        curr_dict['matching_subgraph'] = matching_subgraph

        low_pos_node_list = sum(low_pos_node_dict.values(), [])
        high_pos_node_list = sum(high_pos_node_dict.values(), [])

        curr_dict['low_pos_subgraph'] = simplified_low_G.subgraph(low_pos_node_list)
        curr_dict['high_pos_subgraph'] = simplified_high_G.subgraph(high_pos_node_list)

        matching_list.append(curr_dict)

    for matching_dict in matching_list:
        G1 = matching_dict['low_pos_subgraph']
        G2 = matching_dict['high_pos_subgraph']
        optimal_matching = greedy_node_matching(G1, G2, cartesian_weight=1.0, edge_weight=1.0, connectivity_weight=0, clustering_weight=0)

        optimal_matching_z = [
            ((a[0], a[1], low_frame_idx), (b[0], b[1], high_frame_idx))
            for a, b in optimal_matching if a and b
        ]
        combined_graph.add_edges_from(optimal_matching_z)

        G1 = matching_dict['high_pos_subgraph']
        G2 = matching_dict['low_pos_subgraph']
        optimal_matching = greedy_node_matching(G1, G2, cartesian_weight=1.0, edge_weight=1.0, connectivity_weight=0, clustering_weight=0)

        optimal_matching_z = [
            ((a[0], a[1], high_frame_idx), (b[0], b[1], low_frame_idx))
            for a, b in optimal_matching if a and b
        ]
        combined_graph.add_edges_from(optimal_matching_z)

    return combined_graph



def run_multithreaded(N, num_threads):
    """
    Executes `get_inter_frame_graph` for frames 0 through N in parallel threads.

    Args:
        N (int): Highest low-frame index to process (i.e., runs from 0 to N).
        num_threads (int): Maximum number of threads to run concurrently.

    Returns:
        dict: A dictionary mapping low frame indices to their resulting combined graphs.
    """
    result_dict = {}

    def worker(value):
        result = get_inter_frame_graph(value, raw_arr, frame_matching_G, skel_dict)
        result_dict[value] = result

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(worker, range(N + 1))

    return result_dict




if __name__ == "__main__":

    # inputs
    parser = argparse.ArgumentParser(description="Process an integer input.")
    parser.add_argument("in_num", type=int, help="An integer input for the program.")
    parser.add_argument("in_path", type=str, help="Input path for the folder that contains the processed outputs of the postprocessing step.")
    args = parser.parse_args()

    in_num = args.in_num
    in_path = args.in_path

    print(f"Input number is: {in_num}")
    str_in_num = str(in_num)


    # reads the instance full masks from 3_1, the raw array, and the matching graph
    instance_skel_arr = np.load(os.path.join(in_path,f'{str_in_num}/instance_masks.npz'))['arr_0'].astype(np.uint16)

    raw_arr = np.load(os.path.join(in_path,f'{str_in_num}/raw.npz'))['arr_0'].astype(np.uint8)
    
    # graph structure
    with open(os.path.join(in_path,f'{str_in_num}/matching_nx_graph.pickle', 'rb')) as f:
        frame_matching_G = pickle.load(f)
    


    # skeletonize each frame
    skel_dict = dict()
    
    for i in range(raw_arr.shape[0]):
    
        low_frame_idx = i
    
        low_frame_model_skeleton = instance_skel_arr[low_frame_idx,:,:]
        low_frame_nodes = [x for x in frame_matching_G.nodes() if x[0] == low_frame_idx]
        low_frame_skeletonized = skeletonize_labels(low_frame_model_skeleton)
        low_G = skeletons_to_graph(low_frame_skeletonized)
        simplified_low_G = simplify_full_frame(low_G, low_frame_nodes, 5)
    
        skel_dict[i] = simplified_low_G



    # does the interframe matching of the nodes  multithreaded 
    
    N = len(skel_dict)-1  # Upper limit for function inputs
    num_threads = -1  # Number of threads
    
    start_time = time.time()
    
    results = run_multithreaded(N, num_threads)
    
    taken = time.time() - start_time
    print(taken)
    

    
    # Get a list of all the graphs from the dictionary
    graphs = list(results.values())
    
    # Perform the union of all graphs, merging common nodes
    composed_graph = nx.compose_all(graphs)
    
    
    # # Save the graph to a file with pickle
    # with open('pos_matching_graph.pickle', 'wb') as f:
    #     pickle.dump(composed_graph, f)
    # # Save the graph to a file with pickle
    with open(os.path.join(f'processed_outputs/{str_in_num}','pos_matching_graph.pickle'), 'wb') as f:
        pickle.dump(composed_graph, f)
    







