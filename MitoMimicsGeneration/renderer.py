
### ----------------------- imports ---------------------------------- ###

# logic libraries
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import skimage
import random
import time
import math
from itertools import combinations
from scipy.ndimage.morphology import distance_transform_edt


# 2D PSF Kernel Generator
from astropy.convolution import Gaussian2DKernel, AiryDisk2DKernel


# blobs
import porespy as ps


# torch accell
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False



# io utils
from itertools import product, combinations
from logic.util import create_directory_if_empty_or_not_exists
import os 
from pathlib import Path
import yaml


import argparse




# --------------------- args to pass from file call ------------------

# load in the parser
parser = argparse.ArgumentParser(description="Gen sample training data")

# specific arguments
parser.add_argument("--seed", type=int, help="Seed value")
parser.add_argument("--cupy", default=False, type = bool, help="Use cupy over torch for faster computation on some systems")
parser.add_argument('--gpu', default=0, type=int, help='GPU to use')

# assign the args to their vars
args = parser.parse_args()
seed = args.seed
use_cupy_over_torch = args.cupy
gpu_number = args.gpu

# cupy accel
if use_cupy_over_torch:
    import cupy as cp



# ---------------------- load parameters from the yaml config file ----------------------

# open the yaml
with open('parameters_render.yaml', 'r') as f:
    parameters = yaml.safe_load(f)


# load the yaml params into vars, and print them
sim_save_path = parameters['save_load_params']['sim_save_path']
print('sim_save_path:', sim_save_path)
out_folder = parameters['save_load_params']['out_folder']
print('out_folder:', out_folder)
save_rate = parameters['render_params']['save_rate']
print('save_rate:', save_rate)
render_mult = parameters['render_params']['render_mult']
print('render_mult:', render_mult)
save_mult = parameters['render_params']['save_mult']
print('save_mult:', save_mult)
randomness_seed = parameters['render_params']['randomness_seed']
print('randomness_seed:', randomness_seed)
size_range = (parameters['invis_params']['invis_size_low'], parameters['invis_params']['invis_size_high'])
print('size_range:', size_range)
drop_set_range = (parameters['invis_params']['invis_num_low'], parameters['invis_params']['invis_num_high'])
print('drop_set_range:', drop_set_range)
shape_choice = parameters['fission_fusion_locs']['shape_choice']
print('shape_choice:', shape_choice)


do_compress = parameters['save_load_params']['do_compress']


# -----------------------------------------------
# -------------- PARAMS TO ADDD ----------------------
# -----------------------------------------------

    # TOADDPARAMS

# Fixed Params

# ------ BASE BACKGROUND
# background noise blob size in helpler func
bg_noise_blob_size = 50


# ------ CELL BACKGROUND

# conv factor for the cell bg
cell_bg_conv = 45
# central glow distribution (sigma of the gaussian kernal in the center of the cell background)
cell_bg_glow_central_sigma = 600


# ------ MITO MORPHOLOGY, PRE CONV
# multiplaction factor between the base mito morphology and the blob layer
mito_blob_impact_mult = 0.82


# ------ MITO SIGNAL

# mito dimness range
# half render mult 
half_render_mult_factor = 0.25

# ------ PERI MORPH

# cutoff for the averaged peri areas across all frames
common_peri_area_cutoff = 0.5
# ammount to add to the peri morphology for the initial averaging
peri_average_mito_width_add = 3


# DYNAMIC PARAMS

# photon conversion rate
photon_conversion_range = (20,30)
# blob noise added to the cell background
cell_bg_blobiness_parameter_range = (0.1,0.165)
# how much the cell blobiness impacts the cell bg signal
bg_blob_impact_factor_range = (0.65, 0.75)
# cell bg noise 
cell_bg_noise_1_std_range = (0.003, 0.22)
cell_bg_noise_2_std_range = (0.003, 0.01)
cell_bg_std_range = (0.04, 0.17)
#how much blob is multiplied with base
cell_blob_factor_range = (2,6)
#thickness of baseline mitosignal offset from base pixel width
mito_thickness_singal_size_range = (-3,-1)
# airy conv kernal size
airy_mito_kernal_size_range = (15,21)
# peri signal range choices
peri_singl_width_add_range_1 = (16,4)
peri_singl_width_add_range_2 = (18,6)
peri_singl_width_add_range_3 = (20,8)
# width to add to base morph of peri mito 
peri_morph_width_add_range = (2,7)
# kernal size used for blur on  peri mito
peri_morph_kernal_size_range = (11,20)
# photon conversion for peri glow
photon_conversion_rate_add_peri_glow_range = (50,100)
# photon conversion for peri mito signal 
photon_conversion_rate_add_peri_morph_range = (5,20)
# cell background scale multiplier factor range
cell_bg_scale_factor_range = (1.2,1.6)
# background glow noise range
background_glow_noise_range = (0.0001,0.04)
# mito segment dimmed signal rannge
mito_segment_consistent_dimness_range = (0.8, 1.2)
# interferance blob size
blob_size_add_range = (0,3)
# number of blobx
num_blobs_range = (5,50)
# variance in blob intensity 
temporal_blob_intensity_mult_range = (0.7,1.05)
# kernal size for peri signal
peri_signal_kernal_size_range = (70,100)

# peri glow chance
do_peri_glow_chance = 0.85
# peri glow signal values in final composition
peri_glow_clip_max_range = (0.21,0.41)
# peri morph signal values in final composition
peri_morph_clip_max_range = (0.33, 0.45)
# base mito signal values in final compisition
base_mito_clip_max_range = (0.3,0.46)
# mult factor for mito morph with other
mito_morph_clip_fraction_factor_range = (0.41,0.53)
# mult value for mito with others
max_total_single_range = (0.4,0.54)

# peri glow signal values in final composition for no glow samples
no_glow_peri_glow_clip_max = 0.002
# peri morph signal values in final composition for no glow samples
no_glow_peri_morph_clip_max = 0.002
# base mito signal values in final compisition for no glow samples
no_glow_base_mito_clip_max_range = (0.3,0.46)
# mult factor for mito morph with other for no glow samples
no_glow_mito_morph_clip_fraction_factor_range = (0.41,0.53)
# mult value for mito with others for no glow samples
no_glow_max_total_single_range = (0.12,0.23)


# ---------------------- derived vars ----------------------

# derived var of the simulation that is being rendered
sim_output_folder_path = os.path.join(sim_save_path, str(seed))

# sets randomnes seeds for reproducibility (optional)
if randomness_seed:
    np.random.seed(randomness_seed)
    random.seed(randomness_seed)
    torch.manual_seed(randomness_seed)
    cp.random.seed(randomness_seed)

# set the gpu number if their is more than 1 gpu


# get conv accelerator
if torch.cuda.is_available():
    device = torch.device('cuda:'+str(gpu_number))
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("using device:", device)

# if using cupy over torch for 4000+ series nvidia gpus (faster in some cases)
if use_cupy_over_torch:
    cp.cuda.Device(gpu_number).use()




# ---------------- helper functions ----------------

def rescale_in_range(in_image, nhigh, nlow):
    """
    Rescales the input image array to a specified range.

    Parameters:
    - in_image (numpy.ndarray): The input image array.
    - nhigh (float): The upper bound of the desired range.
    - nlow (float): The lower bound of the desired range.

    Returns:
    - scalled_arr (numpy.ndarray): The rescaled image array.

    """
    min_val = np.min(in_image)
    max_val = np.max(in_image)
    scalled_arr = nlow + ((in_image - min_val) * (nhigh - nlow) / (max_val - min_val))

    return scalled_arr

# gen blobs helper
def gen_blobs_helper(inshape, blob_val):
    out_im = ps.generators.blobs(shape=inshape, porosity=None, blobiness=[blob_val, blob_val]).astype(np.float32)
    return out_im

# dilates an image by the factor specified
def dilation_helper(in_arr, dilation_factor):
    return skimage.morphology.dilation(in_arr, skimage.morphology.disk(dilation_factor))

# generates a 2d gaussian kernel of a given size (size x size) and sigma, used for backbround to roughly represent the increased intensity around the peri region
def gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel.
    
    Parameters:
    - size: int, the size of the kernel (size x size). Should be an odd number.
    - sigma: float, the standard deviation of the Gaussian distribution.
    
    Returns:
    - A 2D numpy array representing the Gaussian kernel.
    """
    size = int(size) // 2
    x, y = np.mgrid[-size:size, -size:size]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


# NB: fft (CUPY) should give the same result, but in practice the approximation isnt perfect?. at least according to testing anyway
# both results are fit for purpose, but the fft version is faster in some cases
def conv_helper(input_matrix_np, kernel_np, device, use_cupy = False):
    
    # fft version that uses cupy-cuda, may be faster on ADA-Lovelace GPUs onwards, due to conv algorithim selection in underlying cuDNN firmware
    if use_cupy:
        
        # Transfer input matrix and kernel to the GPU
        input_matrix_gpu = cp.array(input_matrix_np)
        kernel_gpu = cp.array(kernel_np)
        
        # Get the shape of the input matrix and kernel
        input_shape = input_matrix_gpu.shape
        kernel_shape = kernel_gpu.shape

        # Calculate the shape for the FFT transform
        fft_shape = [input_shape[i] + kernel_shape[i] - 1 for i in range(len(input_shape))]

        # Perform FFT on both the input matrix and the kernel
        input_fft = cp.fft.fftn(input_matrix_gpu, fft_shape)
        kernel_fft = cp.fft.fftn(kernel_gpu, fft_shape)

        # Perform element-wise multiplication in the frequency domain
        result_fft = input_fft * kernel_fft

        # Perform the inverse FFT to get the convolution result
        result = cp.fft.ifftn(result_fft)

        # Extract the real part of the result
        result = cp.real(result)

        # Crop the result to the original input size
        start = [(fft_shape[i] - input_shape[i]) // 2 for i in range(len(input_shape))]
        end = [start[i] + input_shape[i] for i in range(len(input_shape))]
        slices = tuple(slice(start[i], end[i]) for i in range(len(input_shape)))
        result_cropped = result[slices]

        return cp.asnumpy(result_cropped)
    
    
    # basic torch version
    else:
        
        # does conv operation on gpu


        # get in array as torch tensor
        torch_in_arr = torch.from_numpy(input_matrix_np).float().unsqueeze(0).unsqueeze(0)

        # get kernal as torch tensor
        torch_kernal = torch.from_numpy(np.array(kernel_np)).float().unsqueeze(0).unsqueeze(0)

        # set the pad ammount based on the kernal shape
        kernal_shape = torch_kernal.shape
        padd_amount = np.floor(kernal_shape[3]/2).astype(np.uint8)


        torch_arr_gpu = torch_in_arr.to(device)
        kernal_gpu = torch_kernal.to(device)

        
        out_arr_cpu = F.conv2d(torch_arr_gpu, kernal_gpu, padding = padd_amount).to('cpu').numpy().squeeze() # Adjust padding as needed


        return out_arr_cpu

# used for reducing multiple intermidieate fission/fussion/matching steps into a single, by generating all combinations
def valid_edge_combinations(group_a, group_b):
    
    
    # Generate all possible unique edges
    all_edges = list(product(group_a, group_b))
    
    # Minimum and maximum number of edges needed
    min_edges = max(len(group_a), len(group_b))
    max_edges = len(all_edges)
    
    # Store valid combinations
    valid_combinations = []

    # Generate combinations of edges from min to max
    for r in range(min_edges, max_edges + 1):
        for combination in combinations(all_edges, r):
            nodes_a = set(edge[0] for edge in combination)
            nodes_b = set(edge[1] for edge in combination)
            # Check if this combination connects all nodes
            if len(nodes_a) == len(group_a) and len(nodes_b) == len(group_b):
                
                    #formated = [[x[0],x[1]] for x in combination]
                
                    valid_combinations.append(list(combination))
                    #print(combination)
                    #print('a')
                
    return valid_combinations


# rendering funcs


def get_non_leaf_nodes(graph):
    non_leaf_nodes = [node for node in graph.nodes() if graph.degree(node) > 1]
    return non_leaf_nodes

def get_leaf_nodes(graph):
    leaf_nodes = [node for node in graph.nodes() if graph.degree(node) == 1]
    return leaf_nodes

def get_highest_time_nodes(graph):
    node_list = list(graph.nodes())
    max_time = max([x[1] for x in node_list])
    return [x for x in node_list if x[1] == max_time]

def get_lowest_time_nodes(graph):
    node_list = list(graph.nodes())
    min_time = min([x[1] for x in node_list])
    return [x for x in node_list if x[1] == min_time]
    

# function that removes the node subset N from the graph nG
# it then adds an edge inplace of any edges that connected nG to N with an edge that connects the node in nG to a new node with id -1 at the same timestep

def modify_graph(nG, N):
    
    
    # Find all neighbors of nodes in N
    neighbors = set()
    for node in N:
        neighbors.update(nG.neighbors(node))
    
    # get a list of edges that connect nodes in N to nodes in nG
    edges_connecting_N_and_nG = [(u, v) for u, v in nG.edges() if (u in N and v not in N) or (v in N and u not in N)]

    # Remove nodes in N from the graph
    nG.remove_nodes_from(N)

    # for each edge connecting N and nG, add an edge in nG between the node in nG and a new node with id -1 at the same timestep t that the node in N had
    # eg, if (id1, ts1) in only nG was connected to (id2, ts1+1) in N, then add an edge between (id1, ts1) and (-1, ts1+1)
    for edge_pair in edges_connecting_N_and_nG:
        
        if edge_pair[0] in N:
            nG.add_edge((-1, edge_pair[0][1]), edge_pair[1])
            nG.nodes[(-1, edge_pair[0][1])]['class'] = 'void'
            
            print(((-1, edge_pair[0][1]), edge_pair[1]))
        else:
            nG.add_edge((-1, edge_pair[1][1]), edge_pair[0])
            nG.nodes[(-1, edge_pair[1][1])]['class'] = 'void'
            
            print(((-1, edge_pair[1][1]), edge_pair[0]))

    return nG


def get_render_only_nodes(graph):
    
    leaf_nodes = set(get_leaf_nodes(graph))
    highest_time_nodes = set(get_highest_time_nodes(graph))
    lowest_time_nodes = set(get_lowest_time_nodes(graph))
    
    half_node_set = leaf_nodes.union(highest_time_nodes).union(lowest_time_nodes)
    
    real_remove_set = set()
    
    for each in half_node_set:

        # chance for the start/end of disapear to be half rendered
        if random.uniform(0, 1) > 0.05:
            
            real_remove_set.add(each)
            
            if random.uniform(0, 1) > 0.5:
                
                for node in graph.neighbors(each):
                    real_remove_set.add(node)
            
    
    return(real_remove_set)



# functions for outputing the fission and fusion locations

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# finds the midpoint
def midpoint(point1, point2):
    x_m = (point1[0] + point2[0]) / 2
    y_m = (point1[1] + point2[1]) / 2
    return (x_m, y_m)

# given two graph mito objects (nx graphs of balls, with pos attribute), finds the closest pair of nodes between the two graphs
def find_closest_pairs(mito_ball_graph1, mito_ball_graph2):
    
    min_distance = float('inf')
    
    closest_pair = None
    closest_pair_pos = None
    
    for node1 in mito_ball_graph1.nodes():
        
        node1_loc = mito_ball_graph1.nodes[node1]['pos']
        
        
        for node2 in mito_ball_graph2.nodes():
            
            node2_loc = mito_ball_graph2.nodes[node2]['pos']
            
            distance = euclidean_distance(node1_loc, node2_loc)
            
            if distance < min_distance:
                min_distance = distance
                closest_pair = (node1, node2)
                closest_pair_pos = (node1_loc, node2_loc)
    
    return(min_distance, closest_pair, closest_pair_pos)

# get all combinations of indexes of x size at least y big
def generate_combinations(x, y):
    if y < 1 or y > x + 1:
        return []
    return list(combinations(range(x + 1), y))

# sorting func for the tuples
def sort_tuples_by_first_element(tuples_list):
    return sorted(tuples_list, key=lambda x: x[0])

# places a ashape at a specific point in a larger array
def place_shape(array, shape, center):
    array_shape = array.shape
    shape_shape = shape.shape
    
    # Calculate the start and end indices for each dimension
    start_indices = [center[i] - shape_shape[i] // 2 for i in range(3)]
    end_indices = [center[i] + shape_shape[i] // 2 + 1 for i in range(3)]
    
    # Ensure indices are within bounds of the array
    array_start_indices = [max(start_indices[i], 0) for i in range(3)]
    array_end_indices = [min(end_indices[i], array_shape[i]) for i in range(3)]
    
    # Calculate the corresponding start and end indices for the shape
    shape_start_indices = [max(0, -start_indices[i]) for i in range(3)]
    shape_end_indices = [shape_start_indices[i] + (array_end_indices[i] - array_start_indices[i]) for i in range(3)]
    
    # Iterate over the selected region and place the shape
    for i in range(array_start_indices[0], array_end_indices[0]):
        for j in range(array_start_indices[1], array_end_indices[1]):
            for k in range(array_start_indices[2], array_end_indices[2]):
                shape_i = i - array_start_indices[0] + shape_start_indices[0]
                shape_j = j - array_start_indices[1] + shape_start_indices[1]
                shape_k = k - array_start_indices[2] + shape_start_indices[2]
                
                if shape[shape_i, shape_j, shape_k] == 1:
                    array[i, j, k] += shape[shape_i, shape_j, shape_k]

    return array




def unet_weight_map(y, w0 = 10, sigma = 5):


    
    labels = skimage.measure.label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        

    else:
        w = np.zeros_like(y)
    
    return w.astype(np.float16)


def add_blobs_3d(array_shape, n, l, h, intensity_range = (0.2,1)):
    array = np.zeros(array_shape, dtype=float)
    for _ in range(n):
        blob_diameter = random.randint(l, h)
        blob_radius = blob_diameter // 2
        
        clipped_z_high = random.choice([0,1,2])
        clipped_z_low = random.choice([0,1,2])
        
        curr_blob_intensity = random.uniform(intensity_range[0], intensity_range[1])
        
        # Random position for the center of the blob
        center_x = random.randint(-blob_radius, array_shape[0] + blob_radius - 1)
        center_y = random.randint(-blob_radius, array_shape[1] + blob_radius - 1)
        center_z = random.randint(-blob_radius, array_shape[2] + blob_radius - 1)
        
        # Create a sphere of radius `blob_radius`
        for x in range(center_x - blob_radius, center_x + blob_radius + 1):
            for y in range(center_y - blob_radius, center_y + blob_radius + 1):
                for z in range(center_z - clipped_z_low, center_z + clipped_z_high + 1):
                    if (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= blob_radius**2:
                        # Check if the indices are within the bounds of the array
                        if 0 <= x < array_shape[0] and 0 <= y < array_shape[1] and 0 <= z < array_shape[2]:
                            array[x, y, z] += curr_blob_intensity
                            
    array = np.clip(array, 0, 1)

    
    return array


if __name__ == "__main__":



    # -------------- data loading and vars --------------

    # get file locs of all the pickles saved in the simulation
    fission = os.path.join(sim_output_folder_path, 'fission_output.pkl')
    fusion = os.path.join(sim_output_folder_path, 'fusion_output.pkl')
    pos = os.path.join(sim_output_folder_path, 'state_output.pkl')
    meta = os.path.join(sim_output_folder_path, 'metadata.pkl')
    outer_mem = os.path.join(sim_output_folder_path, 'outer_mem_loc.pkl')


    # create save directory
    save_folder_name = sim_output_folder_path.split('/')[-1]
    save_directory = os.path.join(out_folder, save_folder_name)
    create_directory_if_empty_or_not_exists(save_directory)




    # Load the data
    with open(fission, 'rb') as f:
        fission_list = pickle.load(f)
    with open(fusion, 'rb') as f:
        fusion_list = pickle.load(f)
    with open(pos, 'rb') as f:
        pos_dict = pickle.load(f)
    with open(meta, 'rb') as f:
        meta_dict = pickle.load(f)
    with open(outer_mem, 'rb') as f:
        outer_mem_locs_dict = pickle.load(f)

    # set specific metadata to variables for use in the simulation
    base_array_size = meta_dict['frame_size'] # 
    fps = meta_dict['fps'] # 
    vid_len = meta_dict['len_in_seconds'] # 
    base_mito_rad = meta_dict['base_mito_size'] # 
    first_frame = meta_dict['first_frame'] + fps
    frame_len = fps * vid_len
    hr_render_mito_size = base_mito_rad * render_mult - 1
    native_render_mito_size = base_mito_rad - 1


    # print info

    print(f'--- recording vars ---')
    print(f'vid_len                 =  {vid_len} total seconds')
    print(f'fps                     =  {fps} simulated fps')
    print(f'frame_len               =  {frame_len} simulated frames')
    print(f'base_mito_rad           =  {base_mito_rad} simulated size')
    print(f'first_frame             =  {first_frame}')
    print(f'base_array_size         =  {base_array_size}')
    print(f'hr_render_mito_size     =  {hr_render_mito_size}')
    print(f'native_render_mito_size =  {native_render_mito_size}')
    print(f'save_rate               =  {save_rate} frames in sim between saves')
    print(f'--- resulting vars ---')

    print(f'seconds between frames  =  {save_rate/fps}')
    print(f'total frames to save    =  {(fps/save_rate)*vid_len}')



    # ----------------- fission/fysion lookup structures -----------------

    # create lookup structures for the fission and fusion events, to be used in the frame generation, to lower the brightness of recenty fused or fissioned mito

    # lookup list with each entry being a list where each[0] is the frame number, and each[1] is the pair of mito ids stored as a set
    fusion_lookups = list()

    # list with each pair of mito ids stored as a set, for quick check if a fusion or fission event has occured ever between the two mito currently being drawn 
    fusion_tuple_list = list()

    for each in fusion_list:

        t_list = []
        t_list.append(set([each[1][4], each[1][5]]))
        t_list.append(each[0])
        fusion_lookups.append(t_list)
        
        fusion_tuple_list.append(set([each[1][5], each[1][4]]))

    for each in fission_list:

        t_list = []
        t_list.append(set([each[1][4], each[1][5]]))
        t_list.append(each[0])

        fusion_lookups.append(t_list)
        fusion_tuple_list.append(set([each[1][5], each[1][4]]))





    # --------------------- fix frame matchings --------------------------------


    saved_idx_list = list()
    for i in range(first_frame, first_frame + frame_len, save_rate):
        saved_idx_list.append(i)
        
    start_frame = min(saved_idx_list)
    end_frame = max(saved_idx_list)



    G = nx.Graph()
    time_step_dict = dict()


    class_dict = dict()


    # add all nodes for the selected timesteps to the graph
    for i in range (start_frame, end_frame+1):
        
        curr_idx = (0, i)
        
        t_list = list()
        
        t_class_dict = dict()
        
        for inst_id in pos_dict[curr_idx].keys():
            
            node_id = (inst_id, i)
            G.add_node(node_id)
            
            t_list.append(inst_id)
            
            
            
            t_class_dict[inst_id] = pos_dict[curr_idx][inst_id][1]
            
            
        time_step_dict[i] = t_list
        
        class_dict[i] = t_class_dict


        
    # add edgjes between the same node in adjacent timesteps
    for i in range(start_frame, end_frame):
        
        for inst_id in time_step_dict[i]:
            
            curr_node = (inst_id, i)
            
            if inst_id in time_step_dict[i+1]:
                
                next_node = (inst_id, i+1)
                
                G.add_edge(curr_node, next_node)
                
                


            
    # iterate over each timestep in the range of the fissions 
    for i in range(first_frame, first_frame + (end_frame - start_frame)):
        #print(i)
        

        fusions_at_this_timestep = list()
        fissions_at_this_timestep = list()

        temp_timestep_G = nx.Graph()


        for fusion_inst in fusion_list:
            
            if fusion_inst[0] == i:
                fusions_at_this_timestep.append(fusion_inst)
                
        for fission_inst in fission_list:
                
            if fission_inst[0] == i:
                fissions_at_this_timestep.append(fission_inst)
                
                
        step_unique_nodes_list = list()


        for fusion_inst in fusions_at_this_timestep:
            
            step_unique_nodes_list.append(fusion_inst[1][0])
            step_unique_nodes_list.append(fusion_inst[1][1])
            step_unique_nodes_list.append(fusion_inst[1][3])
            
            
        for fission_inst in fissions_at_this_timestep:
                
            step_unique_nodes_list.append(fission_inst[1][0])
            step_unique_nodes_list.append(fission_inst[1][2])
            step_unique_nodes_list.append(fission_inst[1][3])


        for node in step_unique_nodes_list:
            temp_timestep_G.add_node(node)
            

        for fusion_inst in fusions_at_this_timestep:
            
            temp_timestep_G.add_edge(fusion_inst[1][0], fusion_inst[1][3])
            temp_timestep_G.add_edge(fusion_inst[1][1], fusion_inst[1][3])

        for fission_inst in fissions_at_this_timestep:
            
            temp_timestep_G.add_edge(fission_inst[1][0], fission_inst[1][2])
            temp_timestep_G.add_edge(fission_inst[1][0], fission_inst[1][3])
            
            
            
        # get connected components
        temp_timestep_G_connected_components = list(nx.connected_components(temp_timestep_G))
        temp_timestep_G_subraph_list = [temp_timestep_G.subgraph(component).copy() for component in temp_timestep_G_connected_components]
        
        
        for subgraph in temp_timestep_G_subraph_list:
        
            subgraph_node_list = list(subgraph.nodes)
            
            nodes_low = [node for node in subgraph_node_list if node in time_step_dict[i]]
            
            nodes_high = [node for node in subgraph_node_list if node in time_step_dict[i+1]]
            
            for node_low in nodes_low:
                
                for node_high in nodes_high:
                    
                    parent = (node_low, i)
                    child = (node_high, i+1)
                            
                    G.add_edge(parent, child)




            
    # dict that will hold all out frame to outframe matchings
    final_tracking_dict = dict()


    # add the basic matchings where the id exists in both the start and end frame
    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        idx_low = i
        idx_high = i + save_rate
        

        frame_match_list = list()

        
        for inst_id in time_step_dict[idx_low]:
            
            
            if inst_id in time_step_dict[idx_high]:
                
                frame_match_list.append([inst_id, inst_id])
                
        final_tracking_dict[idx_low] = frame_match_list
                
                
    # get subgraphs of nodes that have fissions and fussions between time steps

    ff_node_graphs = dict()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        idx_low = i
        idx_high = i + save_rate
        
        step_node_list = list()

        # list of timesteps between the two points
        sim_idx_list = list(range(idx_low,idx_high))
        
        # print(f'idx_low: {idx_low}')
        # print(f'idx_high: {idx_high}')
        
        changed_nodes = list()
        n1 = [x for x in fusion_list if x[0] in sim_idx_list]
        for each in n1:
            changed_nodes.append(each[1][0])
            changed_nodes.append(each[1][1])
            changed_nodes.append(each[1][3])


        n2 = [x for x in fission_list if x[0] in sim_idx_list]
        for each in n2:
            changed_nodes.append(each[1][0])
            changed_nodes.append(each[1][2])
            changed_nodes.append(each[1][3])
            
        #print(changed_nodes)
        
        
        for timestep in range(idx_low, idx_high+1):
            
            for node_id in time_step_dict[timestep]:
                
                if node_id in changed_nodes:
                    
                    t_node = (node_id, timestep)
                    step_node_list.append(t_node)
                    
        ff_node_graphs[i] = step_node_list
        
        
        
    solved_connections_dict = dict()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):

        
        idx_low = i
        idx_high = i + save_rate
        
        #print(idx_low, idx_high)
        
        
        current_ff_node_step_list = ff_node_graphs[idx_low] 
        current_ff_graph = G.subgraph(current_ff_node_step_list)
        #print(current_ff_graph.nodes)
        
        
        current_conneceted_components = list(nx.connected_components(current_ff_graph))
        current_subgraphs_list = [G.subgraph(component).copy() for component in current_conneceted_components]
        
        
        #print('---')
        
        step_solutions_list = list()
        
        
        for current_subgraph in current_subgraphs_list:
            #print(current_subgraph)
            
            #print(current_subgraph.nodes())
            
            
            low_node_list = [x for x in current_subgraph.nodes() if x[1] == idx_low]
            #print('low nodes', low_node_list)

            high_node_list = [x for x in current_subgraph.nodes() if x[1] == idx_high]      
            #print('high nodes',high_node_list)
            
            
            frame_idx_low = (0, idx_low)
            frame_idx_high = (0, idx_high) 
            # graph list
            graph_list_low = pos_dict[frame_idx_low]
            graph_list_high = pos_dict[frame_idx_high]
            
            
            
            low_node_dict = dict()
            high_node_dict = dict()

            for node_idx in low_node_list:
                
                node_id = node_idx[0]
                
                low_node_dict[node_id] = list(graph_list_low[node_id][0].nodes())

            for node_idx in high_node_list:
                
                node_id = node_idx[0]
                
                high_node_dict[node_id] = list(graph_list_high[node_id][0].nodes())
            
            #print('low node balls dict', low_node_dict)
            #print('high node balls dict', high_node_dict)
            
            # possible matchings
            list_of_candidate_edge_lists = valid_edge_combinations(list(low_node_dict.keys()), list(high_node_dict.keys()))
            
            # no complex fusion solutions
            list_of_valid_candidate_edge_lists = list()
            
            # all nodes used
            list_of_nodes_in_matching = list(low_node_dict.keys())+ list(high_node_dict.keys())

            # subset to non complex fusion solutions
            for candidate_edge_list in list_of_candidate_edge_lists:
                # print(candidate_edge_list)
                # print('--')
                
                t_G = nx.Graph()
                for t_node in list_of_nodes_in_matching:
                    t_G.add_node(t_node)
                    
                for edge in candidate_edge_list:
                    t_G.add_edge(edge[0], edge[1])
                    
                
                # nx.draw(t_G, with_labels=True, font_size = 8, node_color = 'skyblue')
                # plt.show()
                
                
                t_G_connected_components = list(nx.connected_components(t_G))
                t_G_subgraphs_list = [t_G.subgraph(component).copy() for component in t_G_connected_components]
                
                complex_fusion_flag = False
                
                for subgraph_connected_component in t_G_subgraphs_list:

                    
                    
                    low_node_list = [x for x in subgraph_connected_component.nodes() if x in low_node_dict.keys()]
                    high_node_list = [x for x in subgraph_connected_component.nodes() if x in high_node_dict.keys()]


                    degree_list_low = list()
                    for t_node_i in low_node_list:
                        degree_list_low.append(subgraph_connected_component.degree(t_node_i))
                    
                    
                    degree_list_high = list()
                    for t_node_i in high_node_list:
                        degree_list_high.append(subgraph_connected_component.degree(t_node_i))


                    
                    
                    if max(degree_list_low) > 1 and max(degree_list_high) > 1:
                        complex_fusion_flag = True
                        
                        
                if complex_fusion_flag == False:
                    list_of_valid_candidate_edge_lists.append(candidate_edge_list)

                            
            #print(list_of_valid_candidate_edge_lists)


            best_score = 0
            best_score_edge_set = None


            for candidate_edges in list_of_valid_candidate_edge_lists:
                
                score = 0
                for edge in candidate_edges:
                    
                    
                    low_node_list = low_node_dict[edge[0]]
                    high_node_list = high_node_dict[edge[1]]
                    
                    current_edge_match_ammount = len(list(set(low_node_list) & set(low_node_list)))
                    
                    score += current_edge_match_ammount
                    

                if score > best_score:
                    best_score = score
                    best_score_edge_set = candidate_edges
                
                elif score == best_score:
                    if len(candidate_edges) < len(best_score_edge_set):
                        best_score = score
                        best_score_edge_set = candidate_edges
                
                else:
                    pass
            
                    
            #print(best_score)       
            #print(best_score_edge_set)
            
            step_solutions_list.append(best_score_edge_set)
            
            
        if len(step_solutions_list) != 0:
            
            solved_connections_dict[i] = step_solutions_list

            

    for timestep, solution_list in solved_connections_dict.items():

        
        #print(timestep)
        for group_of_edges in solution_list:
            
            for edge in group_of_edges:
                #print(edge)
                
                
                final_tracking_dict[timestep].append([edge[0], edge[1]])




    for recorded_time in final_tracking_dict.keys():
        timestep_dict_low = set(time_step_dict[recorded_time])
        timestep_dict_high = set(time_step_dict[recorded_time + save_rate])
        
        
        
        matched_pairs = final_tracking_dict[recorded_time]
        matched_pairs_low = set([x[0] for x in matched_pairs])
        matched_pairs_high = set([x[1] for x in matched_pairs])
        
        low_dif = timestep_dict_low.symmetric_difference(matched_pairs_low)
        high_dif = timestep_dict_high.symmetric_difference(matched_pairs_high)
        if len(low_dif) > 0:
            print('low_dif', low_dif)
            print('CRITICAL ERROR')
        if len(high_dif) > 0:
            print('high_dif', high_dif)
            print('CRITICAL ERROR')



    # --- make some mito invis ----------------------





    # this creates a graph of all the matchings that are in the base simulation BUT are only between the frames that the renderer will be rendering.
    # this includes nodes in the last frame obviously. 

    nG = nx.Graph()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        #print(i)
        
        
        for each in final_tracking_dict[i]:
            
            node_name = (each[0], i)
            #print(node_name)
            
            nG.add_node(node_name)

    last_frame = first_frame + frame_len - save_rate

    for each in final_tracking_dict[last_frame-save_rate]:
        
        
        node_name = (each[1], last_frame)
        nG.add_node(node_name)

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):

        for each in final_tracking_dict[i]:
            
            node_low = (each[0], i)
            node_high = (each[1], i+save_rate)
            
            nG.add_edge(node_low, node_high)



    # add the classes from class_dict as properties to nodes in the nG graph
    for node_id in nG.nodes():
        
        nG.nodes[node_id]['class'] = class_dict[node_id[1]][node_id[0]]



    # vars for the remove loop





    # num_drop sets
    num_drop_sets = random.randint(drop_set_range[0], drop_set_range[1])
    #num_drop_sets = 50


    print('num sets made invisible:', num_drop_sets)

    # patience counter for trying to find new sets to drop
    set_drop_patience = 1000

    # storage for the node sets to remove
    subset_drop_sets = list()
    render_only_nodes = list()




    drop_set_counter = 0
    patience_counter = 0

    while drop_set_counter < num_drop_sets and patience_counter <= set_drop_patience:
        
        
        # check if the subset is valid 
        flag_valid_subset = True
        
        
        # current set drop number
        print('drop set counter:', drop_set_counter)
        
        # get the size of the current drop set
        current_drop_size = random.randint(size_range[0], size_range[1])
        
        # get a seed node
        seed_node = random.choice(list(nG.nodes()))

        # create a set to store the nodes and add the seed node
        subset_nodes = {seed_node}
        
        
        fail_flag = False

        while len(subset_nodes) < current_drop_size:

            t_neighbors = list()
                
            for each in subset_nodes:
                
                for aeach in list(nG.neighbors(each)):
                    if aeach not in subset_nodes and aeach not in t_neighbors:
                        t_neighbors.append(aeach)
            
            if len(t_neighbors) == 0:
                fail_flag = True
                break
            
            subset_nodes.add(random.choice(t_neighbors))

            for each in subset_nodes:
                if nG.degree(each) == 1:
                    if random.uniform(0,1) < 0.5:
                        current_drop_size -= 1
                        #print('shortended due to butting against start of crop')
            
        if fail_flag:
            patience_counter += 1
            flag_valid_subset = False
            continue
        
        
            
            
        # create a subset of the graph
        nG_subset = nG.subgraph(subset_nodes).copy()

        
        
        # check if any of the nodes in the subset have the class void or peri
        for node in nG_subset.nodes():
            
            if nG_subset.nodes[node]['class'] == 'void' or nG_subset.nodes[node]['class'] == 'peri':
                flag_valid_subset = False
                break
        
        
        # check if the neighbors of any of the nodes in the subset have the class void 
        for node in nG_subset.nodes():
            
            neighbors = list(nG.neighbors(node))
            
            for neighbor in neighbors:
                
                if nG.nodes[neighbor]['class'] == 'void':
                    #print('neighbor with void')
                    flag_valid_subset = False
                    break
                
                
            if flag_valid_subset == False:
                break
                
                
        # if the subset is valid, remove the nodes from the graph
        if flag_valid_subset:
            
            nG = modify_graph(nG, subset_nodes)
            
            subset_drop_sets.append(subset_nodes)
            
            drop_set_counter += 1
            patience_counter = 0
            
            
            
            render_only_nodes_for_itr = get_render_only_nodes(nG_subset)
            render_only_nodes.append(render_only_nodes_for_itr)
            
            
            
        else:
            patience_counter += 1
        
        
        if patience_counter >= set_drop_patience:
            #print('patience limit reached')
            break





    if len(subset_drop_sets) > 0:
        intersection = set.intersection(*subset_drop_sets)
        
        if len(intersection) > 0:
            print('MAJOR LOGIC ERROR IN SET REMOVER')
            print(intersection)
        
        full_subset_drop_list = set.union(*subset_drop_sets)
        
        full_render_only_list = set.union(*render_only_nodes)
        
        
        full_render_only_list = {node for node in full_render_only_list if node[1] != first_frame and node[1] != last_frame}
        full_render_only_list = {node for node in full_render_only_list if node[1] != first_frame + save_rate and node[1] != last_frame - save_rate}

        
    else:
        full_subset_drop_list = set()
        full_render_only_list = set()
        
        print('no sets to remove')
        
        
        
        
        

    # new tracking dict with the removed nodes in the same form as the original matching dict
    redone_out_matching_dict = dict()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        # print(i)
        # print(i+save_rate)
        
        low_set_nodes = [x for x in nG.nodes if x[1] == i]
        high_set_nodes = [x for x in nG.nodes if x[1] == i+save_rate]
        
        full_node_set = set(low_set_nodes + high_set_nodes)
        
        t_supgraph = nG.subgraph(full_node_set).copy()
        
        t_dict_list = list()
        for node_pair in t_supgraph.edges():
            print(node_pair)
            lnodex = [x for x in node_pair if x[1] == i][0]
            lnodey = [x for x in node_pair if x[1] == i+save_rate][0]
            

            t_dict_list.append([lnodex[0], lnodey[0]])
            
        redone_out_matching_dict[i] = t_dict_list
        
        
        


    for each in redone_out_matching_dict.keys():

        with_low_invis = [x for x in redone_out_matching_dict[each] if x[0] == -1]
        
        
        for aeach in with_low_invis:
            #print(each)
            matching = aeach[1]
            
            same_frame_merges = [x for x in redone_out_matching_dict[each] if x[1]==matching and x[0] != -1]
            
            if len(same_frame_merges) >= 1:
                print(each)
                print(same_frame_merges)
                print(aeach)
                
                redone_out_matching_dict[each].remove(aeach)
                
                
                



        with_high_invis = [x for x in redone_out_matching_dict[each] if x[1] == -1]
        
        for aeach in with_high_invis:
            
            matching = aeach[0]
            
            same_frame_merges = [x for x in redone_out_matching_dict[each] if x[0]==matching and x[1] != -1]
            
            if len(same_frame_merges) >= 1:
                print(each)
                print(same_frame_merges)
                print(aeach)
                
                redone_out_matching_dict[each].remove(aeach)
                
                
                
    # new class dict with the removed nodes in the same form as the original class dict
    redone_out_class_dict = dict()

    for i in range(first_frame, first_frame + frame_len, save_rate):
        
        t_class_dict = dict()
        
        for node in nG.nodes():
            if node[1] == i:
                # print(node)
                # print(nG.nodes[node]['class'])
                
                t_class_dict[node[0]] = nG.nodes[node]['class']
        
        redone_out_class_dict[i] = t_class_dict
        
        
        
        
        
    # re index to the frames

    j = 0

    final_tracking_dict_reidx = dict()
    class_dict_subset_and_re_idx = dict()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        
        final_tracking_dict_reidx[j] = redone_out_matching_dict[i]
        class_dict_subset_and_re_idx[j] = redone_out_class_dict[i]
        
        #print(i,j)
        j+=1
        
        
    class_dict_subset_and_re_idx[j] = redone_out_class_dict[first_frame + frame_len - save_rate]
                
                
    # -------------- position renderer ----------------
    
    
    # create a graph based off the final outputs of the tracking post making some mito invisible
    newnew = nx.Graph()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        
        
        for each in redone_out_matching_dict[i]:
            
            node_name = (each[0], i)
            
            newnew.add_node(node_name)

    last_frame = first_frame + frame_len - save_rate

    for each in redone_out_matching_dict[last_frame-save_rate]:
        
        
        node_name = (each[1], last_frame)
        newnew.add_node(node_name)

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):

        for each in redone_out_matching_dict[i]:
            
            node_low = (each[0], i)
            node_high = (each[1], i+save_rate)
            
            newnew.add_edge(node_low, node_high)



    # get the fission and fusion locations
    fission_dict = dict()
    fusion_dict = dict()

    for i in range(first_frame, first_frame + frame_len - save_rate, save_rate):

        
        # get the subset of nodes in the current and next timestep
        node_subset = [x for x in newnew.nodes if x[1] == i+save_rate or x[1] == i]

        
        t_subgraph = newnew.subgraph(node_subset).copy()
        
        t_fission_list = list()
        t_fusion_list = list()
        
        
        for node in t_subgraph.nodes():
            
            # check if the node is in in i and has degree >1
            if node[1] == i and t_subgraph.degree(node) > 1:
                neighbors = list(t_subgraph.neighbors(node))
                
                xlet = [node, neighbors]
                t_fission_list.append(xlet)

                    
            if node[1] == i+save_rate and t_subgraph.degree(node) > 1:
                neighbors = list(t_subgraph.neighbors(node))
                
                xlet = [node, neighbors]
                t_fusion_list.append(xlet)
                
        fission_dict[i] = t_fission_list
        fusion_dict[i] = t_fusion_list
        


    # get the locations of the fissions and fusions and add them to a dict
    fusion_frame_and_loc_dict = dict()
    fission_frame_and_loc_dict = dict()

    for curr_frame in range(first_frame, first_frame + frame_len, save_rate):
        fusion_frame_and_loc_dict[curr_frame] = []
        fission_frame_and_loc_dict[curr_frame] = []



    for curr_frame in range(first_frame, first_frame + frame_len - save_rate, save_rate):
        print(curr_frame)

        high_frame = curr_frame + save_rate



        fission_xlet_list = fission_dict[curr_frame]
        for xlet in fission_xlet_list:
            
            print(xlet)
            
            combined_mito = xlet[0]
            split_mito_list = xlet[1]
            split_mito_count = len(split_mito_list)
            split_points = split_mito_count -1
            
            split_frame_idx = (0, xlet[1][0][1])
            split_frame = xlet[1][0][1]
            
            print('split points:', split_points)
            print('split mito count:', split_mito_count)
            print('combined mito:', combined_mito)
            print('split mito:', split_mito_list)
            print('split_frame_idx:', split_frame_idx)
            
            split_mito_ids = [x[0] for x in xlet[1]]

            
            if split_points == 1:
            
                print('split mito ids:', split_mito_ids)
                split_mito_network_graphs = [pos_dict[split_frame_idx][x][0] for x in split_mito_ids]
                
                closest_pairs = find_closest_pairs(split_mito_network_graphs[0], split_mito_network_graphs[1])
                print('closest_pairs:', closest_pairs)
                
                location_of_split = midpoint(closest_pairs[2][0], closest_pairs[2][1])
                
                print('location_of_split:', location_of_split)
                
                split_tuple = (xlet, location_of_split)
                
                fission_frame_and_loc_dict[split_frame].append(split_tuple)
                
            else:
                
                split_mito_ids = [x[0] for x in xlet[1]]
                
                comb_split_tuple_list = list()
                
                for mito_idx_1, mito_idx_2 in generate_combinations(split_mito_count -1, 2):
                    print(mito_idx_1, mito_idx_2)
                    
                    testing_split_mito_ids = [split_mito_ids[mito_idx_1], split_mito_ids[mito_idx_2]]
                    
                    split_mito_network_graphs = [pos_dict[split_frame_idx][x][0] for x in testing_split_mito_ids]

                    closest_pairs = find_closest_pairs(split_mito_network_graphs[0], split_mito_network_graphs[1])

                    location_of_split = midpoint(closest_pairs[2][0], closest_pairs[2][1])
                    
                    split_tuple = (xlet, location_of_split)

                    comb_split_tuple_list.append((closest_pairs[0], split_tuple))
                    
                chosen_split_tuples = sort_tuples_by_first_element(comb_split_tuple_list)[0:split_points]
                
                for each in chosen_split_tuples:
                    fission_frame_and_loc_dict[split_frame].append(each[1])
                    
                    
                    
        fusion_xlet_list = fusion_dict[curr_frame]
        for xlet in fusion_xlet_list:
            
            print(xlet)
            
            combined_mito = xlet[0]
            split_mito_list = xlet[1]
            split_mito_count = len(split_mito_list)
            split_points = split_mito_count -1
            
            split_frame_idx = (0, xlet[1][0][1])
            split_frame = xlet[1][0][1]

            print('split points:', split_points)
            print('split mito count:', split_mito_count)
            print('combined mito:', combined_mito)
            print('split mito:', split_mito_list)
            print('split_frame_idx:', split_frame_idx)
            
            split_mito_ids = [x[0] for x in xlet[1]]

            
            if split_points == 1:
            
                print('split mito ids:', split_mito_ids)
                split_mito_network_graphs = [pos_dict[split_frame_idx][x][0] for x in split_mito_ids]
                
                closest_pairs = find_closest_pairs(split_mito_network_graphs[0], split_mito_network_graphs[1])
                print('closest_pairs:', closest_pairs)
                
                location_of_split = midpoint(closest_pairs[2][0], closest_pairs[2][1])
                
                print('location_of_split:', location_of_split)
                
                split_tuple = (xlet, location_of_split)
                
                fusion_frame_and_loc_dict[split_frame].append(split_tuple)
                
            else:
                
                split_mito_ids = [x[0] for x in xlet[1]]
                
                comb_split_tuple_list = list()
                
                for mito_idx_1, mito_idx_2 in generate_combinations(split_mito_count -1, 2):
                    print(mito_idx_1, mito_idx_2)
                    
                    testing_split_mito_ids = [split_mito_ids[mito_idx_1], split_mito_ids[mito_idx_2]]
                    
                    split_mito_network_graphs = [pos_dict[split_frame_idx][x][0] for x in testing_split_mito_ids]

                    closest_pairs = find_closest_pairs(split_mito_network_graphs[0], split_mito_network_graphs[1])

                    location_of_split = midpoint(closest_pairs[2][0], closest_pairs[2][1])
                    
                    split_tuple = (xlet, location_of_split)

                    comb_split_tuple_list.append((closest_pairs[0], split_tuple))
                    
                chosen_split_tuples = sort_tuples_by_first_element(comb_split_tuple_list)[0:split_points]
                
                for each in chosen_split_tuples:
                    fusion_frame_and_loc_dict[split_frame].append(each[1])


    # create the label shape
    shape_dict = dict()
    shape_dict['cube_3'] = np.ones((3,3,3))
    shape_dict['cube_5'] = np.ones((5,5,5))
    shape_dict['square_3'] = np.ones((3,3))
    shape_dict['square_5'] = np.ones((5,5))
    shape_dict['ball_5'] = skimage.morphology.ball(2)
    shape_dict['circle_5'] = skimage.morphology.disk(2)

    label_point_shape = shape_dict[shape_choice]


    # add the shape to the fission and fusion arrays
    
    out_shape = (int(base_array_size[0] * save_mult), int(base_array_size[1] * save_mult))
    print(out_shape)
    fission_array = np.zeros((len(fission_frame_and_loc_dict.keys()), out_shape[0], out_shape[1]), dtype=np.uint8)
    fusion_array = np.zeros((len(fusion_frame_and_loc_dict.keys()), out_shape[0], out_shape[1]), dtype=np.uint8)

    print(fission_array.shape)

    zcounter = 0
    for i in fission_frame_and_loc_dict.keys():
        print(i)
        print(zcounter)
        curr_frame_fission = fission_frame_and_loc_dict[i]
        for fission_event in curr_frame_fission:
            print(fission_event)
            print(fission_event[1][0])
            
            fission_event_loc = (zcounter, int(fission_event[1][0]*save_mult), int(fission_event[1][1]*save_mult))
            print(fission_event_loc)
            
            fission_array = place_shape(fission_array, label_point_shape, fission_event_loc)
            
            
        curr_frame_fusion = fusion_frame_and_loc_dict[i]
        for fusion_event in curr_frame_fusion:
            print(fusion_event)
            print(fusion_event[1][0])
            
            fusion_event_loc = (zcounter, int(fusion_event[1][0]*save_mult), int(fusion_event[1][1]*save_mult))
            print(fusion_event_loc)
            
            fusion_array = place_shape(fusion_array, label_point_shape, fusion_event_loc)
            
        
        zcounter +=1
    
    combine_ff_array = fission_array + fusion_array * 2
    
    

    
    def random_odd_between(a, b):
        # Ensure a is odd if it’s even
        if a % 2 == 0:
            a += 1
        # Ensure b is odd if it’s even
        if b % 2 == 0:
            b -= 1
        if a > b:
            raise ValueError("No odd numbers in this range.")
        return random.randrange(a, b + 1, 2)
    
    photon_conversion_rate_mito = random.randint(photon_conversion_range[0], photon_conversion_range[1])
    cell_bg_blobiness_add = random.uniform(cell_bg_blobiness_parameter_range[0], cell_bg_blobiness_parameter_range[1])
    bg_blob_impact_factor = random.uniform(bg_blob_impact_factor_range[0], bg_blob_impact_factor_range[1])
    bg_noise_std = random.choice([random.uniform(cell_bg_noise_1_std_range[0], cell_bg_noise_1_std_range[1]), random.uniform(cell_bg_noise_2_std_range[0], cell_bg_noise_2_std_range[1])])
    bg_noise_mean = random.uniform(cell_bg_std_range[0], cell_bg_std_range[1])
    cell_blob_factor = random.randint(cell_blob_factor_range[0],cell_blob_factor_range[1])
    mito_base_morph_size_add = random.randint(mito_thickness_singal_size_range[0], mito_thickness_singal_size_range[1])
    airy_mito_kernal_size = int(random_odd_between(airy_mito_kernal_size_range[0], airy_mito_kernal_size_range[1]))
    peri_signal_width_add = random.choice([peri_singl_width_add_range_1, peri_singl_width_add_range_2, peri_singl_width_add_range_3])
    peri_moprh_width_add = random.randint(peri_morph_width_add_range[0], peri_morph_width_add_range[1])
    peri_morph_kernal_size = random.randint(peri_morph_kernal_size_range[0],peri_morph_kernal_size_range[1])
    photon_conversion_rate_add_peri_glow =  random.randint(photon_conversion_rate_add_peri_glow_range[0],photon_conversion_rate_add_peri_glow_range[1])
    photon_conversion_rate_add_peri_morph = random.randint(photon_conversion_rate_add_peri_morph_range[0],photon_conversion_rate_add_peri_morph_range[1])
    cell_bg_scale_factor = random.uniform(cell_bg_scale_factor_range[0],cell_bg_scale_factor_range[1])
    raw_bg_glow_no_noise_mult = random.choice([background_glow_noise_range[0], random.uniform(background_glow_noise_range[0],background_glow_noise_range[1])])
    mito_segment_consistent_dimness_range = (mito_segment_consistent_dimness_range[0], mito_segment_consistent_dimness_range[1])
    blob_size_add = random.randint(blob_size_add_range[0], blob_size_add_range[1])
    num_blobs = random.randint(num_blobs_range[0],num_blobs_range[1])
    temporal_blob_intensity_mult = random.uniform(temporal_blob_intensity_mult_range[0], temporal_blob_intensity_mult_range[1])
    peri_signal_kernal_size = random.randint(peri_signal_kernal_size_range[0], peri_signal_kernal_size_range[1])

    do_peri_glow = True
    do_peri_glow_chance_seed = random.uniform(0,1)
    if do_peri_glow_chance_seed > do_peri_glow_chance:
        do_peri_glow = False
    
    #do_peri_glow = random.choice([True, True, True, True, True, True, True, False])



    if do_peri_glow:
        
        peri_glow_max = random.uniform(peri_glow_clip_max_range[0],peri_glow_clip_max_range[1])
        peri_morph_max = random.uniform(peri_morph_clip_max_range[0],peri_morph_clip_max_range[1])

        base_mito_morph_max = random.uniform(base_mito_clip_max_range[0],base_mito_clip_max_range[1])
        clip_fraction_factor = random.uniform(mito_morph_clip_fraction_factor_range[0],mito_morph_clip_fraction_factor_range[1])

        max_signal_from_peri_and_mito = random.uniform(max_total_single_range[0],max_total_single_range[1])
        
    else:
        
        peri_glow_max = no_glow_peri_glow_clip_max
        peri_morph_max = no_glow_peri_morph_clip_max
        
        base_mito_morph_max = random.uniform(no_glow_base_mito_clip_max_range[0],no_glow_base_mito_clip_max_range[1])
        clip_fraction_factor = random.uniform(no_glow_mito_morph_clip_fraction_factor_range[0],no_glow_mito_morph_clip_fraction_factor_range[1])
        
        max_signal_from_peri_and_mito = random.uniform(no_glow_max_total_single_range[0],no_glow_max_total_single_range[1])

    
    
    
    
    
    
    
    param_dict = dict()

    param_dict['photon_conversion_rate_mito'] = photon_conversion_rate_mito
    param_dict['cell_bg_blobiness_add'] = cell_bg_blobiness_add
    param_dict['bg_blob_impact_factor'] = bg_blob_impact_factor
    param_dict['bg_noise_std'] = bg_noise_std
    param_dict['bg_noise_mean'] = bg_noise_mean
    param_dict['cell_blob_factor'] = cell_blob_factor
    param_dict['mito_base_morph_size_add'] = mito_base_morph_size_add
    param_dict['airy_mito_kernal_size'] = airy_mito_kernal_size
    param_dict['peri_signal_width_add'] = peri_signal_width_add
    param_dict['peri_moprh_width_add'] = peri_moprh_width_add
    param_dict['peri_morph_kernal_size'] = peri_morph_kernal_size
    param_dict['photon_conversion_rate_add_peri_glow'] = photon_conversion_rate_add_peri_glow
    param_dict['photon_conversion_rate_add_peri_morph'] = photon_conversion_rate_add_peri_morph
    param_dict['base_mito_morph_max'] = base_mito_morph_max
    param_dict['peri_glow_max'] = peri_glow_max
    param_dict['peri_morph_max'] = peri_morph_max
    param_dict['clip_fraction_factor'] = clip_fraction_factor
    param_dict['max_signal_from_peri_and_mito'] = max_signal_from_peri_and_mito
    param_dict['cell_bg_scale_factor'] = cell_bg_scale_factor
    param_dict['raw_bg_glow_no_noise_mult'] = raw_bg_glow_no_noise_mult
    param_dict['mito_segment_consistent_dimness_range'] = mito_segment_consistent_dimness_range
    param_dict['blob_size_add'] = blob_size_add
    param_dict['num_blobs'] = num_blobs
    param_dict['temporal_blob_intensity_mult'] = temporal_blob_intensity_mult

    
    
    
    
    # ------------------------- dim some mitochondria -------------------------
    
    
    frame_idx_choice_list = [x for x in range(first_frame, first_frame + frame_len, save_rate)]

    choice_dim_frame = random.choice(frame_idx_choice_list)


    choice_frame_dict = pos_dict[(0, choice_dim_frame)]

    dimmer_dict = dict()

    for node in choice_frame_dict.keys():
        
        brightness_choice = random.uniform(mito_segment_consistent_dimness_range[0], mito_segment_consistent_dimness_range[1])
        
        curr_graph = choice_frame_dict[node][0]

        for node in curr_graph.nodes():
            dimmer_dict[node] = brightness_choice
    
    
    # -------------- temporal blobs ---------------------
    

    native_dims_for_out = (int((fps/save_rate)*vid_len), render_mult* base_array_size[0], render_mult* base_array_size[1])

    print(native_dims_for_out)

    temporal_blob_array = add_blobs_3d(native_dims_for_out, num_blobs, 15, 19+blob_size_add)
    temporal_blob_dict = dict()
    temporal_blob_idx_counter = 0


    for i in range(first_frame, first_frame + frame_len, save_rate):
        
        temporal_blob_dict[i] = temporal_blob_array[temporal_blob_idx_counter,:,:]
        
        temporal_blob_idx_counter += 1
        
    


    
    
    
    
    # -------------- background render

    # get a single frame from halfway through the video to generate the outer membrane layer of the render
    # this is static as those outer membrane locations are essentially static, and its a costly operation to do for every frame
    outer_mem_frame = int(first_frame + 8 * vid_len/2)

    # get point list of outermembrane, scale size by factor `shrink_factor`
    point_list = outer_mem_locs_dict[outer_mem_frame]
    # shrink_factor = 1
    # point_list= [  (   (((i[0]-base_array_size[0]/2)*shrink_factor) + base_array_size[0]/2)  , (((i[1]-base_array_size[1]/2)*shrink_factor) + base_array_size[1]/2)       ) for i in point_list_base]
    hr_shape = (base_array_size[0] * render_mult, base_array_size[1] * render_mult)
    cell_highlight_bg = np.zeros(hr_shape).astype(np.float32)

    for i in range(len(point_list)-1):
            pos1 = point_list[i]
            pos2 = point_list[(i+1) % (len(point_list)-1)]


            rr, cc = skimage.draw.line(int(np.round(pos1[0]*render_mult)), int(np.round(pos1[1]*render_mult)), int(np.round(pos2[0]*render_mult)), int(np.round(pos2[1]*render_mult)))

            rr = np.clip(rr, 0, (base_array_size[0]*render_mult-1))
            cc = np.clip(cc, 0, (base_array_size[1]*render_mult-1))


            cell_highlight_bg[rr, cc] = 1


    # Order of operations
    # 1. Dilation size 2
    # 2. flood fill
    # 3. conv with big gaussian kernal
    # 4. Mult with fsize_gauss and rescale to 0,1
    # 5. mult with big blobs (big blobs are scaled 0.9 1.1)
    # 6. conv again with another gaussian kernal and rescale to 0,1
    # 7. rescale to 0, 1.2 and clip to 0,1 to blow out the highlights in the central region

    # 1
    cell_highligh_bg_thick = skimage.morphology.dilation(cell_highlight_bg, skimage.morphology.disk(2))

    # 2
    cell_highligh_bg_thick_filled = skimage.morphology.flood_fill(cell_highligh_bg_thick, base_array_size, 1)

    # 3
    big_gaussian_kernal = Gaussian2DKernel(x_stddev = 60, y_stddev = 60, theta = 0)


    big_gaussian_kernal.shape
    cell_highligh_bg_thick_filled.shape

    start = time.time()
    #print("hello")

    conved_cell_highlight_bg =  conv_helper(cell_highligh_bg_thick_filled, big_gaussian_kernal, device, use_cupy_over_torch)

    end = time.time()
    print(end - start)


    print('post conved shapes match:', conved_cell_highlight_bg.shape == cell_highligh_bg_thick_filled.shape)

    # 4
    fsize_gauss = gaussian_kernel(size = hr_shape[0], sigma=cell_bg_glow_central_sigma)
    fsize_gauss_rescale = rescale_in_range(fsize_gauss, 2, 0)
    conved_cell_highlight_bg_gauss_mult = conved_cell_highlight_bg * fsize_gauss_rescale + conved_cell_highlight_bg*0.3
    conved_cell_highlight_bg_gauss_mult_rescaled = rescale_in_range(conved_cell_highlight_bg_gauss_mult, 1, 0)


    # 5
    cell_bg_por_blobs = gen_blobs_helper(hr_shape, cell_blob_factor).astype(np.float32)
    cell_bg_por_blobs_rescaled = rescale_in_range(cell_bg_por_blobs, 1 + cell_bg_blobiness_add, 1 - cell_bg_blobiness_add)
    conved_cell_highlight_bg_bigblobbed = conved_cell_highlight_bg_gauss_mult_rescaled * cell_bg_por_blobs_rescaled

    start = time.time()
    #print("hello")

    # 6
    big_gaussian_kernal2 = Gaussian2DKernel(x_stddev = cell_bg_conv, y_stddev = cell_bg_conv, theta = 0)
    conved_cell_highlight_bg_bigblobbed_gaussian = conv_helper(conved_cell_highlight_bg_bigblobbed, big_gaussian_kernal2, device, use_cupy_over_torch)
    conved_cell_highlight_bg_bigblobbed_gaussian_rescaled = rescale_in_range(conved_cell_highlight_bg_bigblobbed_gaussian, 1,0)

    end = time.time()
    print(end - start)

    # 7 
    conved_cell_highlight_bg_bigblobbed_gaussian_rescaled_rescaled = rescale_in_range(conved_cell_highlight_bg_bigblobbed_gaussian_rescaled, 1.1, 0)
    conved_cell_highlight_bg_bigblobbed_gaussian_rescaled_rescaled_clipped =  np.clip(conved_cell_highlight_bg_bigblobbed_gaussian_rescaled_rescaled, 0, 1)


    # out ->
    cell_bg_mult = conved_cell_highlight_bg_bigblobbed_gaussian_rescaled_rescaled_clipped

    # # 8
    # bg_noise_gaussian = np.random.normal(1, 1, hr_shape)
    # bg_noise_gaussian_scaled = rescale_in_range(bg_noise_gaussian, 1, 0.95)
    # conved_cell_highlight_bg_bigblobbed_gaussian_rescaled_rescaled_clipped_noised = conved_cell_highlight_bg_bigblobbed_gaussian_rescaled_rescaled_clipped * bg_noise_gaussian_scaled


    # -------------------- PERI STUFF

    peri_array_dict_stack = dict()
    for i in range(first_frame, first_frame + frame_len, save_rate):
        # set the frame key for the pos dict, and get the graph list for the frame (NB: (1,x) keys get the end state of the frame where x is the frame number, (0,x) gets the start state of the frame where x is the frame number
        curr_frame = i
        curr_frame_start_idx = (0, curr_frame)
        
        # graph list
        graph_list = pos_dict[curr_frame_start_idx]

        # create a blank frame array for peri
        blank_frame_peri = np.zeros((base_array_size[0] * render_mult, base_array_size[1] * render_mult)).astype(np.float32)
        
        for graph_id, graph_tuple in graph_list.items():
            if graph_tuple[1] == 'peri':

                half_render_mult = 1
                
                node_idx = (graph_id, curr_frame)
                
                
                if node_idx in full_render_only_list:
                    half_render_mult = half_render_mult_factor
                
                
                if node_idx in full_subset_drop_list and node_idx not in full_render_only_list:
                    #print('in')
                    continue
                
                # checks if the node is peri
        

                # shuffled current mito edge list
                mito = graph_tuple[0]
                t_edge_list = list(mito.edges).copy()
                random.shuffle(t_edge_list)
                
                # iterate over edges, adding them to the blank frame array
                for edge in t_edge_list:

                    # get end node x, y positions
                    pos1 = mito.nodes[edge[0]]['pos']
                    pos2 = mito.nodes[edge[1]]['pos']

                    # draw line
                    rr, cc = skimage.draw.line(int(np.round(pos1[0]*render_mult)), int(np.round(pos1[1]*render_mult)), int(np.round(pos2[0]*render_mult)), int(np.round(pos2[1]*render_mult)))
                    
                    # get brightness of the two nodes
                    b1 = mito.nodes[edge[0]]['brightness']
                    b2 = mito.nodes[edge[1]]['brightness']
                    


                    # get the midpoint of the two brightnesses for use as the base line color
                    midpoint = max(b1,b2) - (abs(b1-b2))/2
                    dimmer_mult = 1

                    # get a set of the two mito ids, and check if they are in the fission and fusion quick lookup lists
                    tup_set = set([edge[0], edge[1]])
                    if any(tup_set == s for s in fusion_tuple_list):
                        
                        # get all fussions/fissions that match with the current mito pair
                        matched_fusions_or_fission = [x[1] for x in fusion_lookups if x[0] == tup_set]

                        # if the time for the fusion/fission is within 16 frames of the current frame, dim the line based on the distance from the current frame
                        for time_val in matched_fusions_or_fission:
                            if (i-16) < time_val and time_val < (i + 16):
                                dist = abs(i - time_val)
                                dimmer_mult = dist/16

                    # draw the line, with the dimmer mult applied
                    
                    if graph_tuple[1] == 'peri':
                        blank_frame_peri[rr,cc] = midpoint#* dimmer_mult * half_render_mult
                    

                    
            # out -> 
        peri_array_dict_stack[i] = blank_frame_peri
    
    # morphology of the peri array
    peri_dict_stack_2 = dict()
    for key in peri_array_dict_stack.keys():
        peri_dict_stack_2[key] = skimage.morphology.dilation(peri_array_dict_stack[key], skimage.morphology.disk(int(np.floor(hr_render_mito_size/2)+peri_average_mito_width_add)))
        
    # take the average of all the peri_dict_stack_2 into a single array of the same shape
    peri_average = np.zeros(hr_shape).astype(np.float32)
    for key in peri_dict_stack_2.keys():
        peri_average += peri_dict_stack_2[key]
        
    peri_average /= len(peri_dict_stack_2.keys())

    # get the high overlap areas
    high_areas = (peri_average > common_peri_area_cutoff).astype(np.uint8)

    # double dilate them
    re_dilate = skimage.morphology.dilation(high_areas, skimage.morphology.disk(int(np.floor(hr_render_mito_size/2)+peri_signal_width_add[0])))
    re_dilate = skimage.morphology.dilation(re_dilate, skimage.morphology.disk(int(np.floor(hr_render_mito_size/2)+peri_signal_width_add[1])))
    # conv it
    big_gaussian_kernal2 = Gaussian2DKernel(x_stddev = peri_signal_kernal_size, y_stddev = peri_signal_kernal_size, theta = 0)
    re_dilate_conved = rescale_in_range(conv_helper(re_dilate, big_gaussian_kernal2, device, use_cupy_over_torch), 1,0)


    # also create the high signal areas 
    alt = skimage.morphology.dilation(high_areas, skimage.morphology.disk(int(np.floor(hr_render_mito_size/2)+peri_moprh_width_add)))
    # conv it
    big_gaussian_kernal3 = Gaussian2DKernel(x_stddev = peri_morph_kernal_size, y_stddev = peri_morph_kernal_size, theta = 0)
    alt_conved = rescale_in_range(conv_helper(alt, big_gaussian_kernal3, device, use_cupy_over_torch), 1,0)









    torch.backends.cudnn.benchmark = True


    # ------------- main render loop -----------------

    render_out_dict = dict()
    label_out_dict = dict()
    label_out_dict_double_res_fmask = dict()
    label_out_dict_double_res_skeleton = dict()
    mito_kernal_rad = airy_mito_kernal_size
    airy_mito_kernal = AiryDisk2DKernel(radius = mito_kernal_rad)
    hr_shape = (int(base_array_size[0] * render_mult), int(base_array_size[1] * render_mult))
    out_shape = (int(base_array_size[0] * save_mult), int(base_array_size[1] * save_mult))


    temp_invis_list = list()

    # iterate over the frames, starting from the first frame in the meta dict, then saving every save_rate frames (default 8)
    for i in range(first_frame, first_frame + frame_len, save_rate):
        print(i)
        
        # ------------------ BASE LINE GENERATION ------------------
        
        # set the frame key for the pos dict, and get the graph list for the frame (NB: (1,x) keys get the end state of the frame where x is the frame number, (0,x) gets the start state of the frame where x is the frame number
        curr_frame = i
        curr_frame_start_idx = (0, curr_frame)
        
        # graph list
        graph_list = pos_dict[curr_frame_start_idx]


        # create a blank frame array
        blank_frame_arr = np.zeros((base_array_size[0] * render_mult, base_array_size[1] * render_mult)).astype(np.float32)
        blank_frame_peri = np.zeros((base_array_size[0] * render_mult, base_array_size[1] * render_mult)).astype(np.float32)
        
        
        # iterate over the graph list, and draw the lines on the blank frame array    
        for graph_id, graph_tuple in graph_list.items():
            
            half_render_mult = 1
            
            node_idx = (graph_id, curr_frame)
            
            
            if node_idx in full_render_only_list:
                half_render_mult = half_render_mult_factor
            
            
            if node_idx in full_subset_drop_list and node_idx not in full_render_only_list:
                #print('in')
                continue
            


            # shuffled current mito edge list
            mito = graph_tuple[0]
            t_edge_list = list(mito.edges).copy()
            random.shuffle(t_edge_list)
            
            
            temp_disapear_mult = 1
            
            if mito.number_of_nodes() < 6:
                
                if len(temp_invis_list) > 0:
                    for old in temp_invis_list:
                        if graph_id == old[0] and curr_frame == old[1]+save_rate:

                            if random.randint(0,3) == 1:
                                temp_disapear_mult = random.uniform(0.001, 0.15)  
                                break
                
                chance_for_sing_frame_disapear = random.randint(0,128)
                if chance_for_sing_frame_disapear == 1:
                    
                    temp_disapear_mult = random.uniform(0.001, 0.15)  
                    
                    temp_invis_list.append((graph_id, curr_frame))          
            
            # iterate over edges, adding them to the blank frame array
            for edge in t_edge_list:

                # get end node x, y positions
                pos1 = mito.nodes[edge[0]]['pos']
                pos2 = mito.nodes[edge[1]]['pos']

                # draw line
                rr, cc = skimage.draw.line(int(np.round(pos1[0]*render_mult)), int(np.round(pos1[1]*render_mult)), int(np.round(pos2[0]*render_mult)), int(np.round(pos2[1]*render_mult)))
                
                # get brightness of the two nodes
                b1 = mito.nodes[edge[0]]['brightness']
                b2 = mito.nodes[edge[1]]['brightness']
                
                
                                    
                # get the mito brightness of the two nodes
                mito_b1 = dimmer_dict[edge[0]]
                mito_b2 = dimmer_dict[edge[1]]
                mito_brightness = (mito_b1 + mito_b2)/2   

                # get the midpoint of the two brightnesses for use as the base line color
                midpoint = max(b1,b2) - (abs(b1-b2))/2
                
                
                if midpoint> 0.18:
                    midpoint = 1.2*midpoint - 0.12
                    
                    
                dimmer_mult = 1

                # get a set of the two mito ids, and check if they are in the fission and fusion quick lookup lists
                tup_set = set([edge[0], edge[1]])
                if any(tup_set == s for s in fusion_tuple_list):
                    
                    # get all fussions/fissions that match with the current mito pair
                    matched_fusions_or_fission = [x[1] for x in fusion_lookups if x[0] == tup_set]

                    # if the time for the fusion/fission is within 16 frames of the current frame, dim the line based on the distance from the current frame
                    for time_val in matched_fusions_or_fission:
                        if (i-16) < time_val and time_val < (i + 16):
                            dist = abs(i - time_val)
                            dimmer_mult = dist/16

                
                # draw the line, with the dimmer mult applied
                blank_frame_arr[rr,cc] = midpoint* dimmer_mult * half_render_mult * mito_brightness * temp_disapear_mult
                
                if graph_tuple[1] == 'peri':
                    blank_frame_peri[rr,cc] = midpoint#* dimmer_mult * half_render_mult
                

                
        # out -> 
        base_line = blank_frame_arr
        base_peri_line = blank_frame_peri
        
        
        # ---------------- GENERATE THE GT LABELS ----------------
        


        
        blank_frame_arr = np.zeros((base_array_size[0], base_array_size[1])).astype(np.float32)
        
        for graph_id, graph_tuple in graph_list.items():
            
            node_idx = (graph_id, curr_frame)
            if node_idx in full_subset_drop_list:
                #print('in')
                continue
            
            mito = graph_tuple[0]

            for edge in mito.edges:

                pos1 = mito.nodes[edge[0]]['pos']
                pos2 = mito.nodes[edge[1]]['pos']

                rr, cc = skimage.draw.line(int(np.round(pos1[0])), int(np.round(pos1[1])), int(np.round(pos2[0])), int(np.round(pos2[1])))

                blank_frame_arr[rr,cc] = graph_id
                
        # fmask label at double double res
        label_native_rez = skimage.morphology.dilation(blank_frame_arr, skimage.morphology.disk(3)).astype(np.int32)
        
        # down ressed label
        label_real_rez = skimage.transform.resize(label_native_rez, (base_array_size[0]*save_mult, base_array_size[1]*save_mult), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
        
        
        label_out_dict[curr_frame] = label_real_rez
        print(label_native_rez.shape)
        
        
        # skeleton label
        label_native_rez_lumen = skimage.morphology.dilation(blank_frame_arr, skimage.morphology.disk(1)).astype(np.int32)
        
    
        
        # save the double res fmask and skeleton
        label_out_dict_double_res_skeleton[curr_frame] = label_native_rez_lumen.astype(np.uint16)

        label_out_dict_double_res_fmask[curr_frame] = label_native_rez.astype(np.uint16)
        
        


        

        
        
        # ------------- BASE MITO MORPHOLOGY GENERATION -------------
        

        # get the base line array for the current frame
        base_mito_lines_hr = base_line
        
        # OPERATION ORDER
        # 1. Dilation
        # 2. mult with blobs and clip 0,1
        # 3. conv with airy_mito_kernal, and rescale to 0,1
        # 4. shot noise via poisson distribution
        # 5. clip and rescale to 0,1

        
        # 1
        base_mito_morph = skimage.morphology.dilation(base_mito_lines_hr, skimage.morphology.disk(int(np.floor(hr_render_mito_size/2)+mito_base_morph_size_add)))
        
        # 2
        mito_blobs = gen_blobs_helper(base_mito_lines_hr.shape, 120)
        mito_blobs_scaled = rescale_in_range(mito_blobs, 1, mito_blob_impact_mult )
        mito_morph_blobbed = np.clip((base_mito_morph*mito_blobs_scaled), 0,1)
        
        
        start = time.time()
        #print("hello")


        # 3
        conved_mito_morph = conv_helper(mito_morph_blobbed, airy_mito_kernal, device, use_cupy_over_torch)
        conved_mito_morph_rescale = rescale_in_range(conved_mito_morph, 1, 0)
        
        end = time.time()
        print(end - start)
        
        # 4
        photon_conversion = photon_conversion_rate_mito
        noisy_mito_morph = np.random.poisson(np.clip(conved_mito_morph_rescale * photon_conversion, 0, None)) / photon_conversion

        # 5
        #noisy_mito_morph_clipped = np.clip(noisy_mito_morph, 0, 0.9)
        noisy_mito_morph_clipped_rescaled = rescale_in_range(noisy_mito_morph, 1, 0)
        
        # out ->
        base_mito_morph = noisy_mito_morph_clipped_rescaled
        



        # ------------ BACKGROUND NOISE GENERATION --------------------
        
        # vibes based correction of real values in the real_example.npy
        noise_std = bg_noise_std
        noise_mean = bg_noise_mean

            
            # OPERATION ORDER
            # 1. gen base gaussian noise
            # 2. gen porus blobs and scale
            # 3. mult and clip to 0,1
            
            
        # generate base gaussian noise
        bg_noise_gaussian = np.random.normal(noise_mean, noise_std, hr_shape)
        bg_noise_blobs = gen_blobs_helper(hr_shape, bg_noise_blob_size).astype(np.float32)
        blob_mult = rescale_in_range(bg_noise_blobs, 1, bg_blob_impact_factor)
        bg_noise = np.clip(bg_noise_gaussian * blob_mult,0,1)

        
        
        #ADD TEMPORAL BLOBS HERE
        temporal_noise_arr = temporal_blob_dict[curr_frame]

        temporal_blob_kernal = AiryDisk2DKernel(radius = mito_kernal_rad)
        temporal_noise_arr_conved = conv_helper(temporal_noise_arr, temporal_blob_kernal, device, use_cupy_over_torch)

        temporal_blob_gaussian = np.random.normal(1, noise_std, hr_shape)
        temporal_noise_arr_conved_noised = temporal_noise_arr_conved * temporal_blob_gaussian 
        temporal_noise_arr_conved_noised_shot = np.random.poisson(np.clip(temporal_noise_arr_conved_noised * photon_conversion, 0, None)) / photon_conversion

        
        
        comb_bg_noises = bg_noise + temporal_noise_arr_conved_noised_shot * .40 * temporal_blob_intensity_mult
        
        # ------------- PERI SIGNAL GENERATION --------------------
        
        # finalise the peri array region and high signal areas
        mito_peri_blobs = gen_blobs_helper(hr_shape, 30).astype(np.float32)
        mito_peri_blobs_mult = rescale_in_range(mito_peri_blobs, 1.1, 0.9)

        photon_conversion = photon_conversion_rate_add_peri_glow
        noisy_peri_morph = np.random.poisson(np.clip(re_dilate_conved * photon_conversion, 0, None)) / photon_conversion
        noisy_peri_morph = rescale_in_range(noisy_peri_morph, 1, 0) * mito_peri_blobs_mult

        photon_conversion = photon_conversion_rate_add_peri_morph
        alt_conved_noisy = np.random.poisson(np.clip(alt_conved * photon_conversion, 0, None)) / photon_conversion
        alt_conved_noisy = rescale_in_range(alt_conved_noisy, 1, 0)

        
        

        
        
        # ----------- ADDING EVERYTHING TOGETHER ----------------
        
        base_mito_morphology = base_mito_morph
        background_noise = comb_bg_noises
        cell_bg_mult = cell_bg_mult
        peri_signal = noisy_peri_morph
        
        scaled_base_mito_morphology = base_mito_morphology * base_mito_morph_max
        scaled_peri_signal = peri_signal * peri_glow_max 
        scaled_peri_moprh = alt_conved_noisy *peri_morph_max
        
        clip_factor = (base_mito_morph_max + peri_glow_max + peri_morph_max) * clip_fraction_factor
        combined_mito_and_peri = np.clip(scaled_base_mito_morphology + scaled_peri_signal + scaled_peri_moprh, 0, clip_factor)
        combined_mito_and_peri_rescaled = rescale_in_range(combined_mito_and_peri, max_signal_from_peri_and_mito, 0)
        
        
        combined_background = background_noise * rescale_in_range(cell_bg_mult, cell_bg_scale_factor, 1) + cell_bg_mult * raw_bg_glow_no_noise_mult

        
        
        full_combined = combined_mito_and_peri_rescaled + combined_background
        full_downscaled = skimage.transform.resize(full_combined, out_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.float16)
        full_cliped = np.clip(full_downscaled, 0, 1)
        

        
        
        render_out_dict[curr_frame] = full_cliped
        
    




        
    out_label_list = list()
    out_render_list = list()
    out_label_fmask = list()
    
    out_label_skeleton = list()
    out_double_res_raw = list()
    
    out_label_skeleton_weighted = list()
    out_label_fmask_weighted = list()
    
    

    for i in range(first_frame, first_frame + frame_len, save_rate):
        
        # reg res label
        out_label_list.append(label_out_dict[i])
        
        # reg res render
        out_render_list.append(render_out_dict[i])
        
        # double res fmask label
        out_label_fmask.append(label_out_dict_double_res_fmask[i])
        
        # double res skeleton label
        out_label_skeleton.append(label_out_dict_double_res_skeleton[i])
        
        # weighted skeleton label
        w = unet_weight_map(label_out_dict_double_res_skeleton[i])
        out_label_skeleton_weighted.append(w)
        
        # weighted fmask label
        w2 = unet_weight_map(label_out_dict_double_res_fmask[i])
        out_label_fmask_weighted.append(w2)
        
        # double res raw render, scaled with resize
        t_double_res_raw = skimage.transform.resize(render_out_dict[i], (label_out_dict_double_res_skeleton[i].shape[0], label_out_dict_double_res_skeleton[i].shape[0] ), order=0, preserve_range=True, anti_aliasing=False)
        out_double_res_raw.append(t_double_res_raw)
        
    out_label_arr = np.stack(out_label_list)
    out_render_arr = np.stack(out_render_list)
    out_render_skeleton_arr = np.stack(out_label_skeleton)
    out_render_fmask_arr = np.stack(out_label_fmask)
    out_render_double_res_raw_arr = np.stack(out_double_res_raw)
    out_render_skeleton_weighted_arr = np.stack(out_label_skeleton_weighted)
    out_render_fmask_weighted_arr = np.stack(out_label_fmask_weighted)


    # compressed
    
    if do_compress:
        out_render_arr = (out_render_arr*256).astype(np.uint8)
        out_render_double_res_raw_arr = (out_render_double_res_raw_arr*256).astype(np.uint8)
        out_render_skeleton_weighted_arr = out_render_skeleton_weighted_arr.astype(np.float16)
        out_render_fmask_weighted_arr = out_render_fmask_weighted_arr.astype(np.float16)
        
        
    # ----------------- save data -----------------
        
    #raw_data_save_loc = os.path.join(save_directory, 'raw_data.npy')
    #label_save_loc = os.path.join(save_directory, 'labels.npy')
    # fission_fusion_save_loc = os.path.join(save_directory, 'ff_loc.npy')

    
    matching_save_loc = os.path.join(save_directory, 'matchings.npy')
    class_save_loc = os.path.join(save_directory, 'class_dict.npy')
    params_loc = os.path.join(save_directory, 'params.npy')

    
    fmask_loc = os.path.join(save_directory, 'double_res_fmask.npy')
    skeleton_loc = os.path.join(save_directory, 'double_res_skeleton.npy')
    
    double_res_raw_loc = os.path.join(save_directory, 'double_res_raw.npy')
    
    fmask_weights_loc = os.path.join(save_directory, 'fmask_weights.npy')
    skeleton_weights_loc = os.path.join(save_directory, 'skeleton_weights.npy')


    #np.save(raw_data_save_loc, out_render_arr)
    #np.save(label_save_loc, out_label_arr)
    np.save(matching_save_loc, final_tracking_dict_reidx)
    np.save(class_save_loc, class_dict_subset_and_re_idx)
    np.save(params_loc, param_dict, allow_pickle=True)

    # np.save(fission_fusion_save_loc, combine_ff_array)
    
    np.save(fmask_loc, out_render_fmask_arr)
    np.save(skeleton_loc, out_render_skeleton_arr)
    
    np.save(double_res_raw_loc, out_render_double_res_raw_arr)
    
    np.save(fmask_weights_loc, out_render_fmask_weighted_arr)

    np.save(skeleton_weights_loc, out_render_skeleton_weighted_arr)
    
