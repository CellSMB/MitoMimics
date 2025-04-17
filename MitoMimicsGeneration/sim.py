# ========================================
# ||||||||||||| Imports  |||||||||||||||||
# ========================================

import pickle
import random
from functools import partial
import os
import sys


import numpy as np


# supress pygame output so we have a clean terminal
stdout, stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = open(os.devnull, 'w')

import pygame
import pymunk
import pymunk.pygame_util

sys.stdout, sys.stderr = stdout, stderr

import yaml

from logic.goals import *
from logic.mem_nuc import *
from logic.mito import *
from logic.util import *
from logic.fission import *

import argparse


# =========================================
# |||||||||| arg parsing ||||||||||||||||||
# =========================================

parser = argparse.ArgumentParser(description='Run the simulation with the given parameters')
parser.add_argument('--seed', type=int, help='The seed for the simulation')
parser.add_argument('--save', type=bool, help='Wether to save the outputs or not')
args = parser.parse_args()




# ========================================
# ||||||||||||| Global Params  |||||||||||
# ========================================



# Load parameters from YAML file
with open('parameters.yaml', 'r') as f:
    parameters = yaml.safe_load(f)

# NB: Parametes are safe loaded, meaning the value in the code below is not a default, but a fallback value if the parameter is not found in the yaml file
# The gui forces ? the parameters to be in the yaml file, so this is not an issue



### -------------- Space/World Parameters ----------------
space_world_parameters = parameters.get("space_world_parameters", {})

# Screen dimensions in width height. (highly recommended to keep these equal)
width = space_world_parameters.get("width", 2272)
height = space_world_parameters.get("height", 2272)

# DERIVED: Location for the center of the screen (default is the center of the screen)
center_loc = (int(np.round(width/2)), int(np.round(height/2)))

# Decimal downscale factor for the display
scale_factor = space_world_parameters.get("scale_factor", 1.3)

# Gravity and damping settings for the physics simulation, THIS GIGA BREAKS STUFF IF YOU CHANGE IT
space_gravity = tuple(space_world_parameters.get("space_gravity", [0, 0]))
space_damping = space_world_parameters.get("space_damping", 1.0e-05)

# Frames per Second for the simulation
fps = space_world_parameters.get("fps", 16)

# FPS limiter, for viewing at the saving/sampling speeds (if true limit to 8, else leave as steady)
fps_limiter = space_world_parameters.get("fps_limiter", False)

# Physics steps per frame 
physics_steps_per_frame = space_world_parameters.get("physics_steps_per_frame", 100)


### ------------- Outer Membrane Parameters ------------------ 
outer_membrane_parameters = parameters.get("outer_membrane_parameters", {})

# Number of balls to compose outer membrane out of
balls_in_outer_mem = outer_membrane_parameters.get("balls_in_outer_mem", 360)

# Nucleus radius limit
nuc_mem_rad_range = tuple(outer_membrane_parameters.get("nuc_mem_rad_range", [0.1, 0.3])) # range for the radius of the nucleus membrane
nuc_mem_rad_mult = random.uniform(nuc_mem_rad_range[0], nuc_mem_rad_range[1]) # multiplyer to apply to the radius of the nucleus membrane (to make smaller <1)
nuc_mem_rad = int(np.round(width/2)*nuc_mem_rad_mult)

# Number of balls in the leading edge in the outer membrane
leading_edge_length = outer_membrane_parameters.get("leading_edge_length", 130)

# Mass of each outer membrane ball
outer_mem_ball_mass = outer_membrane_parameters.get("outer_mem_ball_mass", 10)

# Radius of outer membrane balls
outer_mem_ball_rad = outer_membrane_parameters.get("outer_mem_ball_rad", 15)

# Minimum distance in pixels from the closest point of the outer membrane to the center of the screen
outer_mem_min_dist_to_center = outer_membrane_parameters.get("outer_mem_min_dist_to_center", 250)


outer_mem_rad_range = tuple(outer_membrane_parameters.get("outer_mem_rad_range", [0.8, 1.0])) # range for the radius of the outer membrane
outer_mem_rad_mult = random.uniform(outer_mem_rad_range[0], outer_mem_rad_range[1]) # multiplyer to apply to the radius of the outer membrane (to make smaller <1)
outer_mem_rad = int(np.round(width/2)*outer_mem_rad_mult)
shape_choices = [key for key, value in outer_membrane_parameters.get("shape_choices", {'blob': True, 'triangle': True}).items() if value] # Shape choices for the outer membrane, must include at least 1 shape





### -------------- Mitochondria Parameters ------------------ 
mitochondria_parameters = parameters.get("mitochondria_parameters", {})

# Ball mass multiplier against the radius
ball_mass_mult = mitochondria_parameters.get("ball_mass_mult", 0.1)

# Length added to the distance between balls in a chain
len_adder = mitochondria_parameters.get("len_adder", 1)

# Range of the angle for sequential balls in a chain
mito_ang_range = tuple(mitochondria_parameters.get("mito_ang_range", [15, 80]))

# Ball radius
mito_ball_radius = mitochondria_parameters.get("mito_ball_radius", 4)

# Range for the number of balls in a chain
mito_chain_len_range = tuple(mitochondria_parameters.get("mito_chain_len_range", [3, 12]))
mito_chain_len_range = [mito_chain_len_range[0]+ random.choice([0,0,0,0, 1, 2, 3]), mito_chain_len_range[1]+ random.choice([0, 1, 2, 3,5])]


# Range for the number of chains
num_chains_range = tuple(mitochondria_parameters.get("num_chains_range", [60, 130]))

# Number of chains to move to the nucleus (must be less than num_chains_range[0])
num_nuc_chains_range = tuple(mitochondria_parameters.get("num_nuc_chains_range", [15, 30]))
num_nuc_chains = random.randint(num_nuc_chains_range[0], num_nuc_chains_range[1])
#num_nuc_chains = mitochondria_parameters.get("num_nuc_chains", 25)


### -------------- Nucleus Parameters -------------------
nucleus_parameters = parameters.get("nucleus_parameters", {})

# Nuclear blob generator parameters (see generate_nucleus_points in mem_nuc.py)
nuc_interp_points = nucleus_parameters.get("nuc_interp_points", 100)
nuc_points_sample = nucleus_parameters.get("nuc_points_sample", 18)
nuc_rad_range = tuple(nucleus_parameters.get("nuc_rad_range", [0.8, 1]))

# nuc leading edge range
nuc_leading_edge_range = tuple(nucleus_parameters.get("nuc_leading_edge_range", [0.1, 0.3]))
nuc_leading_edge = random.uniform(nuc_leading_edge_range[0], nuc_leading_edge_range[1])*100


# Ellipse scale factor range, use to elipsify the nucleus
nuc_scale_factor_range = tuple(nucleus_parameters.get("nuc_scale_factor_range", [0.4, 0.9]))

# Number of internal tendrils in the nucleus
num_tendrils_range = tuple(nucleus_parameters.get("num_tendrils_range", [10, 30]))
num_tendrils = random.randint(num_tendrils_range[0], num_tendrils_range[1])

# Tendril length range
tendril_length_range = tuple(nucleus_parameters.get("tendril_length_range", [10, 25]))

# DEREVIDE: Center location of the nucleus, set to the center of the screen
nuc_center_loc = center_loc

perecent_peri_seeking_leading = nucleus_parameters.get("perecent_peri_seeking_leading", 70)


### -------------- Fission Parameters ------------------ 
fission_parameters = parameters.get("fission_parameters", {})

fission_limit = fission_parameters.get("fission_limit", 6) # min number of balls in a chain to allow fission, 
min_chain_len = fission_parameters.get("min_chain_len", 3) # min number of balls in a chain, fissioned chains will allways have at least this many balls in each half, must be less than floor(fission_limit/2)
fission_init_lockout = fps * fission_parameters.get("fission_init_lockout", 4) # lockout for fission after a chain is created (just for initial sim start, then moves to fission_lockout_range), default 4 seconds
fission_lockout_range = tuple(fps * x for x in fission_parameters.get("fission_lockout_range", (1, 32))) # range for the fission lockout, default 1-32 seconds
peri_fission_countdown_range = tuple(fps * x for x in fission_parameters.get("peri_fission_countdown_range", (1, 16))) # range for the peri fission countdown, default 1-16 seconds
fission_count_diff_mult = fission_parameters.get("fission_count_diff_mult", 0.01) # mult factor for the fission lockout applied post calculation with the difference in chain len from the base
fission_mult_range = tuple(fission_parameters.get("fission_mult_range", (0.6, 1.4))) # range for the fission multiplier
fission_limit_diff_mult = fission_parameters.get("fission_limit_diff_mult", 0.02) # base mult to apply to fission coundown, acts after all other fission countdown speed factors
peri_fission_count_diff_mult = fission_parameters.get("peri_fission_count_diff_mult", 0.02) # mult factor for the peri fission countdown applied post calculation with the difference in chain len from the base
peri_fission_mult_range = tuple(fission_parameters.get("peri_fission_mult_range", (0.6, 1.4))) # range for the peri fission multiplier

### -------------- Fusion Parameters ------------------ 
fusion_parameters = parameters.get("fusion_parameters", {})

fusion_lockout_alpha = fusion_parameters.get("fusion_lockout_alpha", 350) # alpha in gamma distribution for the fusion lockout
fusion_lockout_beta = fusion_parameters.get("fusion_lockout_beta", 1.5) # beta in gamma distribution for the fusion lockout
path_len_self_fussion_peri_only = fusion_parameters.get("path_len_self_fussion_peri_only", 10) # path length for self fusion in peri only chains
chance_peri_fussion = fusion_parameters.get("chance_peri_fussion", 2) # chance for a peri only fusion to fail (SHOULD BE LOW, NO LOCKOUT ON PERI FUSSION -> CAN BE TRIED MANY TIMES PER FRAME)
dist_from_edge_fussion = fusion_parameters.get("dist_from_edge_fussion", 3) # distance from the edge of the chain to allow fusion
chance_single_lockout_fusion_fail = fusion_parameters.get("chance_single_lockout_fusion_fail", 98) # chance for a single lockout fusion to fail
chance_base_fussion_fail = fusion_parameters.get("chance_base_fussion_fail", 60) # base chance for a fusion to fail

### ------------- Goal parameters ----------------
goal_parameters = parameters.get("goal_parameters", {})

goal_idle_exit_time = tuple(goal_parameters.get("goal_idle_exit_time", (150, 400))) # range for the time a chain will stay in the idle state
goal_idle_new_goal_idle_factor = goal_parameters.get("goal_idle_new_goal_idle_factor", 1) # influence factor that post fusion the new chain will be in the idle state
goal_idle_new_goal_leading_edge_factor = goal_parameters.get("goal_idle_new_goal_leading_edge_factor", 1) # influence factor that post fusion the new chain will be in the leading edge state
goal_idle_new_goal_peri_migr_factor = goal_parameters.get("goal_idle_new_goal_peri_migr_factor", 1) # influence factor that post fusion the new chain will be in the peri migr state

goal_leading_edge_exit_range = tuple(goal_parameters.get("goal_leading_edge_exit_range", (150, 400))) # range for the time a chain will stay in the leading edge state
goal_leading_edge_distance_till_close = goal_parameters.get("goal_leading_edge_distance_till_close", 50) # distance from the target ball that the chain will close in on the target ball
goal_leading_edge_sepcial_same_goal_fusion_factor = goal_parameters.get("goal_leading_edge_sepcial_same_goal_fusion_factor", 3) # influence factor that post fusion the new chain will be in the leading edge state

goal_peri_migr_exit_range = tuple(goal_parameters.get("goal_peri_migr_exit_range", (150, 400))) # range for the time a chain will stay in the peri migr state
goal_peri_migr_distance_till_close = goal_parameters.get("goal_peri_migr_distance_till_close", 50) # distance from the target ball that the chain will close in on the target ball
goal_peri_migr_sepcial_same_goal_fusion_factor = goal_parameters.get("goal_peri_migr_sepcial_same_goal_fusion_factor", 3) # influence factor that post fusion the new chain will be in the peri migr state

goal_random_target_exit_range = tuple(goal_parameters.get("goal_random_target_exit_range", (150, 400))) # range for the time a chain will stay in the peri migr state
goal_random_target_distance_till_close = goal_parameters.get("goal_random_target_distance_till_close", 150) # distance from the target ball that the chain will close in on the target ball
goal_random_target_special_same_goal_fusion_factor = goal_parameters.get("goal_random_target_special_same_goal_fusion_factor", 3) # influence factor that post fusion the new chain will be in the peri migr state

### -------------- Perturber Parameters ------------------ 
perturber_parameters = parameters.get("perturber_parameters", {})

perturber_val_low = perturber_parameters.get("perturber_val_low", 0.6) # low value for the perturber
perturber_val_high = perturber_parameters.get("perturber_val_high", 1.4) # high value for the perturber
perturber_perturb_max_mag = perturber_parameters.get("perturber_perturb_max_mag", 0.38) # max magnitude of the perturbation
perturber_period_low = perturber_parameters.get("perturber_period_low", 16) # low period for the perturber
perturber_period_high = perturber_parameters.get("perturber_period_high", 16) # high period for the perturber
perturber_bounce_off_hl = perturber_parameters.get("perturber_bounce_off_hl", True) # bounce off high low values
perturber_allow_oob_recov = perturber_parameters.get("perturber_allow_oob_recov", True) # allow out of bounds recovery

### -------------- Goal Repel Parameters ------------------ 
goal_repel_parameters = parameters.get("goal_repel_parameters", {})

goal_peri_init_exit_countdown_init = goal_repel_parameters.get("goal_peri_init_exit_countdown_init", 1000) # initial exit countdown for the peri init goal
idle_repel_func_vars = tuple(goal_repel_parameters.get("idle_repel_func_vars", (300, 70, 50, 10))) # repel function vars for the idle goal (see util.py func: repel_chains_from_point)
peri_repel_func_vars = tuple(goal_repel_parameters.get("peri_repel_func_vars", (400, 70, 80, 50))) # repel function vars for the peri goal (see util.py func: repel_chains_from_point)
leading_target_repel_func_vars = tuple(goal_repel_parameters.get("leading_target_repel_func_vars", (400, 180, 70, 20))) # repel function vars for the leading edge goal (see util.py func: repel_chains_from_point)
peri_init_force = goal_repel_parameters.get("peri_init_force", 1500) # force applied to the peri_init chains at startup to move to the nuc, and disturb the cell state
peri_brownian = tuple(goal_repel_parameters.get("peri_brownian", (1, 3))) # brownian motion force range of the peri chains
idle_brownian = tuple(goal_repel_parameters.get("idle_brownian", (2, 5))) # brownian motion force range of the idle chains
seeker_brownian = tuple(goal_repel_parameters.get("seeker_brownian", (2, 4))) # brownian motion force range of the seeker chains
chance_to_swap = goal_repel_parameters.get("chance_to_swap", 70) # chance to swap the goal of a chain when reaching the goal to the oppisite goal, Eg outer membrane to peri migr, Instead of just getting a random new goal
tip_min_max_diff = tuple(goal_repel_parameters.get("tip_min_max_diff", (600, 200, 8))) # Min/Max force applied to a single tip point in the peri region. Also the ammount reduced by each additional tip
seeker_base_force = goal_repel_parameters.get("seeker_base_force", 100) # base force applied to the seeker chains
seeker_per_ball_force_mult = goal_repel_parameters.get("seeker_per_ball_force_mult", 0.3) # multiplier for the force applied to the seeker chains per ball in the chain

### -------------- Goal Select Parameters ------------------ 
goal_select_parameters = parameters.get("goal_select_parameters", {})

random_goal_select_idle_chance = goal_select_parameters.get("random_goal_select_idle_chance", 45) # chance to select the idle goal
random_goal_select_random_target_chance = goal_select_parameters.get("random_goal_select_random_target_chance", 10) # chance to select the random target goal
random_goal_select_leading_edge_chance = goal_select_parameters.get("random_goal_select_leading_edge_chance", 20) # chance to select the leading edge goal
random_goal_select_peri_migr_chance = goal_select_parameters.get("random_goal_select_peri_migr_chance", 25) # chance to select the peri migr goal


### ------- Simulation Data Saving Parameters ----------
save_simulation_parameters = parameters.get("save_simulation_parameters", {})

# Set whether or not to save the simulation
save_data = save_simulation_parameters.get("save_data", True)

# Simulation file name prefix
sim_name_prefix = save_simulation_parameters.get("sim_name_prefix", "test2_")

# Save path of simulation data
save_path = save_simulation_parameters.get("save_path", "sim_output/")

# Total length of the simulation in seconds
sim_length_seconds = save_simulation_parameters.get("sim_length_seconds", 10)


### ------- Color and Rendering Parameters ----------

# General color parameters
general_color_parameters = parameters.get("general_color_parameters", {})

# Color parameters for the outer membrane, chains, and background
outer_mem_color = tuple(general_color_parameters.get("outer_mem_color", [127, 0, 255])) # Purple
outer_mem_leading_edge_color = tuple(general_color_parameters.get("outer_mem_leading_edge_color", [0, 230, 0])) # Green
chain_link_color = tuple(general_color_parameters.get("chain_link_color", [155, 155, 155])) # Grey
background_color = tuple(general_color_parameters.get("background_color", [0, 0, 0])) # Black
contact_site_color = tuple(general_color_parameters.get("contact_site_color", [255, 0, 0, 100])) # Red
base_color = tuple(general_color_parameters.get("base_color", [255, 255, 255])) # White


# Goal specific color parameters
goal_color_parameters = parameters.get("goal_color_parameters", {})

# Colors for the mito when the goal rendering is active
idle_color = tuple(goal_color_parameters.get("idle_color", [200, 0, 0])) # Red
leading_edge_color = tuple(goal_color_parameters.get("leading_edge_color", [0, 0, 255])) # Blue
peri_color = tuple(goal_color_parameters.get("peri_color", [255, 255, 0])) # Yellow
peri_init_color = tuple(goal_color_parameters.get("peri_init_color", [0, 255, 0])) # Green
peri_migr_color = tuple(goal_color_parameters.get("peri_migr_color", [255, 140, 0])) # Dark Orange

fission_site_color = tuple(goal_color_parameters.get("fission_site_color", [0,255,0])) # green 
random_target_color = tuple(goal_color_parameters.get("random_target_color", [0,255,255])) # aqua


# Rendering Parameters
render_display_parameters = parameters.get("render_display_parameters", {})

# Size of the chain links
chain_link_size = render_display_parameters.get("chain_link_size", 2) 

# Debug parameters to show casting lines between balls with a target, and their target
show_target_lines = render_display_parameters.get("show_target_lines", False)

# Color of the target lines
target_line_color = tuple(render_display_parameters.get("target_line_color", [0, 255, 255])) # Cyan

# Display visualsation type: ('lockout', 'goal', 'fission', 'none')
display_type = render_display_parameters.get("display_type", "goal")

# Display debug text
debug_text = render_display_parameters.get("debug_text", True) 



# seed set

if args.seed is not None:
    seed = args.seed
    

    
else:
    sim_output_folder_path = save_path
    sim_output_folders = [int(x[0].split('/')[-1]) for x in os.walk(sim_output_folder_path) if x[0].split('/')[-1] != sim_output_folder_path and x[0].split('/')[-1] != '']

    #(sim_output_folders)
    
    rand_int_for_seed = random.randint(10000000, 99999999)
    
    while rand_int_for_seed in sim_output_folders:
        rand_int_for_seed = random.randint(10000000, 99999999)
        
    seed = rand_int_for_seed
    
print('staring sim for seed:',seed)
    
random.seed(seed)
np.random.seed(seed)


#_________________________________________________________
# PARAMS_TO_ADD_TO_YAML
#_________________________________________________________

# length before self fusion is possible in peri region
peri_self_fusion_len_range = (5,9)

# how much to randomly add to baseline dist form edge for min 
dist_from_edge_fusion_adder = (0,2)

# OUTERMEMSHAPEPARAMS
triangle_om_samples = 40
triangle_om_interp_points = 2000
traingle_om_xy_translation = .2

blob_om_samples = 25
blob_om_rad_range = (0.65,1.0)
blob_om_interp_points = 2000
blob_om_xy_trans = .2


# params that effect how much signal individual parts of the mitochondria give, and how much that changes over time
mito_brightness_range = (0.1, 1)
mito_brightness_max_var_from_start = 0.36
mito_brightness_single_period_max_range = 0.12
mito_brightness_period_frames = 8




if args.save is not None:
    save_data = args.save



# ========================================
# |||||||||||||||| Main ||||||||||||||||||
# ========================================

def main():

    ### -------------- pygame initilisation ----------------
    
    # Initialize pygame
    pygame.init()
    
    # Set the screen size
    screen = pygame.Surface((width, height))
    
    real_screen = pygame.display.set_mode((width//scale_factor, height//scale_factor))
    
    # Create a clock object to control the frame rate
    clock = pygame.time.Clock()
    
    # Create draw options for debugging (unused in this example, but useful for debugging)
    # draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # Create a space object for the physics simulation
    space = pymunk.Space()
    
    # Set the gravity and damping params for the space
    space.gravity = space_gravity
    space.damping = space_damping
    
    # Global counter for the simulation, used for saving the state output
    g_counter = 0


    # ========================================
    # ||||||| Component Initilisation ||||||||
    # ========================================
    
    ### -------------- Mitochondria, Chain Generation, and Handling ----------------
    
    # Chain class vars
    Chain.fission_threshold = fission_limit
    Chain.init_lockout = fission_init_lockout
    Chain.fission_lockout_range = fission_lockout_range
    Chain.fusion_lockout_alpha = fusion_lockout_alpha
    Chain.fusion_lockout_beta = fusion_lockout_beta
    
    # Create a chain manager object to handle the chains, set the chance params for the fussion
    
    # ADDED
    #f_path_len_self_fussion_peri_only = 5 + random.choice([ 1, 2, 3, 4])
    f_path_len_self_fussion_peri_only = random.randint(peri_self_fusion_len_range[0], peri_self_fusion_len_range[1])

    
    chain_manager = ChainManager(
        path_len_self_fussion_peri_only = f_path_len_self_fussion_peri_only, 
        chance_peri_fussion_fail = chance_peri_fussion, 
        dist_from_edge_fussion = dist_from_edge_fussion + random.randint(dist_from_edge_fusion_adder[0], dist_from_edge_fusion_adder[1]),
        chance_single_lockout_fusion_fail = chance_single_lockout_fusion_fail, 
        chance_base_fussion_fail = chance_base_fussion_fail
        )
        
    
    # Collision manager to store the locations of the collisions, currently used for visualizing the collisions
    collision_manager = Collision_Location_Store()
    

    ### --------------- Outer membrane ----------------

    # ADDED
    
    # Choose between the shape generators, based on the shape_choices list
    choice = random.choice(shape_choices)
    if choice == 'triangle':
        func = partial(traingle_spikey_blob_generator, num_sample = triangle_om_samples, interp_points = triangle_om_interp_points, xy_trans = traingle_om_xy_translation)
    elif choice == 'blob':
        func = partial(blob_generator, num_sample = blob_om_samples, rad_range = (blob_om_rad_range[0], blob_om_rad_range[1]), interp_points = blob_om_interp_points, xy_trans = blob_om_xy_trans)
        global outer_mem_rad
        outer_mem_rad = outer_mem_rad * 0.95   
    

    # Scale the shape generated by the chosen shape generator and check if it is valid (while loops are used to check shapes untill a minimum distance between the shape and the nuc is generated)
    cell_shape_high_res = scale_and_check_valid(func, max_dist_to_scale = outer_mem_rad+outer_mem_ball_rad, min_req_valid = outer_mem_min_dist_to_center, patience = 10000)
    # Get the shape of the cell, to be used for the placement of the chains to make sure they are inbounds
    centered_cell_shape = [(x[0]+center_loc[0], x[1]+center_loc[1]) for x in cell_shape_high_res]
    # Interpolate then subset the shapes to gen consistance spacing between points
    interpolated_points = interpolate_points(cell_shape_high_res, balls_in_outer_mem)
    # Calculate the leading edge of the outer membrane and the nucleus by finding an area of high variance in the gradient of the shape
    leading_edge_outer, leading_edge_nuc = highest_gradient_variation(interpolated_points, leading_edge_length, int(leading_edge_length*0.4))
    # Add the outer membrane and nucleus to the space, along with their constraints, store the objects in the outer_mem variable

    #Fixed params for outermem
    outer_mem = generate_closed_soft_body_from_list_and_point(space, 
                                                center_loc = center_loc, 
                                                radial_spring_stiffness = 2000, 
                                                circ_spring_stiffness = 5000, 
                                                spring_damping = 0, 
                                                ball_rad = outer_mem_ball_rad, 
                                                ball_mass = outer_mem_ball_mass, 
                                                point_list = interpolated_points,
                                                leading_edge= leading_edge_outer,
                                                shape_type=3)
    # store the position of the leading edge for chain generation
    leading_edge_shape = [(x.position[0], x.position[1]) for x in outer_mem if x.is_leading_edge]
    leading_edge_shape.append(center_loc)
    
    # -------------- Goal Parameter Initilization ----------------
    # Read from input params
    
    # Give accesss to the read only global params; space, outermembrane objects, and the chain manager
    Goal.set_chain_manager_access(chain_manager)
    Goal.set_space_access(space)
    Goal.set_outer_mem_access(outer_mem)
    
    random_goal_chance_dict = dict()
    random_goal_chance_dict['idle'] = random_goal_select_idle_chance
    random_goal_chance_dict['random_target'] = random_goal_select_random_target_chance
    random_goal_chance_dict['leading_edge'] = random_goal_select_leading_edge_chance
    random_goal_chance_dict['peri_migr'] = random_goal_select_peri_migr_chance
    # Set the random goal chance dictionary to the goal class for global access
    Goal.set_random_choice_dict(random_goal_chance_dict)


    # Idle goal parameters
    Goal_idle.exit_range = goal_idle_exit_time
    Goal_idle.new_goal_idle_factor = goal_idle_new_goal_idle_factor
    Goal_idle.new_goal_leading_edge_factor = goal_idle_new_goal_leading_edge_factor
    Goal_idle.new_goal_peri_migr_factor = goal_idle_new_goal_peri_migr_factor

    
    # Perinuclear initialisation goal parameters
    Goal_peri_init.peri_init_loc_class_var = center_loc
    Goal_peri_init.exit_countdown_init = goal_peri_init_exit_countdown_init
    
    # Value to initiate the perturber from
    perturber_initiate_from_val = None

    seeking_peterbuer_template = Perturber(
        val_low = perturber_val_low,
        val_high = perturber_val_high,
        perturb_max_mag = perturber_perturb_max_mag,
        period_low= perturber_period_low,
        period_high= perturber_period_high,
        initiate_from_val= perturber_initiate_from_val,
        bounce_off_hl= perturber_bounce_off_hl,
        allow_oob_recov= perturber_allow_oob_recov
        )
    
    # Leading edge goal parameters
    Goal_leading_edge.perturber_obj_template = seeking_peterbuer_template
    Goal_leading_edge.leading_edge_exit_range = goal_leading_edge_exit_range
    Goal_leading_edge.distance_till_close = goal_leading_edge_distance_till_close
    Goal_leading_edge.sepcial_same_goal_fusion_factor = goal_leading_edge_sepcial_same_goal_fusion_factor

    Goal_peri_migr.perturber_obj_template = seeking_peterbuer_template
    Goal_peri_migr.leading_edge_exit_range = goal_peri_migr_exit_range
    Goal_peri_migr.distance_till_close = goal_peri_migr_distance_till_close
    Goal_peri_migr.sepcial_same_goal_fusion_factor = goal_peri_migr_sepcial_same_goal_fusion_factor
    
    Goal_random_target.perturber_obj_template = seeking_peterbuer_template
    Goal_random_target.leading_edge_exit_range = goal_random_target_exit_range
    Goal_random_target.distance_till_close = goal_random_target_distance_till_close
    Goal_random_target.sepcial_same_goal_fusion_factor = goal_random_target_special_same_goal_fusion_factor    
    
    # Initialise the goal manager dict from the input vars
    goal_manager_dict = dict()
    goal_manager_dict['idle_repel_func_vars'] = idle_repel_func_vars
    goal_manager_dict['peri_repel_func_vars'] = peri_repel_func_vars
    goal_manager_dict['leading_target_repel_func_vars'] = leading_target_repel_func_vars
    goal_manager_dict['peri_init_force'] = peri_init_force
    goal_manager_dict['peri_brownian'] = peri_brownian
    goal_manager_dict['idle_brownian'] = idle_brownian
    goal_manager_dict['seeker_brownian'] = seeker_brownian
    goal_manager_dict['chance_to_swap'] = chance_to_swap
    goal_manager_dict['tip_min_max_diff'] = tip_min_max_diff
    goal_manager_dict['seeker_base_force'] = seeker_base_force
    goal_manager_dict['seeker_per_ball_force_mult'] = seeker_per_ball_force_mult
    goal_manager_dict['perecent_peri_seeking_leading'] = perecent_peri_seeking_leading



    # -------------- Nuclear Membrane Instantiation ----------------
    
    # Generate points list, and second list holding distance from point to center_loc
    # See blob generator
    nuc_mem_new_scaled_interp_shifted, point_to_center_dist = generate_nucleus_points(
        nuc_points_sample,
        nuc_rad_range,
        nuc_interp_points,
        nuc_mem_rad, 
        nuc_scale_factor_range, 
        nuc_center_loc, 
        mito_ball_radius, 
        len_adder
        )
    
    # ADDED

    
    cell_brightness_peturber = Brightness_Peturber(
        max_val = mito_brightness_range[1],
        min_val = mito_brightness_range[0],
        local_peturb_range_plus_minus = mito_brightness_max_var_from_start,
        peturb_max_mag = mito_brightness_single_period_max_range,        
        period = mito_brightness_period_frames
        )
        
    

    # Add the nucleus balls/links to the space, return the lists of balls and links to add to a chain object
    # Additionally, return the ball lists for the nuc circle and internal tendrisl for nuc goal logic
    peri_ball_list, peri_link_list, store_peri_circle_balls, store_peri_tendril_balls = nuc_chain_from_points(
        nuc_mem_new_scaled_interp_shifted, 
        mito_ball_radius, 
        ball_mass_mult, 
        point_to_center_dist, 
        nuc_center_loc, 
        space, 
        num_tendrils, 
        tendril_length_range, 
        mito_ball_radius, 
        cell_brightness_peturber,
        central_spring_stiffness = 2000
    )

    # Instatiate a chain object from the lists of balls and links
    peri_chain_obj = Chain()
    peri_chain_obj.instantiate_from_lists(peri_ball_list, peri_link_list)
    peri_chain_obj.update_ball_w_chain()
    
    # Set the nucleus and outer membrane objects for the class wide goal logic wrt the nucleus
    Goal.set_nuc_mem_access(store_peri_circle_balls)
    exclude_balls = store_peri_circle_balls + store_peri_tendril_balls
    
    # Add the chain to the chain manager
    chain_manager.add_chain(peri_chain_obj)

    # Get a peri goal object and add it to the chain object
    n_goal = Goal_peri(peri_chain_obj)
    peri_chain_obj.set_goal(n_goal)
    

    
    
    ### ----------------- Chain Generation ------------------
    # Get the initial chain point locations as list of lists    
    mito_chain_list = chain_generator_closed_shape(
        num_chains_range, # tuple for the range of the number of chains
        1500, # max iterations for the chain generation 
        centered_cell_shape, # the shape of the cell, used as the outer bounds for the chain locations
        leading_edge_shape, # the leading edge of the outer membrane, used to place chains in the leading edge area
        nuc_mem_rad+(mito_ball_radius*2), # min distance from the cetner of the nuc
        width, # Screen width
        height, # Screen height
        mito_chain_len_range, # range for the number of balls in a chain
        mito_ang_range, # range for the angle for sequential balls in a chain
        mito_ball_radius, # ball radius
        ball_mass_mult, # ball mass multiplier against the radius
        cell_brightness_peturber, # brightness perturber
        nuc_leading_edge, # percent of mito in leading edge
        space # pymunk space object
        )


    # Add the chains to the chain manager
    free_chain_list = []
    for t_chain_as_body_list in mito_chain_list:

        t_chain_obj = Chain()
        t_link_list = []
        
        for i in range(1, len(t_chain_as_body_list)):
            
            body_a = t_chain_as_body_list[i-1]
            body_b = t_chain_as_body_list[i]

            link_obj = add_offset_pin_joint(space, body_a, body_b)

            t_link_list.append([body_a, body_b, link_obj])

        # Instantiate the chain object from the lists of balls and links, add the chain to the chain manager, and update the ball with the chain object
        t_chain_obj.instantiate_from_lists(t_chain_as_body_list, t_link_list)
        chain_manager.add_chain(t_chain_obj)
        t_chain_obj.update_ball_w_chain()
        
        free_chain_list.append(t_chain_obj)
        
    # Split the list randomly
    peri_init_chains = random.sample(free_chain_list, num_nuc_chains)
    normal_chains = [x for x in free_chain_list if x not in peri_init_chains]
        
    # Add the goals using the random goal instance generator
    for chain_obj in normal_chains:
        # Get a random goal and set it to the chain object
        n_goal = get_random_goal_instance(chain_obj)
        chain_obj.set_goal(n_goal)
        
    # Reinitiate the goals to attach to the nucleus
    for chain_obj in peri_init_chains:
        # Get the nucleus seeking goal and set it to the chain object
        n_goal = Goal_peri_init(chain_obj)
        chain_obj.set_goal(n_goal)
    
    
    
    # ========================================
    # ||||||||||||| Main Loop  |||||||||||||||
    # ========================================


    # ---------------- Main Loop Variables ----------------

    # Used for storing the output at regular intervals
    state_output = dict()
    
    # Running parameter: Sets to false when stopping simulation, currently on quit event or after sim_length steps when save_data is True
    running = True
    
    # Init over parameter: Used to check if the initialisation is over
    in_init_state = True
    
    # Starting number of non peri_init or peri chains
    init_num_non_peri_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type != 'peri_init' and x.get_goal().goal_type != 'peri'])
    
    # Starting number of perinuclear chains
    init_num_peri_balls = sum([x.num_balls() for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'peri'])    
    
    # Starting peri_size
    peri_fission_countdown = random.uniform(peri_fission_countdown_range[0], peri_fission_countdown_range[1])
    
    # Flag used to track if a peri fission has been attempted
    peri_fission_flag = False

    # Simulation start frame
    sim_start_frame = 0
    
    # Cell outer membrane position dictionary
    cell_outer_mem_pos_dict = dict()
    
    # sim finished sucsesfully flag
    sim_finished_sucsessfully = False
    

    init_extra_counter = fps * random.randint(1,17)
    print(init_extra_counter)


    # Main loop
    while running:
        
        # --------------- Initialisation Phase Check ----------------------
        
        # Get the number of remaining peri_init chains
        num_peri_init_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'peri_init'])
        
        # Do only once the init phase is over
        if num_peri_init_chains == 0 and in_init_state == True:
            
            if init_extra_counter >= 0:
                init_extra_counter -= 1
            else:

                # Set the init state to false
                in_init_state = False
                
                # Retrieve the number of non peri_init chains, for fission handling rates (if there is more than the starting number of chains, increase the fission rate)
                init_num_non_peri_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type != 'peri_init' and x.get_goal().goal_type != 'peri'])
                
                # Retrieve the initial peri size for the peri fission countdown, used to increase the countdown based on the size of the peri chains
                init_num_peri_balls = sum([x.num_balls() for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'peri'])  
                
                sim_start_frame = g_counter  

            #print('INIT OVER, SIM STARTED')

        # -------------- Pygame Input Handling  ----------------

        # Input handling (checks for quit events and stops the simulation)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # ------------- g_counter store in collision_manager ----------------
        
        collision_manager.set_g_counter(g_counter)
        
        # ------------- save state data at start of frame --------------------
        
        if save_data:
            current_state = return_graphs(space, chain_manager)
            state_output[(0,g_counter)] = current_state
        
        
        # -------------- Fission and Fusion Lockout Handling ----------------
        
        # Update the counters of all chains
        chain_manager.lower_all_lockouts()

        # Fission lockout for non peri region
        fission_multiplier = 1 # base fission multiplier
        curr_num_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type != 'peri_init' and x.get_goal().goal_type != 'peri']) # get the current number of non peri chains
        diff_chains_from_start = (curr_num_chains - init_num_non_peri_chains) # get the difference in the number of chains from the start
        fission_num_chain_offset = diff_chains_from_start * fission_count_diff_mult # get the offset for the fission multiplier based on the difference in the number of chains
        fission_multiplier += -1 * fission_num_chain_offset # add the offset to the fission multiplier
        fission_multiplier = np.clip(fission_multiplier, a_min= fission_mult_range[0], a_max= fission_mult_range[1]) # clip the fission multiplier to the range
        chain_manager.lower_all_fission_lockouts(multiplier=fission_multiplier, len_factor= fission_limit_diff_mult) # lower the fission lockouts for all chains, mod by the mult
        
        # Perinuclear fission handling 
        peri_fission_multiplier = 1 # base peri fission multiplier
        curr_num_peri_balls = sum([x.num_balls() for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'peri'])
        peri_diff_balls_from_start = (curr_num_peri_balls - init_num_peri_balls) # get the difference in the number of chains from the start
        peri_fission_num_chain_offset = peri_diff_balls_from_start * peri_fission_count_diff_mult # get the offset for the fission multiplier based on the difference in the number of chains
        peri_fission_multiplier += peri_fission_num_chain_offset
        peri_fission_multiplier = np.clip(peri_fission_multiplier, a_min= peri_fission_mult_range[0], a_max= peri_fission_mult_range[1]) # clip the fission multiplier to the range
        
        # Decrease the counter by 1 * the multiplier, if the counter is less than or equal to 0, set the flag to true and reset the counter
        if not in_init_state:
            if peri_fission_countdown <= 0:
                peri_fission_flag = True
                peri_fission_countdown = random.uniform(peri_fission_countdown_range[0], peri_fission_countdown_range[1])
            else:
                peri_fission_countdown -= 1 * peri_fission_multiplier

        

        # -------------- Fussion Handling ----------------

        # Calls all collision logic
        setup_collision_handler(space, chain_manager, collision_manager)
        
        # Updates the collision manager, decrementing the collision counter variable so that the collisions are only shown for a few frames
        collision_manager.update()
        
        
        # ------------- Fission Handling non peri ----------------
        
        # Calls the fission manager, which handles the fission of chains (see fission.py)
        fission_manager(chain_manager, space, collision_manager, fission_threshold=fission_limit, min_chain_len=min_chain_len)


        # ------------- Fission Handling peri ----------------

        # check if the simulation is not in the init state
        if not in_init_state:
            # if the peri count down is less than or equal to 0, try to fission a peri chain
            if peri_fission_flag:
                peri_fission_worked = fission_manager_peri(chain_manager, space, collision_manager, exclude_balls)
                if not peri_fission_worked: # overide the countdown to be .5 second if the fission did not work
                    peri_fission_countdown = fps/2
                    peri_fission_flag = False
                else:
                    peri_fission_flag = False


        # ------------- Goal Managment ----------------- 
        
        manage_goals(chain_manager, space, center_loc, goal_manager_dict) 

        # -------------- Physics ------------------------
        
        # Move the physics simulation forward in time
        space.step(1/physics_steps_per_frame)
        

        # update brightness of each portion of each mito
        for chain in chain_manager.get_chain_list():
            for ball in chain.get_chain_bodies():
                ball.store_perturber.update()


        # ------------------- Rendering --------------------
        
        # Draw background
        screen.fill(background_color)
        
        
        # Draws the collisions: Currently used for visualizing the collisions that add a link between chains
        for x in collision_manager.get_collision_list():
            pygame.draw.circle(screen, contact_site_color, (int(x[0][0]), int(x[0][1])), x[1]*2)
            
        for x in collision_manager.get_fussion_list():
            pygame.draw.circle(screen, fission_site_color, (int(x[0][0]), int(x[0][1])), x[1]*2)
            
        # Draws balls for both the chains and the nucleus/outer membrane
        for shape in space.shapes:
            
            # Check if ball is a circle (only used by the balls in chains and outer membrane)
            if isinstance(shape, pymunk.Circle):
                
                # checks if the shape is part of a chain by checking the collision type
                if shape.collision_type == 1:
                    
                    # Checks what the rendering mode is based on the display type variable
                    color_mult = 1
                    
                    if hasattr(shape.body, 'store_perturber'):
                        color_mult =shape.body.store_perturber.get_val_without_update()
                    
                    
                    # Lockout display type
                    if display_type == 'lockout':
                    
                        lockout_base_color = [255,255,255]

                        # If the chain is not in a lockout state, draw the ball with the base color
                        if shape.body.store_chain.get_lockout() <= 0: 
                            pygame.draw.circle(screen, lockout_base_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                        
                        # If the chain is in a lockout state, draw the ball with a color that is based on the lockout counter (currently the r and b valuess)
                        else: 
                            lockout_counter = 255 - (np.clip(int(shape.body.store_chain.get_lockout()), a_min=1, a_max=255))
                            lockout_base_color[0] = int(lockout_counter)
                            lockout_base_color[2] = int(lockout_counter)

                            pygame.draw.circle(screen, lockout_base_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))

                    # Lockout display type
                    elif display_type == 'fission':
                    
                        lockout_base_color = [255,255,255]
                        
                        if len(shape.body.store_chain.return_nodes_w_degree_one()) == 3:
                            pygame.draw.circle(screen, [0,255,255], (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius)+5)

                        
                        if shape.body.store_chain.get_goal().goal_type == 'peri_init':
                            pygame.draw.circle(screen, [255,0,0], (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        if shape.body.store_chain.get_goal().goal_type == 'peri':
                            pygame.draw.circle(screen, peri_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        if shape.body.store_chain.num_balls() < 6:
                            pygame.draw.circle(screen, [255, 0, 255], (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                        
                        else:
                            # If the chain is not in a lockout state, draw the ball with the base color
                            if shape.body.store_chain.get_curr_fission_lockout() <= 0: 
                                pygame.draw.circle(screen, lockout_base_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                            # If the chain is in a lockout state, draw the ball with a color that is based on the lockout counter (currently the r and b valuess)
                            else: 
                                lockout_counter = 255 - (np.clip(int(shape.body.store_chain.get_curr_fission_lockout()), a_min=1, a_max=255))
                                lockout_base_color[0] = int(lockout_counter)
                                lockout_base_color[2] = int(lockout_counter)

                                pygame.draw.circle(screen, lockout_base_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                                
                            
                    # Goal display type
                    elif display_type == 'goal':
                        
                        if shape.body.store_chain.get_goal().goal_type == 'idle':
                            new_color = [int(x*color_mult) for x in idle_color]
                            pygame.draw.circle(screen, new_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        elif shape.body.store_chain.get_goal().goal_type == 'peri_init':
                            new_color = [int(x*color_mult) for x in peri_init_color]
                            pygame.draw.circle(screen, new_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        elif shape.body.store_chain.get_goal().goal_type == 'leading_edge':
                            new_color = [int(x*color_mult) for x in leading_edge_color]
                            pygame.draw.circle(screen, new_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        elif shape.body.store_chain.get_goal().goal_type == 'peri':
                            new_color = [int(x*color_mult) for x in peri_color]
                            pygame.draw.circle(screen, new_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        elif shape.body.store_chain.get_goal().goal_type == 'peri_migr':
                            new_color = [int(x*color_mult) for x in peri_migr_color]
                            pygame.draw.circle(screen, new_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                            
                        elif shape.body.store_chain.get_goal().goal_type == 'random_target':
                            new_color = [int(x*color_mult) for x in random_target_color]
                            pygame.draw.circle(screen, new_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                        else:
                            print('check goal logic somethings effed, cause a classless chain was rendered')
                            pygame.draw.circle(screen, base_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                    
                    # No display type (all balls set to base_color)
                    else:
                        pygame.draw.circle(screen, base_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                    

                # Checks if the shape is part of the outer membrane or the nucleus by checking the collision type
                # Additionally, sets a different color for the leading edge of the outer membrane
                elif shape.collision_type == 3 or shape.collision_type == 4:
                    if shape.body.is_leading_edge:
                        pygame.draw.circle(screen, outer_mem_leading_edge_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                    else:
                        pygame.draw.circle(screen, outer_mem_color, (int(shape.body.position.x), int(shape.body.position.y)), int(shape.radius))
                        


        # Draws all chains by using pygame draw lines between balls connected by a PinJoint constraint     
        for constraint in space.constraints:
            if isinstance(constraint, pymunk.PinJoint):
                
                # NB: this draws the line between the centers of the balls, not where the pin joint is connected
                
                # Get the positions of the two bodies connected by the PinJoint
                body_a_pos = constraint.a.position
                body_b_pos = constraint.b.position
                
                # Retrieve body positions
                pygame_a_pos = body_a_pos.x, body_a_pos.y
                pygame_b_pos = body_b_pos.x, body_b_pos.y
                
                # Draw a line between the two positions
                pygame.draw.line(screen,  chain_link_color, pygame_a_pos, pygame_b_pos, chain_link_size)
            
            
        # Draw the target lines between the leading edge and the target ball, if show_target_lines is True
        show_target_lines = True
        if show_target_lines:  
            
            for chain_obj in chain_manager.get_chain_list():
                curr_goal = chain_obj.get_goal()
                if curr_goal.goal_type == 'leading_edge' or curr_goal.goal_type == 'peri_migr' or curr_goal.goal_type == 'random_target':
                    
                    leading_ball = curr_goal.lead_ball
                    target_ball = curr_goal.target_ball
                    
                    if curr_goal.goal_type == 'leading_edge':
                        line_color = leading_edge_color
                    elif curr_goal.goal_type == 'peri_migr':
                        line_color = peri_migr_color
                    elif curr_goal.goal_type == 'random_target':
                        line_color = random_target_color
                    
                    pygame.draw.line(screen, line_color, (int(leading_ball.position.x), int(leading_ball.position.y)), (int(target_ball.position.x), int(target_ball.position.y)), 1)



        # -------- Game Physics/Screenspace Updates -----------

        scaled_surface = pygame.transform.smoothscale(screen, (width // scale_factor, height // scale_factor))
        real_screen.blit(scaled_surface, (0, 0))
        
        if debug_text:
            
            text_bg_padding = 4
            
            # Text counter object
            j = Text_counter(text_bg_padding,16)
        
            # Fonts
            font_small = pygame.font.SysFont('Arial', 12)
            font_big = pygame.font.SysFont('Arial', 14)
            
            # List to store the text to render
            text_render_list = []
            
            # ------------- gen sim info ----------------
            # Display the current state of the simulation
            if in_init_state:
                text_render_list.append(( font_big.render(f'INITIALIZING', True, (255, 255, 255)), (text_bg_padding, j.update()) ))

            else:
                text_render_list.append((font_big.render(f'RUNNING', True, (255, 255, 255)), (text_bg_padding, j.update())))
            
            j.update()
            
            text_render_list.append((font_big.render(f'--- Gen Info ---', True, (255, 255, 255)), (text_bg_padding, j.update())))
            
            # Time in sim seconds (on slow)
            text_render_list.append((font_small.render(f'Sim Seconds: {g_counter/16}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            # Frame counter
            text_render_list.append((font_small.render(f'Frame: {g_counter}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            # FPS estimate
            text_render_list.append((font_small.render(f'FPS: {np.round(clock.get_fps(),2)}', True, (255, 255, 255)), (text_bg_padding, j.update())))


            # ------------ Chain Information ----------------
            
            j.update()
            text_render_list.append((font_big.render(f'--- Goal Info ---', True, (255, 255, 255)), (text_bg_padding, j.update())))
            
            num_idle_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'idle'])
            num_leading_edge_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'leading_edge'])
            num_peri_seeking_chains = len([x for x in chain_manager.get_chain_list() if x.get_goal().goal_type == 'peri_migr'])
            text_render_list.append((font_small.render(f'number of idle chains: {num_idle_chains}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'number of leading edge chains: {num_leading_edge_chains}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'number of peri migr chains: {num_peri_seeking_chains}', True, (255, 255, 255)), (text_bg_padding, j.update())))

            # ------------ Fission Information ----------------
            j.update()
            text_render_list.append((font_big.render(f'--- Fission Info ---', True, (255, 255, 255)), (text_bg_padding, j.update())))
            
            text_render_list.append((font_small.render(f'init num chains: {init_num_non_peri_chains}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'number of chains: {curr_num_chains}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            test_list = []
            for chain_obj in chain_manager.get_chain_list():
                if chain_obj.num_balls() >= fission_limit and chain_obj.get_goal().goal_type != 'peri_init' and chain_obj.get_goal().goal_type != 'peri':
                    test_list.append(chain_obj)
            text_render_list.append((font_small.render(f'number of fission valid chains: {len(test_list)}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'diff chains from start: {diff_chains_from_start}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'fission multiplier: {np.round(fission_multiplier,2)}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            
            # ------------ Perinuclear Information -----------------
            
            j.update()
            text_render_list.append((font_big.render(f'--- Peri Info ---', True, (255, 255, 255)), (text_bg_padding, j.update())))
            
            text_render_list.append((font_small.render(f'init size of peri chain: {init_num_peri_balls}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'size of peri chain: {curr_num_peri_balls}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'diff peri balls from start: {curr_num_peri_balls- init_num_peri_balls}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'peri fission countdown: {np.round(peri_fission_countdown,2)}', True, (255, 255, 255)), (text_bg_padding, j.update())))
            text_render_list.append((font_small.render(f'peri fission mult: {np.round(peri_fission_multiplier,2)}', True, (255, 255, 255)), (text_bg_padding, j.update())))



            # -------------- Actual render text ----------------
            
            # get max width of text rects
            max_width = 0
            for x in text_render_list:
                text_rect = x[0].get_rect(topleft=(4, x[1][1]))
                if text_rect.width > max_width:
                    max_width = text_rect.width
                    
            # Create a semi transparent surface to render on
            background_surface = pygame.Surface((max_width + text_bg_padding*2, j.return_curr()), pygame.SRCALPHA)
            background_color2 = (128, 128, 128, 64)  # Semi-transparent black
            background_surface.fill(background_color2)
            real_screen.blit(background_surface, (0,0))


            for x in text_render_list:
                
                real_screen.blit(x[0], x[1])


        #fps_limiter = False
        if fps_limiter != False:
            if g_counter % fps_limiter == 0:
            # update the display
                pygame.display.flip()

        else:
            pygame.display.flip()

        # Frame limiter: Sets the fps of the simulation by waiting for the time it takes to reach the desired fps
        clock.tick(fps)
        
        
        # ----------- Saving State Output at end of frame (if save_data is True) -----------
        
        if save_data:
                
            current_state = return_graphs(space, chain_manager)
            state_output[(1,g_counter)] = current_state
            
            t_outer_mem_pos_list = list()
            for outer_mem_body in outer_mem:
                t_outer_mem_pos_list.append(np.array(outer_mem_body.position))
            cell_outer_mem_pos_dict[g_counter] = t_outer_mem_pos_list
            
            
            g_counter += 1
        
            # End after sim_length steps
            if not in_init_state:
            
                if g_counter > ((sim_length_seconds * fps) + sim_start_frame + (fps * 2)):
                    
                    sim_finished_sucsessfully = True
                    
                    running = False
            
        else:
            g_counter += 1
            
            

    # ===============================================
    # ||||||||||| Post Simulation Logic |||||||||||||
    # ===============================================

    # Quit the simulation and run post simulation logic
    pygame.quit()
    
    
    # Writes a pickle file with the state output
    if save_data and sim_finished_sucsessfully:
        
        
        
        
        metadata_dict = dict()
        metadata_dict['sim_length'] = sim_length_seconds
        metadata_dict['fps'] = fps
        metadata_dict['first_frame'] = sim_start_frame
        metadata_dict['len_in_frames'] = (sim_length_seconds * fps)
        metadata_dict['len_in_seconds'] = sim_length_seconds
        metadata_dict['frame_size'] = (width, height)
        metadata_dict['fps'] = fps
        metadata_dict['base_mito_size'] = mito_ball_radius
        
        
        
        
        # send this to the sim file tbh

        all_keys = []

        for key, value in state_output.items():
            for key, value in value.items():
                all_keys.append(key)
                
        for each in collision_manager.save_list2:
            all_keys.append(each[1][0])
            all_keys.append(each[1][2])
            all_keys.append(each[1][3])
            
        # fusion keys
        for each in collision_manager.save_list:
            all_keys.append(each[1][0])
            all_keys.append(each[1][1])
            all_keys.append(each[1][3])

        all_keys = list(set(all_keys))

        # dict from index to key
        key_dict = {}
        i = 2
        for key in all_keys:
            key_dict[key] = i
            i+=1

        # remap the unique keys to values starting at 
        #print(f'number of unique keys: {len(all_keys)}')

        for each in collision_manager.save_list2:
            # parent
            each[1][0] = key_dict[each[1][0]]
            
            # children
            each[1][2] = key_dict[each[1][2]]
            each[1][3] = key_dict[each[1][3]]

        for each in collision_manager.save_list:
            # parents
            each[1][0] = key_dict[each[1][0]]
            each[1][1] = key_dict[each[1][1]]
            
            # child
            each[1][3] = key_dict[each[1][3]]

        for key, value in state_output.items():
            t_dict = {}
            for key2, value2 in value.items():
                t_dict[key_dict[key2]] = value2
                
            state_output[key] = t_dict
        
        
        
        save_folder_path = os.path.join(save_path, str(seed))
        create_directory_if_empty_or_not_exists(save_folder_path)

        
        # Save the state output
        with open(os.path.join(save_folder_path, 'state_output.pkl'), 'wb') as f:
            pickle.dump(state_output, f)
        
        with open(os.path.join(save_folder_path, 'fusion_output.pkl'), 'wb') as f:
            pickle.dump(collision_manager.save_list , f)

        with open(os.path.join(save_folder_path, 'fission_output.pkl'), 'wb') as f:
            pickle.dump(collision_manager.save_list2 , f)
            
        with open(os.path.join(save_folder_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata_dict, f)
            
        with open(os.path.join(save_folder_path, 'outer_mem_loc.pkl'), 'wb') as f:
            pickle.dump(cell_outer_mem_pos_dict, f)
            
            


            
    
            

if __name__ == "__main__":
    main()
    
    