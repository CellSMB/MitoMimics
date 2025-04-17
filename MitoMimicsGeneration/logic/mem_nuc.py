# ------------ Imports --------------

import math
import operator
import random
from functools import reduce
from math import cos, radians, sin

import numpy as np
import pymunk
import pymunk.pygame_util
from scipy.interpolate import make_interp_spline

from .util import *

# -------------------------------------------------------------------------------------
#                             cell generation
# -------------------------------------------------------------------------------------

def blob_generator(num_sample=25, rad_range=(0.65, 1.0), interp_points=500, xy_trans=0.2):
    """
    Generate a blob shape by generating random points within a circle, interpolating the points to create a smoother shape,
    and then condensing and sorting the coordinates into a clockwise list of tuples. The function also checks for sharp joins
    between adjacent points and smooths them out.

    Parameters:
    - num_sample (int): The number of random points to generate within the circle (default: 25).
    - rad_range (tuple): The range of radii for the random points (default: (0.65, 1.0)).
    - interp_points (int): The number of points to interpolate between the random points (default: 500).
    - xy_trans (float): The maximum translation in the x and y directions (default: 0.2).

    Returns:
    - return_coords (list): A list of tuples representing the coordinates of the smoothed blob shape.
    """
    # Generate random points within a circle
    num_points = num_sample
    theta = np.linspace(0, 2*np.pi, num_points)
    r = np.random.uniform(rad_range[0], rad_range[1], size=num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    while np.linalg.norm(np.array([x[-1], y[-1]]) - np.array([x[0], y[0]])) > 0.03:
        theta = np.linspace(0, 2*np.pi, num_points)
        r = np.random.uniform(rad_range[0], rad_range[1], size=num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    # Interpolate the points to create a smoother shape
    interp_points = interp_points
    f_x = make_interp_spline(np.linspace(0, 1, num_points), x, k=3)
    f_y = make_interp_spline(np.linspace(0, 1, num_points), y, k=3)
    x_smooth = f_x(np.linspace(0, 1, interp_points))
    y_smooth = f_y(np.linspace(0, 1, interp_points))

    # Condense and sort x, y coordinates into a clockwise list of tuples
    coords = list(zip(x_smooth, y_smooth))
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    sorted_coordinates = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    sorted_coordinates.append(sorted_coordinates[0])

    # Check for sharp joins which are above a certain threshold value between two adjacent points
    threshold = 0.001
    distances = np.sqrt(np.sum(np.diff(sorted_coordinates, axis=0)**2, axis=1))
    sharp_joins = np.where(distances > threshold)[0]

    # Smooth out sharp joins
    smoothed_coordinates = sorted_coordinates.copy()
    for idx in sharp_joins:
        if idx == 0 or idx == len(sorted_coordinates) - 1:
            continue  # Skip the first and last points
        smoothed_coordinates[idx] = tuple((np.array(smoothed_coordinates[idx - 1]) + np.array(smoothed_coordinates[idx + 1])) / 2)

    x_trans = random.uniform(-xy_trans, xy_trans)
    y_trans = random.uniform(-xy_trans, xy_trans)

    return_coords = [(x[0] + x_trans, x[1] + y_trans) for x in smoothed_coordinates]

    return return_coords


def traingle_spikey_blob_generator(num_sample=40, interp_points=500, xy_trans=0.2):
    """
    Generates a spikey blob shape within a triangle.

    Args:
        num_sample (int): The number of sample points to generate within the triangle. Default is 40.
        interp_points (int): The number of points to interpolate between the generated sample points. Default is 500.
        xy_trans (float): The maximum translation in the x and y directions applied to the generated shape. Default is 0.2.

    Returns:
        list: A list of (x, y) coordinates representing the spikey blob shape.

    """
    # Generate random points within a triangle
    # Adapted from: https://www.desmos.com/calculator/9tdpmy2djo
    num_points = num_sample
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Fixed Values
    a = random.uniform(np.pi / 2, 3 * np.pi / 4)  # Stretch/Deformation
    a = random.uniform(2.05, 2.22)  # Stretch/Deformation

    b = random.choice([0, 1])  # Reflection
    c = random.uniform(np.pi / 2 - a, a - np.pi / 2)  # Skew/Weightedness
    c = random.uniform(-0.2, 0.2)  # Stretch/Deformation

    phi = random.uniform(a, 2 * np.pi - a)  # Rotation

    # Functions for generating triangle in Polar coordinates
    def j(x):
        return abs(x - phi - a) - abs(x - phi - c) + abs(x - phi + a) - b * np.pi

    r = 1 / np.cos(j(theta))

    # Add random protrusions to the shape
    num_protrusions = random.randint(7, 13)

    protrusion_angles = np.linspace(0, 2 * np.pi, num_protrusions)
    # protrusion_angles = [np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]  # Angles at which protrusions occur
    protrusion_size = 0.6  # Size of the protrusions

    for angle in protrusion_angles:
        index = np.abs(theta - angle).argmin()
        r[index] += random.uniform(0, protrusion_size)

    # Convert Polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Use a while loop until the first and last points are within some distance of each other to prevent sharp joins
    while np.linalg.norm(np.array([x[-1], y[-1]]) - np.array([x[0], y[0]])) > 0.001:
        theta = np.linspace(0, 2 * np.pi, num_points)

        a = random.uniform(np.pi / 2, 3 * np.pi / 4)  # Stretch/Deformation
        a = random.uniform(2.05, 2.22)  # Stretch/Deformation

        b = random.choice([0, 1])  # Reflection
        c = random.uniform(np.pi / 2 - a, a - np.pi / 2)  # Skew/Weightedness
        c = random.uniform(-0.2, 0.2)  # Stretch/Deformation

        phi = random.uniform(a, 2 * np.pi - a)  # Rotation

        r = 1 / np.cos(j(theta))
        # Add random protrusions to the shape
        num_protrusions = random.randint(7, 13)

        protrusion_angles = np.linspace(0, 2 * np.pi, num_protrusions)
        # protrusion_angles = [np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]  # Angles at which protrusions occur
        protrusion_size = 0.5  # Size of the protrusions

        for angle in protrusion_angles:
            index = np.abs(theta - angle).argmin()
            r[index] += random.uniform(0, protrusion_size)

        # Convert Polar coordinates to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    # Condense and sort x, y coordinates into a clockwise list of tuples
    coords = list(zip(x, y))
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    sorted_coordinates = sorted(coords,
                                key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    sorted_coordinates.append(sorted_coordinates[0])

    # Check for sharp joins which are above a certain threshold value between two adjacent points
    threshold = 1
    distances = np.sqrt(np.sum(np.diff(sorted_coordinates, axis=0) ** 2, axis=1))
    sharp_joins = np.where(distances > threshold)[0]

    # Smooth out sharp joins
    smoothed_coordinates = sorted_coordinates.copy()
    for idx in sharp_joins:
        if idx == 0 or idx == len(sorted_coordinates) - 1:
            continue  # Skip the first and last points
        smoothed_coordinates[idx] = tuple((np.array(smoothed_coordinates[idx - 1]) + np.array(
            smoothed_coordinates[idx + 1])) / 2)
    x, y = zip(*smoothed_coordinates)

    # Interpolate the points to create a smoother shape
    f_x = make_interp_spline(np.linspace(0, 1, len(x)), x, k=3)
    f_y = make_interp_spline(np.linspace(0, 1, len(y)), y, k=3)
    x_smooth = f_x(np.linspace(0, 1, interp_points))
    y_smooth = f_y(np.linspace(0, 1, interp_points))

    smoothed_coords = list(zip(x_smooth, y_smooth))

    x_trans = random.uniform(-xy_trans, xy_trans)
    y_trans = random.uniform(-xy_trans, xy_trans)

    return_coords = [(x[0] + x_trans, x[1] + y_trans) for x in smoothed_coords]

    return return_coords


def scale_and_check_valid(lop_func, max_dist_to_scale=1136, min_req_valid=250, patience=1000):
    """
    Scales a list of points generated by the given function and checks if the scaled points are valid with respect to minimum distance from the origin.

    Args:
        lop_func (function): A function that generates a list of points.
        max_dist_to_scale (float): The maximum distance to scale the points. Default is 1136.
        min_req_valid (float): The minimum required distance for the scaled points to be considered valid. Default is 250.
        patience (int): The maximum number of iterations to wait for valid points. Default is 1000.

    Returns:
        list or None: The scaled points if they are valid, or None if no valid points are found within the patience limit.
    """
    found = False
    patience_counter = 0
    vals_to_return = None

    while not found:
        # generate points from the list of points function (lop_func)
        lop = lop_func()

        # calc the distance of each point from the center and find the max and min
        dist_list = [math.dist((0, 0), x) for x in lop]
        max_dist = max(dist_list)
        min_dist = min(dist_list)

        # set the scale factor
        scale_fac = max_dist_to_scale / max_dist

        # check if the min distance is greater than the min required valid
        if min_dist * scale_fac <= min_req_valid:
            if patience_counter >= patience:
                return None
            else:
                patience_counter += 1
        # if it is then scale the points and return them
        else:
            vals_to_return = [(x[0] * scale_fac, x[1] * scale_fac) for x in lop]
            found = True

    return vals_to_return


def interpolate_points(shape, y):
    """
    Interpolates points along the perimeter of a shape.

    Args:
        shape (list): List of (x, y) coordinates representing the shape.
        y (int): Number of points to interpolate along the perimeter.

    Returns:
        list: List of interpolated points along the perimeter of a closed shape.

    """
    
    # Function to interpolate between two points by generating a new point at an arbitrary fraction of the distance between them from the first point
    def interpolate(point1, point2, fraction):
        return (point1[0] + fraction * (point2[0] - point1[0]), point1[1] + fraction * (point2[1] - point1[1]))
    
    # get an estimate of the perimeter
    perimeter = 0
    for i in range(len(shape)):
        perimeter += math.dist(shape[i], shape[(i+1) % len(shape)])
    
    # set the segment length based on a division of the perimeter by the number of points required
    segment_length = perimeter / y
    
    # set the initial values
    result = []
    remaining_distance = segment_length
    current_point = shape[0]
    result.append(current_point)
    
    # loop through the shape and interpolate points based on the segment length
    for i in range(len(shape)):
        next_point = shape[(i+1) % len(shape)]
        dist_to_next_point = math.dist(current_point, next_point)
        
        while dist_to_next_point >= remaining_distance:
            fraction = remaining_distance / dist_to_next_point
            new_point = interpolate(current_point, next_point, fraction)
            result.append(new_point)
            current_point = new_point
            
            dist_to_next_point -= remaining_distance
            remaining_distance = segment_length
            
            if len(result) == y:
                return result
        
        remaining_distance -= dist_to_next_point
        current_point = next_point
    
    # Add the last point to the result
    while len(result) < y:
        result.append(shape[-1])
    
    return result

def interpolate_points_dist(shape, segment_length):
    """
    Interpolates points along the perimeter of a shape.

    Args:
        shape (list): List of (x, y) coordinates representing the shape.
        y (int): Number of points to interpolate along the perimeter.

    Returns:
        list: List of interpolated points along the perimeter of a closed shape.

    """
    
    # Function to interpolate between two points by generating a new point at an arbitrary fraction of the distance between them from the first point
    def interpolate(point1, point2, fraction):
        return (point1[0] + fraction * (point2[0] - point1[0]), point1[1] + fraction * (point2[1] - point1[1]))
    
    # get an estimate of the perimeter
    perimeter = 0
    for i in range(len(shape)):
        perimeter += math.dist(shape[i], shape[(i+1) % len(shape)])
    
    # set the segment length based on a division of the perimeter by the number of points required

    y = perimeter/segment_length
    
    # set the initial values
    result = []
    remaining_distance = segment_length
    current_point = shape[0]
    result.append(current_point)
    
    # loop through the shape and interpolate points based on the segment length
    for i in range(len(shape)):
        next_point = shape[(i+1) % len(shape)]
        dist_to_next_point = math.dist(current_point, next_point)
        
        while dist_to_next_point >= remaining_distance:
            fraction = remaining_distance / dist_to_next_point
            new_point = interpolate(current_point, next_point, fraction)
            result.append(new_point)
            current_point = new_point
            
            dist_to_next_point -= remaining_distance
            remaining_distance = segment_length
            
            if len(result) == y:
                return result
        
        remaining_distance -= dist_to_next_point
        current_point = next_point
    
    # Add the last point to the result
    while len(result) < y:
        result.append(shape[-1])
    
    return result



def highest_gradient_variation(coords, y, y_plus):
    """
    Calculate the start and end indices of the sequence with the highest variation in gradient.

    Parameters:
    - coords (list): List of coordinate tuples representing points in a 2D space.
    - y (int): Length of the window to calculate the variation in gradient.
    - y_plus (int): Additional length to adjust the start index of the sequence.

    Returns:
    - tuple: A tuple containing two tuples. The first tuple represents the start and end indices of the sequence
        with the highest variation in gradient. The second tuple represents the adjusted start and end indices
        considering the additional length.

    """
    # Calculate gradients
    gradients = [(coords[(i+1) % len(coords)][1] - coords[i][1]) / 
                (coords[(i+1) % len(coords)][0] - coords[i][0]) 
                if coords[(i+1) % len(coords)][0] - coords[i][0] != 0 
                else np.inf for i in range(len(coords))]
    
    # Adjust the calculation for variation in gradients to correctly handle cyclic nature
    variations = []
    for start in range(len(coords)):
        variation = 0
        for i in range(y - 1): # Calculate variation in gradient for a window of length y
            current_idx = (start + i) % len(coords)
            next_idx = (start + i + 1) % len(coords)
            variation += abs(gradients[next_idx] - gradients[current_idx])
        variations.append(variation)
    
    # Find the start index of the sequence with the highest variation in gradient
    max_variation_start = np.argmax(variations)
    max_variation_start_2 = max_variation_start - y_plus
    
    # Correctly calculate end index considering cyclic nature
    max_variation_end = (max_variation_start + y - 1) % len(coords)
    max_variation_end_2 = (max_variation_start + y + y_plus - 1) % len(coords)

    return (max_variation_start, max_variation_end), (max_variation_start_2, max_variation_end_2)


def generate_closed_soft_body_from_list_and_point(space, center_loc, radial_spring_stiffness, circ_spring_stiffness, spring_damping, ball_rad, ball_mass, point_list, leading_edge, shape_type = 3):
    """
    Generates a closed soft body in a physics space based on a list of points and a center location.

    Args:
        space (pymunk.Space): The physics space in which the soft body will be created.
        center_loc (tuple): The center location of the soft body.
        radial_spring_stiffness (float): The stiffness of the radial springs connecting the center to the balls.
        circ_spring_stiffness (float): The stiffness of the circular springs connecting the balls.
        spring_damping (float): The damping of the springs.
        ball_rad (float): The radius of the balls.
        ball_mass (float): The mass of the balls.
        point_list (list): A list of points defining the shape of the soft body.

    Returns:
        list: A list of pymunk.Body objects representing the balls in the soft body.
    """

    # Center the list of points around the center location by adding the x and y coordinates of the center to each point
    centered_lop = [(x[0]+center_loc[0], x[1]+center_loc[1]) for x in point_list]

    # Create the central body and assign its position
    central_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    central_body.position = center_loc
        
    
    # Add the balls and springs connecting the balls to the central body into the space
    balls = []
    for i in range(len(centered_lop)):
        
        # Set the position of the ball from the centered list of points
        ball_position = centered_lop[i]

        # Create the ball body, shape, and add them to the space
        ball_body = pymunk.Body(mass=ball_mass, moment=pymunk.moment_for_circle(1, 0, ball_rad))
        ball_body.position = ball_position
        ball_shape = pymunk.Circle(ball_body, ball_rad)

        # Set the elasticity and friction of the ball shape to 0
        ball_shape.elasticity = 0
        ball_shape.friction = 0

        # Set the collision type and filter of the ball shape
        ball_shape.collision_type = shape_type
        ball_shape.filter = pymunk.ShapeFilter(group=shape_type)

        space.add(ball_body, ball_shape)
        balls.append(ball_body)
        
        # set the attr if the ball is part of the leading edge
        # if leading edge [1] is greater than leading edge [0] then the leading edge wraps around to the start
        if leading_edge[1] > leading_edge[0]:
            if i >= leading_edge[0] and i <= leading_edge[1]:
                ball_body.is_leading_edge = True
            else:
                ball_body.is_leading_edge = False
        else:
            if i >= leading_edge[0] or i <= leading_edge[1]:
                ball_body.is_leading_edge = True
            else:
                ball_body.is_leading_edge = False

        # Create the central spring
        t_rad = math.dist(ball_body.position, central_body.position)
        central_spring = pymunk.DampedSpring(central_body, ball_body, (0, 0), (0, 0), t_rad, radial_spring_stiffness, spring_damping)
        space.add(central_spring)

    for i in range(len(balls)):
        next_i = (i + 1) % len(balls)  # Wrap around to the first ball
        dist = math.dist(balls[i].position, balls[next_i].position)

        # Create the neighbor spring
        neighbor_spring = pymunk.DampedSpring(balls[i], balls[next_i], (0, 0), (0, 0), dist*.75, circ_spring_stiffness, spring_damping)
        space.add(neighbor_spring)

    return balls


    

def generate_nucleus_points(num_sample, rad_range, interp_points, nuc_size_rad, scale_factor_elipse, nuc_center_loc, ball_rad, len_adder):

    # generate blob with default blob generator
    nuc_mem_new = blob_generator(num_sample=num_sample, rad_range=rad_range, interp_points=interp_points, xy_trans=0)
    # get the max distance from the center of the nuc to the edge
    max_distance_nuc_mem = max(math.sqrt(x**2 + y**2) for x, y in nuc_mem_new)
    # get the scaling factor for the nuc mem, based off the max distance and the desired radius of the nuc
    mult_factor = nuc_size_rad / max_distance_nuc_mem
    # get a random scale factor for the elipse
    scale_factor_elipse = random.uniform(0.4,0.9)
    # scale the nuc mem points
    nuc_mem_new_scaled = [(x * mult_factor, y * mult_factor*scale_factor_elipse) for x, y in nuc_mem_new]
    # interpolate the points to get a more consistent spacing between points
    nuc_mem_new_scaled_interp = interpolate_points_dist(nuc_mem_new_scaled, ((ball_rad * 2) + len_adder))
    
    # get a random rotation for the nuc mem
    theta_degrees = random.randint(0,360)
    theta_radians = radians(theta_degrees)
    rotation_point = (0, 0)

    # Rotate each point around the rotation point by theta degrees
    rotated_points = [
        (
            (x - rotation_point[0]) * cos(theta_radians) - (y - rotation_point[1]) * sin(theta_radians) + rotation_point[0],
            (x - rotation_point[0]) * sin(theta_radians) + (y - rotation_point[1]) * cos(theta_radians) + rotation_point[1]
        )
        for x, y in nuc_mem_new_scaled_interp
    ]
    
    # get the distance from each point to the center
    point_to_center_dist = [math.sqrt((x - rotation_point[0])**2 + (y - rotation_point[1])**2) + random.uniform(-5,5) for x, y in rotated_points]
    # shift the points to the center of the screen
    nuc_mem_new_scaled_interp_shifted = [(x + nuc_center_loc[0], y + nuc_center_loc[1]) for x, y in rotated_points]

    # return the list of points, and a list of the distance of the said points to the center of the 
    return nuc_mem_new_scaled_interp_shifted, point_to_center_dist



def nuc_chain_from_points(nuc_mem_new_scaled_interp_shifted, mito_ball_radius, ball_mass_mult, point_to_center_dist, center_loc, space, num_tendrils, tendril_length_range, ball_rad,  perturber_obj_template, central_spring_stiffness = 2000):
    
    # extra store for the circular chain balls and the tendril balls, to be used for the peri_nuclear targeting chains, will be added as global var to the goal class
    nuc_circle_store = []
    nuc_internal_tendril_store = []


    # store for chain ball objects for func return (used to instatiate the peri chain object)
    nuc_ball_list = []
    # store for chain link objects for func return (used to instatiate the peri chain object)
    t_link_list = []


    # create the nucleus chain from the points list
    for i in range(len(nuc_mem_new_scaled_interp_shifted)):
        x, y = nuc_mem_new_scaled_interp_shifted[i]
        ball_obj = add_ball(space, loc_x = x, loc_y = y, radius = mito_ball_radius, mass = (mito_ball_radius*ball_mass_mult), elasticity=0, friction = 0, perturber_obj_template=perturber_obj_template)
        nuc_ball_list.append(ball_obj)
        nuc_circle_store.append(ball_obj)

    # create the central body for the nucleus chain to tendril to
    central_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    central_body.position = center_loc
    
    # create the links between the balls in the nucleus chain
    for i in range(0, len(nuc_ball_list)):
                
        body_a = nuc_ball_list[i]
        body_b = nuc_ball_list[(i+1) % len(nuc_ball_list)]

        link_obj = add_offset_pin_joint(space, body_a, body_b)

        t_link_list.append([body_a, body_b, link_obj])
        
        central_spring = pymunk.DampedSpring(central_body, body_a, (0, 0), (0, 0), point_to_center_dist[i], central_spring_stiffness, 0)
        space.add(central_spring)
        
        
    # randomly sample the balls in the nucleus ring, to add internal tendril chains to 
    tendril_list = random.sample(nuc_ball_list, num_tendrils)
    
    # create the internal tendril chains
    for each in tendril_list:
        
        # set the body_a to the current ball in the tendril list
        body_a = each
        
        # get a random length for the tendril chain len range (tendril_length_range)
        #curr_tendril_length = random.randint(tendril_length_range[0], tendril_length_range[1])
        curr_tendril_length = int(random.betavariate(2,8)*tendril_length_range[1]+tendril_length_range[0])
        
        # create the tendril chain iterativly
        for i in range(curr_tendril_length):
            
            # get a position radius 3 away from the body towards the center
            newx, newy = body_a.position - (body_a.position - central_body.position).normalized() * ball_rad
            # add the ball to space
            ball_obj = add_ball(space, loc_x = newx, loc_y = newy , radius = mito_ball_radius, mass = (mito_ball_radius*ball_mass_mult), elasticity=0, friction = 0, perturber_obj_template=perturber_obj_template)
            
            # add the ball to the nuc_ball_list and the nuc_internal_tendril_store
            nuc_ball_list.append(ball_obj)
            nuc_internal_tendril_store.append(ball_obj)
            
            # 
            link_obj = add_offset_pin_joint(space, body_a, ball_obj)
            t_link_list.append([body_a, ball_obj, link_obj])
            
            body_a = ball_obj


    return nuc_ball_list, t_link_list, nuc_circle_store, nuc_internal_tendril_store
