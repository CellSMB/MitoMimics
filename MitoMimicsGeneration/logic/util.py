# ------------ Imports --------------
import math
import random

import numpy as np
import pymunk
import pymunk.pygame_util

import copy

import os

# -------------------------------------------------------------------------------------
#                             basic pygame/pymunk utils
# -------------------------------------------------------------------------------------




def add_ball(
    space, loc_x, loc_y, radius, mass=1, elasticity=0, friction=0, override_inertia=None, perturber_obj_template=None
):
    """
    Adds a ball to a pymunk space with specified properties.

    Parameters:
    - space: The pymunk Space to which the ball will be added.
    - loc_x: The x-coordinate of the ball's initial position.
    - loc_y: The y-coordinate of the ball's initial position.
    - radius: The radius of the ball.
    - mass (optional): The mass of the ball. Defaults to 1.
    - elasticity (optional): The elasticity of the ball's surface. Defaults to 0.leading_edges to 0.
    - override_inertia (optional): If provided, this value will override the calculated moment of inertia for the ball.

    Returns:
    The pymunk Body object representing the ball, with the shape stored as an attribute.
    """

    # Calculate the moment of inertia for the ball, or use the override value if provided
    inertia = (
        override_inertia
        if override_inertia is not None
        else pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    )

    # Create the body with the specified mass and inertia, then define its shape as a circle
    body = pymunk.Body(mass, inertia)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = elasticity  # Set the elasticity for collision response
    shape.friction = friction  # Set the friction to affect movement interactions

    # Set the initial position of the body
    body.position = (loc_x, loc_y)

    # Set the objects collision type (type one is reserved specifically for ball objects)
    shape.collision_type = 1

    # Add both the body and its shape to the given space
    space.add(body, shape)

    # Store the shape in the body object for easy access and set a placeholder for chain ID
    body.store_shape = shape
    body.store_chain = None
    
    if perturber_obj_template is not None:

    
        perturber = copy.deepcopy(perturber_obj_template)
        perturber.reset_val()
        body.store_perturber = perturber
    

    return body



# could refactor this so that the points are based off the ball radius, using only the collision point for direction.
# then the ball radius can be used as a mult of the unit vector to get the pl,ace to place the pin joint
# is this better?
def add_offset_pin_joint(
    space, body_a, body_b, offset_a_frac=0.5, offset_b_frac=0.5, len_adder=1
):
    """
    Creates and adds an offset pin joint between two bodies in a pymunk space.

    Parameters:
    - space: The pymunk Space where the pin joint will be added.
    - body_a: The first body to be joined.
    - body_b: The second body to be joined.
    - offset_a_frac (optional): Fraction to calculate the offset for body_a from the midpoint. Defaults to 0.5.
    - offset_b_frac (optional): Fraction to calculate the offset for body_b from the midpoint. Defaults to 0.5.
    - len_adder (optional): Additional length added to the distance between the joint points. Defaults to 2.

    Returns:
    The pymunk PinJoint object created and added to the space.
    """

    # Calculate the midpoint and offsets for both bodies
    apos, bpos = body_a.position, body_b.position
    mid_point = (apos + bpos) / 2
    offset_point_a = (apos + mid_point) * offset_a_frac
    offset_point_b = (bpos + mid_point) * offset_b_frac

    # Convert offsets to local coordinates for each body
    local_offset_a = body_a.world_to_local(offset_point_a)
    local_offset_b = body_b.world_to_local(offset_point_b)

    # Create and configure the pin joint
    pin_joint = pymunk.PinJoint(body_a, body_b, local_offset_a, local_offset_b)
    pin_joint.distance += len_adder

    # Add the pin joint to the space
    space.add(pin_joint)

    return pin_joint


# -------------------------------------------------------------------------------------
#                             Generic simulation functions
# -------------------------------------------------------------------------------------


class Perturber:
    def __init__(
        self,
        val_low,
        val_high,
        perturb_max_mag,
        period_low=0,
        period_high=0,
        initiate_from_val=None,
        bounce_off_hl=True,
        allow_oob_recov=True,
    ):
        """
        Initializes the Perturber with given parameters, setting up the initial value and perturbation behavior.

        Parameters:
        - val_low: The lower bound for the value.
        - val_high: The upper bound for the value.
        - perturb_max_mag: The maximum magnitude of perturbation.
        - period_low (optional): The minimum period after which to apply perturbation. Defaults to 0.
        - period_high (optional): The maximum period after which to apply perturbation. Defaults to 0.
        - initiate_from_val (optional): The initial value from which to start perturbations. If None, a random value within bounds is chosen. Defaults to None.
        - bounce_off_hl (optional): Whether the value should bounce off the high and low limits instead of stopping or going out of bounds. Defaults to True.
        - allow_oob_recov (optional): Whether to allow recovery if the initial value is set out of bounds. Defaults to True.

        Raises:
        - Exception: If the perturbation magnitude is too large or if an out-of-bounds initial value is provided without recovery allowed.
        """

        # Check if the perturbation magnitude is too large
        if perturb_max_mag >= ((val_high - val_low) / 2) * 0.9999:
            raise Exception(
                "Trying to perturb by too much, could cause out of bounds values to be returned"
            )

        # Initialize current value, checking for out-of-bounds issues if necessary
        if initiate_from_val is not None:
            if (
                initiate_from_val < val_low or initiate_from_val > val_high
            ) and not allow_oob_recov:
                raise Exception(
                    "Tried to initiate perturber from out-of-bounds fixed value but allow_oob_recov was set to False"
                )
            self.curr_val = initiate_from_val
        else:
            self.curr_val = random.uniform(val_low, val_high)

        # Set other attributes
        self.countdown = random.randint(period_low, period_high)
        self.val_high = val_high
        self.val_low = val_low
        self.perturb_max_mag = perturb_max_mag
        self.period_low = period_low
        self.period_high = period_high
        self.bounce_off_hl = bounce_off_hl
        self.allow_oob_recov = allow_oob_recov
        
        
    def reset_val(self):
        """
        Resets the current value to a random value within the bounds.
        """
        self.curr_val = random.uniform(self.val_low, self.val_high)
    

    def update(self):
        """
        Updates the value based on the perturbation rules. If the countdown reaches zero, perturbs the value
        and resets the countdown. Otherwise, decrements the countdown.

        Returns:
        - The current value after potential perturbation.
        """

        # Debugging output for countdown

        # Apply perturbation if countdown is zero
        if self.countdown <= 0:
            # Handle edge cases for bouncing off limits
            if self.curr_val >= self.val_high:
                self.curr_val += random.uniform(-self.perturb_max_mag, -0.0001)
            elif self.curr_val <= self.val_low:
                self.curr_val += random.uniform(0.0001, self.perturb_max_mag)
            else:
                # Apply random perturbation
                perturb_val = random.uniform(
                    -self.perturb_max_mag, self.perturb_max_mag
                )
                curr_prop = self.curr_val + perturb_val

                # Adjust for out-of-bounds result based on settings
                if curr_prop >= self.val_high or curr_prop <= self.val_low:
                    if self.bounce_off_hl:
                        curr_prop = self.curr_val - perturb_val
                    else:
                        curr_prop = (
                            self.val_high - 0.0001
                            if curr_prop >= self.val_high
                            else self.val_low + 0.0001
                        )

                self.curr_val = curr_prop

            # Debugging output for current value

            # Reset the countdown
            self.countdown = random.randint(self.period_low, self.period_high)
        else:
            # Decrement countdown if not yet at zero
            self.countdown -= 1

        # Debugging output for current value

        return self.curr_val
    
    # gets the val without updating
    def get_val_without_update(self):
        
        return self.curr_val
    
    

class Brightness_Peturber:
    def __init__(
        self,
        max_val = 1,
        min_val = 0.1,
        local_peturb_range_plus_minus = 0.15,
        peturb_max_mag = 0.1,        
        period = 16

    ):

        # Check if the perturbation magnitude is too large
        if peturb_max_mag >= local_peturb_range_plus_minus:
            raise Exception(
                "Trying to perturb by too much, could cause out of bounds values to be returned"
            )
            
        self.max_val = max_val
        self.min_val = min_val
        self.local_peturb_range_plus_minus = local_peturb_range_plus_minus
        self.perturb_max_mag = peturb_max_mag
        self.period = period
            

        if min_val + local_peturb_range_plus_minus > max_val - local_peturb_range_plus_minus:
            raise Exception(
                "peturb range to high, not given space to perturb in the total max and min vals"
            )

        # needed dired vars for updates
        self.my_mid_point = random.uniform(self.min_val + self.local_peturb_range_plus_minus, self.max_val - self.local_peturb_range_plus_minus)
        self.my_max_point = self.my_mid_point + self.local_peturb_range_plus_minus
        self.my_min_point = self.my_mid_point - self.local_peturb_range_plus_minus
        self.curr_val = random.uniform(self.my_min_point, self.my_max_point)
        self.countdown = self.period
        
        
    def reset_val(self):

        self.my_mid_point = random.uniform(self.min_val + self.local_peturb_range_plus_minus, self.max_val - self.local_peturb_range_plus_minus)
        self.my_max_point = self.my_mid_point + self.local_peturb_range_plus_minus
        self.my_min_point = self.my_mid_point - self.local_peturb_range_plus_minus
        self.curr_val = random.uniform(self.my_min_point, self.my_max_point)
        self.countdown = self.period


    def update(self):

        # Apply perturbation if countdown is zero
        if self.countdown <= 0:

            # Apply random perturbation
            perturb_val = random.uniform(
                -self.perturb_max_mag, self.perturb_max_mag
            )
            
            curr_prop = self.curr_val + perturb_val

            # Adjust for out-of-bounds result based on settings
            if curr_prop >= self.my_max_point or curr_prop <= self.my_min_point:
                curr_prop = self.curr_val - perturb_val


            self.curr_val = curr_prop


            # Reset the countdown
            self.countdown = self.period
        else:
            # Decrement countdown if not yet at zero
            self.countdown -= 1


        return self.curr_val
    
    def get_val_without_update(self):
        
        return self.curr_val



def apply_brownian_motion(body, strength_low, strength_high):
    """
    Applies Brownian motion to a given body.

    Parameters:
    - body: The body to apply Brownian motion to.
    - strength_low: The lower bound of the strength amount.
    - strength_high: The upper bound of the strength amount.
    """
    # get strength amount
    apply_strength = random.uniform(strength_low, strength_high)
    # Generate a random direction by choosing an angle
    angle = random.uniform(0, 2 * math.pi)
    # Create a force vector with the given magnitude in the random direction
    force = pymunk.Vec2d(math.cos(angle), math.sin(angle)) * apply_strength
    # Apply the force to the center of mass of the body
    body.apply_impulse_at_local_point(force)
    
    
    
    

def repel_chains_from_point(chain_obj, point, max_dist, min_dist, max_force, min_force):
    """
    Repels chains from a given point within a specified max distance range, with a linear force function between the min and max distances, using the max and min forces respectivly

    Args:
        chain_manager (ChainManager): The chain manager object that contains the chains.
        point (Vector2D): The point from which the chains should be repelled.
        max_dist (float): The maximum distance at which the repulsion force is applied.
        min_dist (float): The minimum distance at which the repulsion force is applied.
        max_force (float): The maximum force to be applied for repulsion, applied at the minimum distance.
        min_force (float): The minimum force to be applied for repulsion, applied at the maximum distance.

    Returns:
        None
    """

        
    for body in chain_obj.get_chain_bodies():
        # Calculate the distance from the body to the point
        body_distance = body.position.get_distance(point)
        # Check if the body is within the specified distance
        if body_distance <= max_dist:
            # Calculate the direction from the body to the point
            direction = body.position - point
            # Normalize the direction vector
            normalized_direction = direction.normalized()
            # Apply a force in the opposite direction to repel the body from the point
            force_scaled = min_force + (max_force - min_force) * (max_dist - body_distance) / (max_dist - min_dist)
            
            
            body.apply_force_at_world_point(normalized_direction * force_scaled, body.position)



# -------------------------------------------------------------------------------------
#                        Functions for storing simulation state
# -------------------------------------------------------------------------------------


class Collision_Location_Store:
    """
    A class that stores collision locations and the remaining time to display them.
    Also tracks which chains fused, for saving purposes.

    Attributes:
        list_of_collisions (list): A list of collision locations and their remaining time.
    """
    

    def __init__(self):
        """
        Initializes a Collision_Location_Store object.

        The list_of_collisions attribute is initialized as an empty list.
        """
        self.list_of_collisions = []
        self.list_of_fissions = []
        self.save_list = []
        self.save_list2 = []
        self.g_counter = None
        
    def set_g_counter(self, g_counter):
            
        self.g_counter = g_counter
        
    def add_fusion(self, chain_a_id, chain_b_id, collision_location, chain_c_id, ball_a, ball_b, pos1, pos2):
        
        self.save_list.append((self.g_counter, [chain_a_id, chain_b_id, collision_location, chain_c_id, ball_a, ball_b, pos1, pos2]))
        
    def add_fission(self, chain_a_id, collision_location, chain_b_id, chain_c_id, ball_a, ball_b, pos1, pos2):
        
        self.save_list2.append((self.g_counter, [chain_a_id, collision_location, chain_b_id, chain_c_id, ball_a, ball_b, pos1, pos2]))

    def add_collision(self, collision_location):
        """
        Adds a collision location to the list_of_collisions.

        Args:
            collision_location: The collision location to be added.

        The collision location is appended to the list_of_collisions along with a default remaining time of 15 (in frames).
        """
        self.list_of_collisions.append([collision_location, 15])
        

    def add_fission_loc(self, collision_location):
        """
        Adds a collision location to the list_of_collisions.

        Args:
            collision_location: The collision location to be added.

        The collision location is appended to the list_of_collisions along with a default remaining time of 15 (in frames).
        """
        self.list_of_fissions.append([collision_location, 15])

    def update(self):
        """
        Updates the remaining time for each collision location.

        The remaining time for each collision location is decreased by 1.
        Any collision location with a remaining time of 0 or less is removed from the list_of_collisions.
        """
        for collision in self.list_of_collisions:
            collision[1] -= 1

        self.list_of_collisions = [
            collision for collision in self.list_of_collisions if collision[1] > 0
        ]
        
        for collision in self.list_of_fissions:
            collision[1] -= 1

        self.list_of_fissions = [
            collision for collision in self.list_of_fissions if collision[1] > 0
        ]

    def get_collision_list(self):
        """
        Returns a list of collision locations.

        Returns:
            A list of collision locations extracted from the list_of_collisions.
        """
        return [x for x in self.list_of_collisions]
    
    def get_fussion_list(self):
        """
        Returns a list of collision locations.

        Returns:
            A list of collision locations extracted from the list_of_collisions.
        """
        return [x for x in self.list_of_fissions]




def return_graphs(space, chain_manager):
    """
    Returns the current state of the simulation as a numpy array for storage.

    Parameters:
    - space: The pymunk Space object containing the simulation.
    - chain_manager: The ChainManager object containing the simulation's chains.

    Returns:
    - A numpy array containing the current state of the simulation.
    """
    # Initialize the array
    state = dict()

    # Add the positions of all bodies in the space
    for chain in chain_manager.chains:
        temp_graph = chain.return_chain_state()
        state[id(chain)] = (temp_graph.copy(), chain.get_goal().goal_type)

    return state


# -------------------------------------------------------------------------------------
#                        Functions instantiating the mitochondria locationns
# -------------------------------------------------------------------------------------


def is_point_outside_polygon(point, polygon):
    """
    Check if a point is outside a closed polygon defined by a list of Cartesian points.

    :param point: Tuple (x, y) representing the point to check.
    :param polygon: List of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] defining the closed polygon.
    :return: True if the point is outside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    outside = False

    j = n - 1
    for i in range(0, n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if point is on the same level as the vertex, and if the segment is crossing the ray
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            outside = not outside

        j = i

    return outside


def sample_points_soft_donut(inner_radius, outer_shape, screen_size=(1500, 1500)):
    """
    Generate random points within a donut-shaped region.

    Parameters:
    - inner_radius (float): The radius of the inner circle of the donut.
    - outer_shape (list of tuples): The vertices of the outer shape defining the donut.
    - screen_size (tuple, optional): The size of the screen or canvas to generate points on. Defaults to (1500, 1500).

    Returns:
    - x (float): The x-coordinate of the generated point.
    - y (float): The y-coordinate of the generated point.
    """

    point_valid = False

    # Generate random points until a valid point is found
    while not point_valid:
        # Generate a random point within the screen size
        test_point = (
            random.uniform(0, screen_size[0]),
            random.uniform(0, screen_size[1]),
        )

        # Check if the point is within the inner circle, and if so, continue to the next iteration
        if (
            math.dist((screen_size[0] / 2, screen_size[1] / 2), test_point)
            < inner_radius
        ):
            continue

        # Check if the point is within the outer shape
        if is_point_outside_polygon(test_point, outer_shape):
            point_valid = True

    return test_point[0], test_point[1]






def sample_random_points_in_disk(
    inner_radius, outer_radius, offset_x=1136, offset_y=1136
):
    """
    Generate random points within a disk.

    Parameters:
    - inner_radius (float): The inner radius of the disk.
    - outer_radius (float): The outer radius of the disk.
    - offset_x (float, optional): The x-coordinate offset for the generated points. Default is 1136.
    - offset_y (float, optional): The y-coordinate offset for the generated points. Default is 1136.

    Returns:
    - x (float): The x-coordinate of the generated point.
    - y (float): The y-coordinate of the generated point.
    """
    # Generate a random angle
    theta = np.random.uniform(0, 2 * np.pi)

    # Generate a random radius with proper weighting
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2))

    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x + offset_x, y + offset_y


class Text_counter:
    
    def __init__(self, start, diff):
            
            self.curr = start
            self.diff = diff

    def update(self):
        
        self.curr += self.diff
        
        return self.curr - self.diff
    
    def return_curr(self):
        
        return self.curr
        
        
def create_directory_if_empty_or_not_exists(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Check if the directory is empty; listdir() returns a list of the entries in the directory.
        # If the list is empty, it means the directory is empty.
        if not os.listdir(path):
            # The directory exists and is empty, do not return an error.
            return "Directory exists and is empty, no action needed."
        else:
            # The directory exists and is not empty, return an error.
            return "Error: Directory exists and is not empty."
    else:
        # The directory does not exist, try to create it.
        try:
            os.makedirs(path)
            return "Directory created successfully."
        except OSError as e:
            # In case the parent directory does not exist or any other OSError, return an error.
            return f"Error: Failed to create the directory. {e}"