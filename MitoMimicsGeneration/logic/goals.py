# imports
import random
from abc import ABC, abstractmethod
from itertools import repeat

import networkx as nx
import numpy as np

from .util import *

import copy

# -------------------------------------------------------------------------------------
#                             Goal Types
# -------------------------------------------------------------------------------------

# REFACTOR TO HAVE NO SPACE OR CHAIN MANAGER CALLS DUE TO GLOBAL ACCESS


class Goal(ABC):
    
    
    # class variables, to be set pre simulation start
    space_access = None
    chain_manager_access = None
    outer_mem_access = None
    nuc_mem_access = None
    random_choice_dict = None
    
    
    @classmethod
    def set_space_access(cls, space):
        """
        Set the space access class variable.

        Args:
            space (object): The space object.
        """
        cls.space_access = space

    @classmethod
    def set_chain_manager_access(cls, chain_manager):
        """
        Set the chain manager access class variable.

        Args:
            chain_manager (object): The chain manager object.
        """
        cls.chain_manager_access = chain_manager

    @classmethod
    def set_outer_mem_access(cls, outer_mem):
        """
        Set the outer memory access class variable.

        Args:
            outer_mem (object): The outer memory object.
        """
        cls.outer_mem_access = outer_mem

    @classmethod
    def set_nuc_mem_access(cls, nuc_mem):
        """
        Set the nuclear memory access class variable.

        Args:
            nuc_mem (object): The nuclear memory object.
        """
        cls.nuc_mem_access = nuc_mem
        
    @classmethod
    def set_random_choice_dict(cls, random_choice_dict):
        """
        Set the random choice dict class variable.

        Args:
            random_choice_dict (dict): The random choice dict.
        """
        cls.random_choice_dict = random_choice_dict
    
    
    def __init__(self, chain_obj):
        self.goal_type = None
        self.exit_countdown = None
        self.chain_obj = chain_obj
        

    @abstractmethod
    def get_inherited_goal(self, new_chain_obj):
        """
        Get the inherited goal for the given space and chain manager.

        Args:
            space (object): The space object.
            chain_manager (object): The chain manager object.

        Returns:
            object: The inherited goal object.
        """
        pass


    @abstractmethod
    def should_exit(self) -> bool:
        """
        Check if the goal should be exited.

        Returns:
            bool: True if the goal should be exited, False otherwise.
        """
        pass
    
    
    def update_goal_state(self):
        """
        Update the goal state

        Args:
            chain_manager (object): The chain manager object.
        """
        
        if self.exit_countdown is not None:
        
            self.exit_countdown -= 1
            
        


class Goal_idle(Goal):
    
    exit_range = None
    new_goal_idle_factor = 1
    new_goal_leading_edge_factor = 1
    
    def __init__(self, chain_obj):
        super().__init__(chain_obj=chain_obj)

        # mandatory overrides
        self.goal_type = "idle"
        self.exit_countdown = random.randint(self.exit_range[0], self.exit_range[1])


    def should_exit(self):
        
        if self.exit_countdown <= 0:
            return True
        else:
            return False
        

    def get_inherited_goal(self, new_chain_obj):
        
        goal_list = []
        
        goal_list.append([self.new_goal_idle_factor, Goal_idle(new_chain_obj)])
        
        goal_list.append([self.new_goal_leading_edge_factor, Goal_leading_edge(new_chain_obj)])
                
        return goal_list



# special goal class just used on initialization
class Goal_peri_init(Goal):
    
    peri_init_loc_class_var = None
    exit_countdown_init = None
    
    def __init__(self, chain_obj):
        
        super().__init__(chain_obj=chain_obj)

        # mandatory overrides
        self.goal_type = "peri_init"
        # extra long exit countdown, as they are only used on initialization
        self.exit_countdown = self.exit_countdown_init  
        
        # goal specific vars
        self.peri_init_loc = self.peri_init_loc_class_var

    def should_exit(self):
        
        if self.exit_countdown <= 0:
            return True
        else:
            return False

    def get_inherited_goal(self, new_chain_obj):
    
        # defaults to random goal if it fusses with something on the way to the nuc. 
        goal_list = []
                        
        return goal_list



class Goal_leading_edge(Goal):
    
    perturber_obj_template = None
    leading_edge_exit_range = (300,700)
    distance_till_close = 40
    sepcial_same_goal_fusion_factor = 1
    
    def __init__(self, chain_obj):
        
        super().__init__(chain_obj=chain_obj)

        # mandatory overrides
        self.goal_type = "leading_edge"
        self.exit_countdown = random.randint(self.leading_edge_exit_range[0], self.leading_edge_exit_range[1])
        
        # PART THAT ACCTUALLY MATTERS, EVERYTHING ELSE CAN BE REUSED FOR THE SIMILAR CLASSES
        # goal specific vars   
        # get a random leading edge ball, set it as the object isntance variable self.target ball      
        leading_edge_balls = [x for x in Goal.outer_mem_access if x.is_leading_edge]
        leading_edge_idx = random.randint(0, len(leading_edge_balls) - 1)
        self.target_ball = leading_edge_balls[leading_edge_idx]
        
        # get the closest head ball to the target ball, set it as the object isntance variable self.lead_ball
        # if there are no head balls, set the lead ball to a random body
        heads = self.chain_obj.return_nodes_w_degree_one()
        if len(heads) == 0:
            self.lead_ball = random.choice(self.chain_obj.get_chain_bodies())
        else:
            self.lead_ball = min(heads, key=lambda ball: np.linalg.norm(np.array(ball.position) - np.array(self.target_ball.position)))
        
        # instantiate the perturber
        self.perturber = copy.deepcopy(self.perturber_obj_template)
        self.perturber.reset_val()


    def should_exit(self):
        
        # check if any of the bodies are close to the target location        
        for body in self.chain_obj.get_chain_bodies():
            if (
                np.linalg.norm(np.array(body.position) - np.array(self.target_ball.position))
                < self.distance_till_close
            ):
                return True
            
        if self.exit_countdown <= 0:
            return True
        else:
            return False


    def get_inherited_goal(self, new_chain_obj):
        
        goal_list = []
        
        # special fuse case, if it is fused, adds a leading edge goal with the same target ball
        t_goal = Goal_leading_edge(new_chain_obj)
        t_goal.target_ball = self.target_ball
        
        curr_mult = self.get_force_multiplier()
        t_goal.perturber = copy.deepcopy(self.perturber_obj_template)
        t_goal.perturber.reset_val()
        t_goal.curr_val = curr_mult
        
        
        goal_list.append([self.sepcial_same_goal_fusion_factor, t_goal])

    
        return goal_list
    
    
    # gets the force mult from the stored perturber object
    def get_force_multiplier(self):
        
        r_val = self.perturber.update()
        
        return r_val
    

    
    
class Goal_peri_migr(Goal):
    
    perturber_obj_template = None
    leading_edge_exit_range = (300,700)
    distance_till_close = 40
    sepcial_same_goal_fusion_factor = 1
    
    def __init__(self, chain_obj):
        
        super().__init__(chain_obj=chain_obj)

        # mandatory overrides
        self.goal_type = "peri_migr"
        self.exit_countdown = random.randint(self.leading_edge_exit_range[0], self.leading_edge_exit_range[1])
        
        # PART THAT ACCTUALLY MATTERS, EVERYTHING ELSE CAN BE REUSED FOR THE SIMILAR CLASSES
        # goal specific vars   
        # get a random leading edge ball, set it as the object isntance variable self.target ball  
        
        peri_main_balls_list = self.nuc_mem_access
        ball_idx = random.randint(0, len(peri_main_balls_list) - 1)
        self.target_ball = peri_main_balls_list[ball_idx]
        
        # get the closest head ball to the target ball, set it as the object isntance variable self.lead_ball
        # if there are no head balls, set the lead ball to a random body
        heads = self.chain_obj.return_nodes_w_degree_one()
        if len(heads) == 0:
            self.lead_ball = random.choice(self.chain_obj.get_chain_bodies())
        else:
            self.lead_ball = min(heads, key=lambda ball: np.linalg.norm(np.array(ball.position) - np.array(self.target_ball.position)))
        
        # instantiate the perturber
        self.perturber = copy.deepcopy(self.perturber_obj_template)
        self.perturber.reset_val()
        


    def should_exit(self):
        
        # check if any of the bodies are close to the target location        
        for body in self.chain_obj.get_chain_bodies():
            if (
                np.linalg.norm(np.array(body.position) - np.array(self.target_ball.position))
                < self.distance_till_close
            ):
                return True
            
        if self.exit_countdown <= 0:
            return True
        else:
            return False


    def get_inherited_goal(self, new_chain_obj):
        
        goal_list = []
        
        # special fuse case, if it is fused, adds a leading edge goal with the same target ball
        t_goal = Goal_peri_migr(new_chain_obj)
        t_goal.target_ball = self.target_ball
        
        curr_mult = self.get_force_multiplier()
        t_goal.perturber = copy.deepcopy(self.perturber_obj_template)
        t_goal.perturber.reset_val()
        t_goal.curr_val = curr_mult
        
        goal_list.append([self.sepcial_same_goal_fusion_factor, t_goal])
    
        return goal_list
    
    
    # gets the force mult from the stored perturber object
    def get_force_multiplier(self):
        
        r_val = self.perturber.update()
        
        return r_val
    
    
    
class Goal_random_target(Goal):
    
    perturber_obj_template = None
    random_target_exit_range = (300,700)
    distance_till_close = 150
    sepcial_same_goal_fusion_factor = 1
    
    def __init__(self, chain_obj):
        
        super().__init__(chain_obj=chain_obj)

        # mandatory overrides
        self.goal_type = "random_target"
        self.exit_countdown = random.randint(self.random_target_exit_range[0], self.random_target_exit_range[1])
        
        # PART THAT ACCTUALLY MATTERS, EVERYTHING ELSE CAN BE REUSED FOR THE SIMILAR CLASSES
        # goal specific vars   
        # get a random leading edge ball, set it as the object isntance variable self.target ball  
        
        non_leading_edge_balls = [x for x in Goal.outer_mem_access if not x.is_leading_edge]
        target_idx = random.randint(0, len(non_leading_edge_balls) - 1)
        self.target_ball = non_leading_edge_balls[target_idx]
        
        # get the closest head ball to the target ball, set it as the object isntance variable self.lead_ball
        # if there are no head balls, set the lead ball to a random body
        heads = self.chain_obj.return_nodes_w_degree_one()
        if len(heads) == 0:
            self.lead_ball = random.choice(self.chain_obj.get_chain_bodies())
        else:
            self.lead_ball = min(heads, key=lambda ball: np.linalg.norm(np.array(ball.position) - np.array(self.target_ball.position)))
        
        # instantiate the perturber
        self.perturber = copy.deepcopy(self.perturber_obj_template)
        self.perturber.reset_val()
        


    def should_exit(self):
        
        # check if any of the bodies are close to the target location        
        for body in self.chain_obj.get_chain_bodies():
            if (
                np.linalg.norm(np.array(body.position) - np.array(self.target_ball.position))
                < self.distance_till_close
            ):
                return True
            
        if self.exit_countdown <= 0:
            return True
        else:
            return False


    def get_inherited_goal(self, new_chain_obj):
        
        goal_list = []
        
        # special fuse case, if it is fused, adds a leading edge goal with the same target ball
        t_goal = Goal_peri_migr(new_chain_obj)
        t_goal.target_ball = self.target_ball
        
        curr_mult = self.get_force_multiplier()
        t_goal.perturber = copy.deepcopy(self.perturber_obj_template)
        t_goal.perturber.reset_val()
        t_goal.curr_val = curr_mult
        
        goal_list.append([self.sepcial_same_goal_fusion_factor, t_goal])
    
        return goal_list
    
    
    # gets the force mult from the stored perturber object
    def get_force_multiplier(self):
        
        r_val = self.perturber.update()
        
        return r_val


    
    

    
class Goal_peri(Goal):
    
    def __init__(self, chain_obj):
        super().__init__(chain_obj=chain_obj)

        # mandatory overrides
        self.goal_type = "peri"
        self.exit_countdown = None
        




    def should_exit(self):
        
        return False
        

    def get_inherited_goal(self, new_chain_obj):
        
        # no inherited goals, all fusions involving peri result in peri
        # this functionality is overridden in the decide_fussion_goal function
        None
    
    

    


# -------------------------------------------------------------------------------------
#                             Goal Functions
# -------------------------------------------------------------------------------------


# dict for chance range for the goals




def get_random_goal_instance(chain_obj):

    gcd = Goal.random_choice_dict

    total = sum(gcd.values())
    rand_val = random.randint(1, total)
    running_sum = 0
    goal_name = None
    
    for class_name, chance in gcd.items():
        
        running_sum += chance
        
        if rand_val <= running_sum:
            
            goal_name = class_name
            break
            

    if goal_name == 'idle':
        n_goal = Goal_idle(chain_obj)
        
    elif goal_name == 'random_target':
        n_goal = Goal_random_target(chain_obj)

    elif goal_name == 'leading_edge':
        n_goal = Goal_leading_edge(chain_obj)
        
    elif goal_name == 'peri_migr':
        n_goal = Goal_peri_migr(chain_obj)

    return n_goal




def decide_fusion_goal(chain_a, chain_b, new_chain):
    
    # returns the goal object and the impact factor
    if chain_a.get_goal().goal_type == "peri" or chain_b.get_goal().goal_type == "peri":
        return Goal_peri(new_chain)  
    

    # default is to return the goal of the larger chain and the impact factor of 1
    a_goal_list = chain_a.get_goal().get_inherited_goal(new_chain)
    b_goal_list = chain_b.get_goal().get_inherited_goal(new_chain)
    
    goal_list = a_goal_list + b_goal_list
    
    goal_list.append([1, get_random_goal_instance(new_chain)])
    
    choice_list = []
    
    for goal in goal_list:
        choice_list += list(repeat(goal[1], goal[0]))
        
        
    goal_choice = random.choice(choice_list)
    
    
    return goal_choice






# -------------------------------------------------------------------------------------
#                             Per frame goal updater
# -------------------------------------------------------------------------------------





def manage_goals(chain_manager, space, center_loc, goal_manager_dict):
    

    
    for chain_obj in chain_manager.get_chain_list():
        
        curr_goal = chain_obj.get_goal()
        
        if curr_goal.goal_type == 'idle':
        
            
            for body in chain_obj.get_chain_bodies():
                apply_brownian_motion(body, goal_manager_dict['idle_brownian'][0], goal_manager_dict['idle_brownian'][1])
                
                
            repel_chains_from_point(chain_obj, center_loc, *goal_manager_dict['idle_repel_func_vars'])    

                
                
            if curr_goal.should_exit():
                
                
                n_goal = get_random_goal_instance(chain_obj)

                chain_obj.set_goal(n_goal)
                
        
        elif curr_goal.goal_type == 'peri':
            
            for body in chain_obj.get_chain_bodies():
                apply_brownian_motion(body, goal_manager_dict['peri_brownian'][0], goal_manager_dict['peri_brownian'][1])
                
                
            repel_chains_from_point(chain_obj, center_loc, *goal_manager_dict['peri_repel_func_vars'])    

        
            tips = chain_obj.return_nodes_w_degree_one()
            
            if len(tips) > 0:
                
                for tip in tips:
                    
                    # get the target for the tip, maintend in the chain manager object
                    # higher chance for target to be leading edge
                    # targets are swapped with low odds
                    if tip not in chain_manager.peri_leaf_target_dict.keys():
                        
                        #print(goal_manager_dict['perecent_peri_seeking_leading'])
                        seed3 = random.randint(0, 100)
                        if seed3 > goal_manager_dict['perecent_peri_seeking_leading']:
                            target = random.choice([x for x in Goal.outer_mem_access if not x.is_leading_edge]).position
                        else:
                            target = random.choice([x for x in Goal.outer_mem_access if x.is_leading_edge]).position
                        chain_manager.peri_leaf_target_dict[tip] = target
                    else:
                        
                        seed4 = random.uniform(0, 100)
                        if seed4 > 99.9:
                            #print('swaped')
                            seed3 = random.randint(0, 100)
                            if seed3 > goal_manager_dict['perecent_peri_seeking_leading']:
                                target = random.choice([x for x in Goal.outer_mem_access if not x.is_leading_edge]).position
                            else:
                                target = random.choice([x for x in Goal.outer_mem_access if x.is_leading_edge]).position
                            chain_manager.peri_leaf_target_dict[tip] = target
                        else:
                            target = chain_manager.peri_leaf_target_dict[tip]

                    # target = random.choice([x for x in Goal.outer_mem_access if not x.is_leading_edge]).position
                    #target = random.choice([x for x in Goal.outer_mem_access if x.is_leading_edge]).position

                    
                    # get the direction to the target
                    direction = target - tip.position
                    direction = direction.normalized()
                    
                    # get the force factor, scales with the number of tips to not be too strong
                    max_force = goal_manager_dict['tip_min_max_diff'][0]
                    min_force = goal_manager_dict['tip_min_max_diff'][1]
                    reduce_per_tip = goal_manager_dict['tip_min_max_diff'][2]
                    
                    force_factor = np.clip((max_force - math.sqrt(len(tips))* reduce_per_tip), min_force, max_force)
                    
                    # apply the force
                    force = direction * force_factor
                    tip.apply_force_at_world_point(force, tip.position)

        
        
                
        elif curr_goal.goal_type == 'peri_init':
        
        
            for body in chain_obj.get_chain_bodies():
                apply_brownian_motion(body, goal_manager_dict['peri_brownian'][0], goal_manager_dict['peri_brownian'][1])
                
            
            # moves the chain towards the peri_init using global position of the chain
            for body in chain_obj.get_chain_bodies():
                
                direction = pymunk.Vec2d(curr_goal.peri_init_loc[0], curr_goal.peri_init_loc[1]) - body.position
                direction = direction.normalized()
                force = direction * goal_manager_dict['peri_init_force']
                
                body.apply_force_at_world_point(force, body.position)
                
            if curr_goal.should_exit():
                
                n_goal = get_random_goal_instance(chain_obj)
                chain_obj.set_goal(n_goal)   
                
                
                
                
        elif curr_goal.goal_type == 'leading_edge' or curr_goal.goal_type == 'peri_migr' or curr_goal.goal_type == 'random_target':
        
        
            # apply brownian motion
            for body in chain_obj.get_chain_bodies():
                apply_brownian_motion(body, goal_manager_dict['seeker_brownian'][0], goal_manager_dict['seeker_brownian'][1])
        
            # repel the leading edge chains from the center so that they avoid the nuc
            if curr_goal.goal_type == 'leading_edge' or curr_goal.goal_type == 'random_target':
                repel_chains_from_point(chain_obj, center_loc, *goal_manager_dict['leading_target_repel_func_vars'])    

            # get the head ball and the target ball in the peri or leading edge goal
            leading_ball = curr_goal.lead_ball
            target_ball = curr_goal.target_ball
            
            # get the direction to the target
            direction = target_ball.position - leading_ball.position
            direction = direction.normalized()
            
            # get the number of balls in the chain
            num_balls = chain_obj.num_balls()
            
            # get the force multiplier from the perturber object in the goal object
            peterbur_mult = curr_goal.get_force_multiplier()
        
        
            total_force = (goal_manager_dict['seeker_base_force'] + goal_manager_dict['seeker_per_ball_force_mult'] * num_balls) * peterbur_mult
            
            
            # apply a decreasing force dependent on the balls distance from the leading ball
            for body in chain_obj.get_chain_bodies():
                if body != leading_ball:
                    
                    # get the distance from the leading ball (in num edges)
                    dist = nx.shortest_path_length(chain_obj.G, source=leading_ball, target=body)
                    
                    # scale the force down accordingly (linearly with num edges from leading ball)
                    force = direction * (total_force - ((total_force/num_balls) * dist) + (total_force/num_balls))
                    
                    # apply the force
                    body.apply_force_at_world_point(force, body.position)


            # apply increased force to leading ball
            leading_ball_force = direction * (total_force * peterbur_mult + (total_force/num_balls))
            leading_ball.apply_force_at_world_point(leading_ball_force, leading_ball.position)


            # check if the goal should exit
            if curr_goal.should_exit():
                
                
                # 70% chance to swap to the opposite goal, or 30% to swap to a random goal
                should_swap_to_oppisite = random.randint(0, 100)
                
                if should_swap_to_oppisite < goal_manager_dict['chance_to_swap']:
                    if curr_goal.goal_type == 'leading_edge' or curr_goal.goal_type == 'random_target':
                        n_goal = Goal_peri_migr(chain_obj)
                        chain_obj.set_goal(n_goal)
                    elif curr_goal.goal_type == 'peri_migr':
                        n_goal = Goal_leading_edge(chain_obj)
                        chain_obj.set_goal(n_goal)
                        
                    
                else:
                    n_goal = get_random_goal_instance(chain_obj)
                    chain_obj.set_goal(n_goal)

                

        curr_goal.update_goal_state()