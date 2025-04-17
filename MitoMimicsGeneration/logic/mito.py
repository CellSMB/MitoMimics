# ------------ Imports --------------
import random
from math import cos, radians, sin

import networkx as nx
import numpy as np
import pymunk

from .goals import decide_fusion_goal
from .goals import *
from .util import *
from .util import add_offset_pin_joint

# -------------------------------------------------------------------------------------
#                              Fusion functions
# -------------------------------------------------------------------------------------


def fusion_lockout_function(alpha = 1.5, beta = 350):
    lockout_countdown = int(random.gammavariate(alpha=alpha, beta=beta)) + 1
    return lockout_countdown

# defaults to 2 and 64 seconds
def fission_lockout_function(low = 16*2, high = 16* 64):
    
    # get a random lockout value, using uniform sampling
    lockout_countdown = int(random.uniform(low,high))
    
    # if the lockout is less than 1, set it to 1
    if lockout_countdown < 1:
        lockout_countdown = 1
    
    return lockout_countdown
    

def setup_collision_handler(space, chain_manager, col_loc_store):
    """
    Sets up a collision handler for objects within a pymunk space.

    Parameters:
    - space: The pymunk Space where the collision handler will be set up.
    - chain_manager: The manager responsible for handling chain connections upon collisions.
    - col_loc_store: The collision location store object used to track collision locations.

    This function does not return a value but configures collision handling to trigger chain connections.
    """

    # Setup a collision handler for specific types of collisions
    handler = space.add_collision_handler(1, 1)

    # Define the post-solve action for collisions
    handler.post_solve = lambda arbiter, space, data: connect_on_chain_collision(
        arbiter, space, data, chain_manager, col_loc_store
        )

    return True



def connect_on_chain_collision(arbiter, space, _, chain_manager, col_loc_store):


    # get the bodie objects from the arbiter that were involved in the collision
    body_a, body_b = arbiter.shapes[0].body, arbiter.shapes[1].body

    # uses the chain manager can valid connect function, 
    #   if the bodies can not connect, return None and bail from this entire function, 
    #   if it can connect, continue
    if not chain_manager.can_valid_connect(body_a, body_b):
        return None
    
    
    
    chain_a = chain_manager.find_chain_from_body(body_a)
    chain_b = chain_manager.find_chain_from_body(body_b)
    
    
    if chain_a == chain_b and chain_a.goal_obj.goal_type != 'peri':
        
        pin_joint = add_offset_pin_joint(space, body_a, body_b)

        chain_a.G.add_edge(body_a, body_b, link=pin_joint)

        return True
    

    # adds the pin join between the bodies (done before any states are updated)
    pin_joint = add_offset_pin_joint(space, body_a, body_b)


    # joins the chains into a new chain, adds it to the chain manager, removes the old chains from the chain manager, and updates the ball objects with the new chain object
    # if both chains are peri (special case of self fusion), the function acts as normal
    created_chain = chain_manager.join_graphs(
        chain_a, chain_b, body_a, body_b, pin_joint
    )

    # get a new goal object, based of the properties of the two chains and the new chain
    new_goal = decide_fusion_goal(chain_a, chain_b, created_chain)
    # assign the new goal object to the new chain
    created_chain.set_goal(new_goal)

    # add the collision location to the collision location store class object
    mid_point = (body_a.position + body_b.position) / 2
    
    pos1 = (body_a.position[0], body_a.position[1])
    pos2 = (body_b.position[0], body_b.position[1])
    
    col_loc_store.add_collision(mid_point)
    col_loc_store.add_fusion(id(chain_a), id(chain_b), mid_point, id(created_chain), id(body_a), id(body_b), pos1, pos2)

    return True


# -------------------------------------------------------------------------------------
#                             Chain Class
# ------------------------------------------------------------------------------------


class Chain:
    
    # global class var that determins the number of balls in the chain required for the fission lockout to be set, and to update on each frame.
    fission_threshold = 4
    init_lockout = 16*4
    fission_lockout_range = (16*1, 16*32)
    
    fusion_lockout_alpha = 350
    fusion_lockout_beta = 1.5
    
    
    cycle_fusion_lockout = 32
    cycle_fission_lockout = 32
    
    # ------------ constructor
    def __init__(self):
        # empty place to store graph
        self.G = None
        self.lockout = self.init_lockout# ~4 seconds
        self.goal_obj = None
        self.fission_lockout = fission_lockout_function(low = self.fission_lockout_range[0], high = self.fission_lockout_range[1])
        
        
    # ----------- basic logic checking functions (used by other functions/goals/chain manager)

    # checks if a body is a head
    def is_head(self, body):
        return self.G.degree(body) == 1

    # returns the nodes with degree 1
    def return_nodes_w_degree_one(self):
        return [node for node, degree in self.G.degree() if degree == 1]

    # returns the distance to the closest edge of the body passed into the function (body must be in the chain, else returns None)
    def dist_to_edge(self, body):
        edge_nodes = self.return_nodes_w_degree_one()

        shortest_path_length = float("inf")

        for target_node in edge_nodes:
            try:
                path_length = nx.shortest_path_length(
                    self.G, source=body, target=target_node
                )
                shortest_path_length = np.min([shortest_path_length, path_length])
            except nx.NetworkXNoPath:
                continue

        return shortest_path_length if shortest_path_length != float("inf") else None

    # checks if a body is in the chain
    def is_body_in_chain(self, query_body):
        if query_body in self.G:
            return True
        return False

    # returns the number of balls in the chain
    def num_balls(self):
        return len(self.G)

    def get_chain_bodies(self):
        return list(self.G.nodes())

    # -------------- Populating the chain from graph or lists (nb: the list method should be removed in favor of the graph method, but is used for testing and debugging purposes)

    # create a chain from a graph
    def instantiate_from_graph(self, graph):
        ### input a full graph (used for creating children, eventually should be used inplace of instantiate_from_lists, with the chain generator directly generating graphs)

        if self.G is None:
            self.G = graph

            return True

        return False

    # create a chain from lists
    def instantiate_from_lists(self, body_list, link_list):
        # NB: SHOULD BE REMOVED IN FAVOR OF instantiate_from_graph

        ### input
        # body_list: list of body objects
        # links_list: list of tuples with (body_obj_a, body_obj_b, link_obj)

        if self.G is None:
            self.G = nx.Graph()

            for body in body_list:
                self.G.add_node(body)

            for link_triple in link_list:
                self.G.add_edge(link_triple[0], link_triple[1], link=link_triple[2])

            return True

        return False

    # ------------ chain manipulation functions

    # usefull for debugging by allowing visualisation of the chain to easily check the chain properties as they are accessed on a per body basis
    # remove in final version
    # updates the ball objects with their respective chain object
    def update_ball_w_chain(self):
        if self.num_balls() is not None:
            for ball in self.G.nodes:
                ball.store_chain = self


    # fusion lockout functions

    # sets the lockout to a value
    def set_lockout(self, lockout_val):
        self.lockout = lockout_val

    # returns the lockout value
    def get_lockout(self):
        return self.lockout

    def lower_lockout(self):
        self.lockout -= 1
        return self.lockout
    
    
    # ------------------- cycle handling -------------------
    
    def lower_cycle_fusion_lockout(self):
        self.cycle_fusion_lockout -= 1
        
    def lower_cycle_fission_lockout(self):
        self.cycle_fission_lockout -= 1
        
        
    def get_cycle_fusion_lockout(self):
        return self.cycle_fusion_lockout
    
    def get_cycle_fission_lockout(self):
        return self.cycle_fission_lockout
    
    
    def set_cycle_fusion_lockout(self, lockout_val):
        self.cycle_fusion_lockout = lockout_val
    
    def set_cycle_fission_lockout(self, lockout_val):
        self.cycle_fission_lockout = lockout_val
        
    

    # ------------- data saving functions

    # returns a graph object, with the nodes replaced with their positions
    def return_chain_state(self):
        temp_graph = nx.Graph()

        # copy over the nodes and their positions
        for node in self.G.nodes:
            temp_graph.add_node(
                id(node),
                pos=(node.position[0], node.position[1]),
                brightness = node.store_perturber.get_val_without_update()
            )

        # add the linkslockout
        for edge in self.G.edges:
            temp_graph.add_edge(id(edge[0]), id(edge[1]))

        return temp_graph

    # ------------- goal functions

    def set_goal(self, goal_obj):
        self.goal_obj = goal_obj

    def get_goal(self):
        return self.goal_obj
    
    
    # ------------- fission functions
    
    def set_fission_lockout(self, lockout_val):
        self.fission_lockout = lockout_val
        
    
    def get_curr_fission_lockout(self):
        return self.fission_lockout
    
    def update_fission_lockout(self, multiplier = 1, len_factor = .2):
        
        if self.num_balls() >= self.fission_threshold:
        
            # lower the fission lockout
            # base lower is 1
            # multiplier is optional, but can be used to increase the rate of fission lockout decrease
                # will be used increased if the number of long chains is high
            # the lockout is also increased by the number of balls in the chain minus the threshold (num balls over 6 by defualt)
            
            self.fission_lockout -= 1 * multiplier * (self.num_balls() - self.fission_threshold + 1) * len_factor
            
        return self.fission_lockout
    


# -------------------------------------------------------------------------------------
#                             Chain Manager Class
# ------------------------------------------------------------------------------------



class ChainManager:
    

    
    
    
    def __init__(self, path_len_self_fussion_peri_only = 10, chance_peri_fussion_fail = 2, dist_from_edge_fussion = 3, chance_single_lockout_fusion_fail = 96, chance_base_fussion_fail = 60):
        self.chains = list()
        
        self.path_len_self_fussion_peri_only = path_len_self_fussion_peri_only
        self.chance_peri_fussion_fail = chance_peri_fussion_fail
        self.dist_from_edge_fussion = dist_from_edge_fussion
        self.chance_single_lockout_fusion_fail = chance_single_lockout_fusion_fail
        self.chance_base_fussion_fail = chance_base_fussion_fail
        self.peri_leaf_target_dict = dict()
        
        self.self_intersection_length = random.choice([4,5,6,7,8])
        print(self.self_intersection_length)


    # ----------------- Functions to manage stored chains

    # add a chain onject to the chain manager
    def add_chain(self, chain_obj):
        self.chains.append(chain_obj)

    # remove chain obj from chain manager
    def remove_chain(self, chain_obj):
        self.chains.remove(chain_obj)

    def lower_all_lockouts(self):
        for chain in self.chains:
            chain.lower_lockout()
            chain.lower_cycle_fusion_lockout()
            
    def lower_all_fission_lockouts(self, multiplier, len_factor):
        for chain in self.chains:
            chain.update_fission_lockout(multiplier, len_factor)
            chain.lower_cycle_fission_lockout()


    # get list of chains
    def get_chain_list(self):
        return self.chains

    # ----------------- Functions to find chains and check chain properties

    # finds a chain from a body
    def find_chain_from_body(self, query_body):
        for chain in self.chains:
            if chain.is_body_in_chain(query_body):
                return chain

        return None

    # checks if 2 body objects are in the same chain
    def are_in_same_chain(self, body_a, body_b):
        chain_a = self.find_chain_from_body(body_a)
        chain_b = self.find_chain_from_body(body_b)
        return chain_a is not None and chain_a == chain_b

    # checks if 2 body objects share an edge
    def check_if_bodies_share_edge(self, body_a, body_b):
        if not self.are_in_same_chain(body_a, body_b):
            return False

        chain_obj = self.find_chain_from_body(body_a)

        if chain_obj.G.has_edge(body_a, body_b):
            edge_data_dict = chain_obj.G.get_edge_data(body_a, body_b)
            link_obj = edge_data_dict["link"]

            return link_obj

        else:
            return False

    def can_valid_connect(self, body_a, body_b):
        
    
        # gets the chain objects
        chain_a = self.find_chain_from_body(body_a)
        chain_b = self.find_chain_from_body(body_b)
        
        goal_type_a = chain_a.get_goal().goal_type
        goal_type_b = chain_b.get_goal().goal_type
        
        can_connect = True
        
        # special case for fusions involving the peri goal type
        if goal_type_a == 'peri' or goal_type_b == 'peri':
            
            # random seed to decide if the peri peri fusion fails
            peri_fail_seed = random.uniform(0,100)

            # if they are in the same chain,
            # check if the path length is greater than 10, if it is, they can connect, else they cannot
            if chain_a == chain_b:
                
                path_length = nx.shortest_path_length(
                    chain_a.G, source=body_a, target=body_b
                )
                
                if path_length > self.path_len_self_fussion_peri_only: # 4 - 7 by default
                    can_connect = True
                    return can_connect  
                
                else:
                    can_connect = False
                    return can_connect
            # if either is peri_init they connect. else connect 1 in 99, no lockout
            else:
                if goal_type_a == 'peri_init' or goal_type_b == 'peri_init':
                    can_connect = True
                    return can_connect
                elif peri_fail_seed > self.chance_peri_fussion_fail:
                    can_connect = False
                    
                    
                    if goal_type_b == 'peri':
                        new_leading_edge_goal = Goal_leading_edge(chain_a)
                        chain_a.set_goal(new_leading_edge_goal)
                        
                    else:
                        new_leading_edge_goal = Goal_leading_edge(chain_b)
                        chain_b.set_goal(new_leading_edge_goal)
                    
                    
                    return can_connect
                else:
                    can_connect = True
                    return can_connect
            


        # all other cases

        # check if they are in the same chain
        if chain_a == chain_b:
            min_cycle_length = self.self_intersection_length
            
            
            # if they are in the same chain, they cannot connect
            # can_connect = False
            # return False
            #return False
            #print(chain_a.cycle_fusion_lockout)
            
        
            path_length = nx.shortest_path_length(
                chain_a.G, source=body_a, target=body_b
            )
            
            if path_length <= self.self_intersection_length:
                return False
            
            
            # get number of balls in chain_a
            a_num_balls = chain_a.num_balls()
            
            len_cycle_basis = len(nx.cycle_basis(chain_a.G))
            
            
            looped_factor = min_cycle_length * len_cycle_basis  + 3
            
            
            if a_num_balls > looped_factor:
                
                if len_cycle_basis > 6:            
                    return False

                if path_length >= min_cycle_length: 
                    
                    if chain_a.cycle_fusion_lockout <= 0:
                        
                        self_fusion_seed = random.uniform(0,100)
                        
                        if self_fusion_seed < 50:
                            chain_a.set_cycle_fusion_lockout(360)
                            return True
                        
                        else:
                            chain_a.set_cycle_fusion_lockout(360)
                            return False
                        
                
            else:
                return False


        # think this is redundent
        if chain_a.G.has_edge(body_a, body_b):
            return False

        # check if either body was not in a node
        if chain_a is None or chain_b is None:
            can_connect = False

        # following apply only if both chains have at least 1 head (avoids issues with stray loops)
        a_heads = len(chain_a.return_nodes_w_degree_one())
        b_heads = len(chain_b.return_nodes_w_degree_one())

        if a_heads != 0 and b_heads != 0:

            # set temp vars for head calls
            body_a_head = chain_a.is_head(body_a)
            body_b_head = chain_b.is_head(body_b)


            # removes cases when neither is a head
            if not body_a_head and not body_b_head:
                
                random_seed = random.uniform(0,100)
                
                if random_seed < 50:
                    can_connect = False
                
                pass

            else:
                # the cases where 1 is head and one is body
                if not (body_a_head and body_b_head):
                    da = chain_a.dist_to_edge(body_a)
                    db = chain_b.dist_to_edge(body_b)

                    # checks if the futher away head is not to close to the edge
                    # DIST FROM EDGE CONTROLLER (COUNTS BY EDGES NOT NODES)
                    if np.max([da, db]) < self.dist_from_edge_fussion:
                        can_connect = False


            #checks if both bodies have 2 degree 1 nodes only (avoids complex chains)
            if (a_heads > 2) or (b_heads > 2):
                if (body_a_head and body_b_head) and (a_heads == 2 or b_heads == 2):
                    pass
                else:
                    
                    if random.uniform(0,100) < 50:
                        can_connect = False
                    
                    # total_num_heads = a_heads + b_heads
                    
                    # chance_num = total_num_heads * 10
                    
                    
                    # if random.uniform(0,chance_num)< 40:
                    #     can_connect = True
                    
                    
        elif chain_a.num_balls() == 1 or chain_b.num_balls() == 1:
            pass
        else:
            #print('no head collision?')
            pass

        
        # final check before returning;
        # if both chains are locked, they cannot connect
        # if one chain is locked and the other is not, they have a 2% chance of connecting
        # if both chains are unlocked, they have a 60% chance of connecting, if they fail, they get a lockout value
    
        if (chain_a.lockout >= 0 and chain_b.lockout >= 0):  # means chain_a and chain_b are locked
            can_connect = False
        elif (chain_a.lockout >= 0 and chain_b.lockout <= 0) or (chain_a.lockout <= 0 and chain_b.lockout >= 0):  # means one is locked and one is unlocked
            if random.uniform(0, 100) < self.chance_single_lockout_fusion_fail:  # 98% chance of failing
                can_connect = False
        else: # random connect fail state, 60% chance of failing, if failed, sets lockout to a random value decided by a gamma distribution
            if random.uniform(0, 100) < self.chance_base_fussion_fail and can_connect:
                can_connect = False

                # set lockout to a random value cause the connection failed
                if chain_a.lockout <= 0:  # means a is unlocked
                    
                    chain_a_new_lockout = fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta)
                    chain_a.set_lockout(chain_a_new_lockout)  # max approx 3500 ish normally 250 ish (6 ish seconds)
                    
                if chain_b.lockout <= 0:  # means b is unlocked
                    
                    chain_b_new_lockout = fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta)
                    chain_b.set_lockout(chain_b_new_lockout)  # max approx 3500 ish normally 250 ish (6 ish seconds)

        return can_connect







    def join_graphs(self, chain_obj1, chain_obj2, exst_body_obj1, exst_body_obj2, new_link_obj):
        
        # function takes:
        # 2 chain objects (the chains from which the bodies are being connected)
        # 2 body objects (the bodies that are being connected)
        # 1 link object (the link object that has just been added to the space)
        
        
        
        # special case for peri peri fusion,
        # checks if both chains are peri type, (note; as there is only 1 peri type chain, this function will only occur on self fusions)
        if chain_obj1.goal_obj.goal_type == 'peri' and chain_obj2.goal_obj.goal_type == 'peri':
            
            # creates a graph of the first chain object
            t_graph = chain_obj1.G
            
            # adds the edge between the two bodies
            t_graph.add_edge(exst_body_obj1, exst_body_obj2, link=new_link_obj)
            
            # creates a chain object
            t_chain = Chain()
            
            # instantiates it from the graph
            t_chain.instantiate_from_graph(t_graph)

            # sets the lockout
            t_chain.set_lockout(fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta))
            
            # updates the bals
            t_chain.update_ball_w_chain()

            # removes the old chain from the chain manager
            self.remove_chain(chain_obj1)
            
            # adds the new chain to the chain manager
            self.add_chain(t_chain)


            return t_chain  
        
        
        if chain_obj1.goal_obj.goal_type == 'peri' and chain_obj2.goal_obj.goal_type == 'peri' and chain_obj1 != chain_obj2:
            print('wtf super error')
            return False
        
        
        # TODO: self fusion for non-peri chains
        if chain_obj1 == chain_obj2 and chain_obj1.goal_obj.goal_type != 'peri' and chain_obj2.goal_obj.goal_type != 'peri':
            print('------------------------------_______WARNING CATCH SELF FUSION, SHOULDNT BE HERE')
                        # creates a graph of the first chain object
            t_graph = chain_obj1.G
            
            # adds the edge between the two bodies
            t_graph.add_edge(exst_body_obj1, exst_body_obj2, link=new_link_obj)
            
            # creates a chain object
            t_chain = Chain()
            
            # instantiates it from the graph
            t_chain.instantiate_from_graph(t_graph)

            # sets the lockout
            t_chain.set_lockout(fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta))
            
            # updates the bals
            t_chain.update_ball_w_chain()

            # removes the old chain from the chain manager
            self.remove_chain(chain_obj1)
            
            # adds the new chain to the chain manager
            self.add_chain(t_chain)


            return t_chain  
            
            
        
        
            
        
        # get the graph of the union of the two chains
        t_graph = nx.union(chain_obj1.G, chain_obj2.G)
        
        # add the edge between the two bodies
        t_graph.add_edge(exst_body_obj1, exst_body_obj2, link=new_link_obj)
        
        # create new chain object and instantiate it from the graph
        t_chain = Chain()
        t_chain.instantiate_from_graph(t_graph)

        # set the fusion lockout
        t_chain.set_lockout(fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta))
        
        # update the ball objects with the new chain object
        t_chain.update_ball_w_chain()
    
        # get existing fission lockouts
        fission_lockout_1 = chain_obj1.get_curr_fission_lockout()
        fission_lockout_2 = chain_obj2.get_curr_fission_lockout()
        # get min
        old_min = np.min([fission_lockout_1, fission_lockout_2])
        
        # get new fission lockout based on old
        new_fission_lockout = fission_lockout_function(low = Chain.fission_lockout_range[0], high = Chain.fission_lockout_range[1])

        # if both are over 0, reduce the new lockout by a random value between and the minimum of the two and the new lockout
        # added to reduce the lockout time if the chains connected were close to fission event
        if fission_lockout_1 > .1 and fission_lockout_2 > .1 and old_min < new_fission_lockout*.89:
            new_fission_lockout -= random.uniform(old_min, new_fission_lockout*.9)
        
        # set the new fission lockout
        t_chain.set_fission_lockout(new_fission_lockout)
    
        # remove old chain objects from the manager, add the new one
        self.remove_chain(chain_obj1)
        self.remove_chain(chain_obj2)
        self.add_chain(t_chain)

        # return the new chain object
        return t_chain

    def delete_link(self, space, exst_body1, exst_body2, col_loc_store):
        
        # check if they are in the same chain, bail
        if not self.are_in_same_chain(exst_body1, exst_body2):
            print('1BIG ERROR SEE FISSION LOGIC')
            return False

        # get the chain object (will match)
        chain_obj = self.find_chain_from_body(exst_body1)

        # check if they share an edge, else bail
        link_obj = self.check_if_bodies_share_edge(exst_body1, exst_body2)
        if not link_obj:
            print('2BIG ERROR SEE FISSION LOGIC')
            return False
        
        
        # check if the split would create 2 chains, else bail
        graph_copy = chain_obj.G.copy()
        graph_copy.remove_edge(exst_body1, exst_body2)
        subgraphs = [
            graph_copy.subgraph(c).copy() for c in nx.connected_components(graph_copy)
        ]
        if len(subgraphs) != 2:
            print('3BIG ERROR SEE FISSION LOGIC')
            return False

        
        # update repr
        chain_obj.G.remove_edge(exst_body1, exst_body2)

        subgraphs = [
            chain_obj.G.subgraph(c).copy() for c in nx.connected_components(chain_obj.G)
        ]

        if len(subgraphs) == 2:
            new_graph_1 = subgraphs[0]
            new_graph_2 = subgraphs[1]

            # create new chain objects
            t_chain_1 = Chain()
            t_chain_1.instantiate_from_graph(new_graph_1)
            t_chain_2 = Chain()
            t_chain_2.instantiate_from_graph(new_graph_2)

            # update the ball objects with the new chain object
            t_chain_1.update_ball_w_chain()
            t_chain_2.update_ball_w_chain()

            # set lockouts
            t_chain_1.set_lockout(fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta))
            t_chain_2.set_lockout(fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta))
            
            # get random goal objects
            t_goal_1 = get_random_goal_instance(t_chain_1)
            t_goal_2 = get_random_goal_instance(t_chain_2)

            t_chain_1.set_goal(t_goal_1)
            t_chain_2.set_goal(t_goal_2)

            # set fission lockouts
            t_chain_1.set_fission_lockout(fission_lockout_function(low = Chain.fission_lockout_range[0], high = Chain.fission_lockout_range[1]))
            t_chain_2.set_fission_lockout(fission_lockout_function(low = Chain.fission_lockout_range[0], high = Chain.fission_lockout_range[1]))

            # remove the old chain object and add the new ones to the chain manager (self)
            self.remove_chain(chain_obj)
            self.add_chain(t_chain_1)
            self.add_chain(t_chain_2)
            
            # remove the link object from the space
            space.remove(link_obj)

            # add the collision location to the collision location store class object
            mid_point = (exst_body1.position + exst_body2.position) / 2
            
            pos1 = (exst_body1.position[0], exst_body1.position[1])
            pos2 = (exst_body2.position[0], exst_body2.position[1])

            col_loc_store.add_fission_loc(mid_point)
            col_loc_store.add_fission(id(chain_obj), mid_point, id(t_chain_1), id(t_chain_2), id(exst_body1), id(exst_body2), pos1, pos2)

            return True

        else:
            print('4BIG ERROR SEE FISSION LOGIC')
            return False



    def special_peri_fission(self, space, peri_body, init_body, col_loc_store):


        # get the chain object (will match)
        chain_obj = self.find_chain_from_body(peri_body)

        # check if they share an edge, else bail
        link_obj = self.check_if_bodies_share_edge(peri_body, init_body)


        # update repr
        chain_obj.G.remove_edge(peri_body, init_body)

        subgraphs = [
            chain_obj.G.subgraph(c).copy() for c in nx.connected_components(chain_obj.G)
        ]
        
        if peri_body in subgraphs[1]:
            new_peri = subgraphs[0]
            new_init = subgraphs[1]
            
        else:
            new_peri = subgraphs[1]
            new_init = subgraphs[0]
        

        if len(subgraphs) == 2:

            # create new chain objects
            new_peri_chain = Chain()
            new_peri_chain.instantiate_from_graph(new_peri)
            new_init_chain = Chain()
            new_init_chain.instantiate_from_graph(new_init)

            # update the ball objects with the new chain object
            new_peri_chain.update_ball_w_chain()
            new_init_chain.update_ball_w_chain()

            # set lockouts
            #new_peri_chain.set_lockout(fusion_lockout_function())
            new_init_chain.set_lockout(fusion_lockout_function(alpha=Chain.fusion_lockout_alpha, beta=Chain.fusion_lockout_beta))
            
            # get random goal objects
            t_goal_1 = Goal_peri(new_peri_chain)
            
            seed = random.uniform(0,100)
            if seed > 20:
                t_goal_2 = Goal_leading_edge(new_init_chain)
            else:
                t_goal_2 = Goal_idle(new_init_chain)
            

            new_peri_chain.set_goal(t_goal_1)
            new_init_chain.set_goal(t_goal_2)

            # set fission lockouts
            #new_peri_chain.set_fission_lockout(fission_lockout_function())
            new_init_chain.set_fission_lockout(fission_lockout_function(low = Chain.fission_lockout_range[0], high = Chain.fission_lockout_range[1]))

            # remove the old chain object and add the new ones to the chain manager (self)
            self.remove_chain(chain_obj)
            self.add_chain(new_peri_chain)
            self.add_chain(new_init_chain)
            
            # remove the link object from the space
            space.remove(link_obj)

            # add the collision location to the collision location store class object
            mid_point = (peri_body.position + init_body.position) / 2
            
            pos1 = (peri_body.position[0], peri_body.position[1])
            pos2 = (init_body.position[0], init_body.position[1])
            
            col_loc_store.add_fission_loc(mid_point)
            col_loc_store.add_fission(id(chain_obj), mid_point, id(new_peri_chain), id(new_init_chain), id(peri_body), id(init_body), pos1, pos2)

            return True




# -------------------------------------------------------------------------------------
#                             Chain Manager Class
# ------------------------------------------------------------------------------------


def chain_generator_closed_shape(num_chains_range, patience, centered_cell_shape, leading_edge_shape, inner_rad_dist, width, height, chain_len_range, ang_range, mito_ball_radius, ball_mass_mult, perturber_obj_template, portion_leading, space):

    # patience for the chain generator
    patience = 1000
    
    # list to store the chains before being added to the chain manager
    t_chain_list = []
    
    # number of chains to generate (based on the range set in the input variable: num_chains_range)
    num_chains = random.randint(num_chains_range[0],num_chains_range[1])

    # vars to keep track of the number of chains generated and the number of failed chains
    c_counter = 0
    patience_counter = 0
    
    while c_counter < num_chains:

        # random starting point in disk (50 percent sampled in leading edge)
        seed1 = random.uniform(0,100)
        if seed1 > portion_leading:
            t_x_loc, t_y_loc = sample_points_soft_donut(inner_rad_dist, centered_cell_shape, screen_size = (width,height))
        else:
            t_x_loc, t_y_loc = sample_points_soft_donut(inner_rad_dist, leading_edge_shape, screen_size = (width,height))

        # random starting angle for disk direction
        curr_ang = radians(random.uniform(0,360))

        # flag to check if chain is valid
        failed_flag = False

        # running loc and angle variables
        curr_loc = [t_x_loc, t_y_loc]
        
        # runnning chain object list
        t_list = []

        # get random chain length
        t_chain_len = random.randint(chain_len_range[0], chain_len_range[1])

        # chain_ang_range
        ang_high = random.randint(ang_range[0], ang_range[1])
        
        # try create new chain
        for i in range(t_chain_len):
        
            # check if the seed point has no intersections
            query = space.point_query_nearest((curr_loc[0],curr_loc[1]), mito_ball_radius, pymunk.ShapeFilter())

            # if the seed point is valid, tries again
            if query is not None:
                failed_flag = True
                break
        
            else:
                # create the now checked to be valid ball object and add it to the list
                ball_obj = add_ball(space, loc_x = curr_loc[0], loc_y = curr_loc[1], radius = mito_ball_radius, mass = (mito_ball_radius*ball_mass_mult), elasticity=0, friction = 0, perturber_obj_template=perturber_obj_template)
                t_list.append(ball_obj)

                curr_ang += radians(random.uniform(-ang_high,ang_high))

                # get a new loc for the next proposal
                x_new = curr_loc[0] + (mito_ball_radius*2+0.001)*cos(curr_ang)
                y_new = curr_loc[1] + (mito_ball_radius*2+0.001)*sin(curr_ang)

                # store the next proposal loc
                curr_loc = [x_new, y_new]
        
        # if the chain failed, remove all the balls
        if failed_flag == True:
        
            # remove all the balls from the space, cause the chain failed
            for i in range(len(t_list)):
                # get the shape and body
                tshape = t_list[i].store_shape
                tbody = t_list[i]
                # remove the shape and body from the space
                space.remove(tshape, tbody)
            
            # increment the patience counter
            patience_counter += 1

        # if the chain is valid, add it to the list of chains
        else:
            c_counter+=1
            patience_counter = 0

            # create chain
            t_chain_list.append(t_list)    

        # if the generator is out of patience, break the loop, but still add the chains that were generated and print the number of chains generated
        if patience_counter > patience:
            print(f'WARNING: generator out of patience, only generated {c_counter} chains')
            break
        
    return t_chain_list