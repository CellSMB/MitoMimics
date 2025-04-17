
import random



from .goals import *
from .mem_nuc import *
from .mito import *
from .util import *



def fission_manager(chain_manager, space, collision_manager, fission_threshold, min_chain_len):
    
    chain_list_valid = chain_manager.get_chain_list()
    
    # get chains that are not peri_init class
    chain_list_valid = [x for x in chain_list_valid if x.get_goal().goal_type != 'peri_init']
    # get chains that are not peri class
    chain_list_valid = [x for x in chain_list_valid if x.get_goal().goal_type != 'peri']
    # get chains that are 6 or more balls long
    chain_list_valid = [x for x in chain_list_valid if x.num_balls() >= fission_threshold]
    # get chains that are not in a lockout state
    chain_list_valid = [x for x in chain_list_valid if x.get_curr_fission_lockout() <= 0]

        
    # chain_list_valid_t = list()
    
    # for t_chain in chain_list_valid:
    #     if not nx.cycle_basis(t_chain.G):
    #         chain_list_valid_t.append(t_chain)

    #     else:
    #         #print(t_chain.G)
    #         pass
        
    # chain_list_valid = chain_list_valid_t        
        
        
        
        

    while len(chain_list_valid) > 0:
        
        chain_obj = random.choice(chain_list_valid)
        
        chain_obj_graph = chain_obj.G
        
        
        
        
        
        bridges = list(nx.bridges(chain_obj_graph))
        

        valid_bridges = []
        
        
        for bridge in bridges:
            G_temp = chain_obj_graph.copy()
            G_temp.remove_edge(*bridge)

            # Check the subgraphs resulting from the removal of the bridge
            subgraphs = list(nx.connected_components(G_temp))

            # Ensure both resulting subgraphs have more than one node
            if all(len(subgraph) > 1 for subgraph in subgraphs):
                valid_bridges.append(bridge)
                
                
        
        valid_non_bridges = [x for x in chain_obj_graph.edges if x not in bridges]
        
        
        if len(valid_bridges) == 0 and len(valid_non_bridges) == 0:
            
            chain_list_valid.remove(chain_obj)
            print('a chain had no valid nodes to remove, this is an error')
            
            return False    
        

        
        if len(valid_bridges) != 0 and len(valid_non_bridges) != 0:
            pref_bridges_seed = random.randint(0, 100)
        elif len(valid_bridges) == 0:
            pref_bridges_seed = 101
        elif len(valid_non_bridges) == 0:
            pref_bridges_seed = -1
            
            
        
        if pref_bridges_seed < 40:
            
                    # if their are no links of this len, return False to do nothing
            if len(valid_bridges) == 0:
                
                print('THERE WERE NO VALID BRIDGES, huge error')
                
                return False
            
            random_edge = random.choice(valid_bridges)
            
            # fission the chains
            chain_manager.delete_link(space, random_edge[0], random_edge[1], collision_manager)   
            
            chain_list_valid.remove(chain_obj)
            
            #print('fission a brdige')   
            
            return True
        
        else:
            #print('REMOVED AN EDGE THAT DIDNT SPLIT, NOT an error')
            
            random_edge = random.choice(valid_non_bridges)
            
            body_a = random_edge[0]
            body_b = random_edge[1]
            
            link_obj = chain_manager.check_if_bodies_share_edge(body_a, body_b)
                        
            space.remove(link_obj)

            chain_obj.G.remove_edge(body_a, body_b)
            
            chain_obj.set_fission_lockout = chain_obj.get_curr_fission_lockout() + random.uniform(8, 256)
        
            return False

        

    # while len(chain_list_valid) > 0:
        
    #     chain_obj = random.choice(chain_list_valid)
        
    #     valid_links_to_break =[]
        
        
    #     # get the links that have at least min_chain_len balls on each side
    #     for b1, b2 in chain_obj.G.edges():
    #         if chain_obj.dist_to_edge(b1) >= min_chain_len - 1 and chain_obj.dist_to_edge(b2) >= min_chain_len - 1:
    #             valid_links_to_break.append((b1, b2))
        
    #     # if their are no links of this len, return False to do nothing
    #     if len(valid_links_to_break) == 0:
            
    #         chain_list_valid.remove(chain_obj)
            
    #         return False

    #     # choose a random link to break
        
        
        
    #     random_edge = random.choice(valid_links_to_break)
        
    #     # fission the chains
    #     chain_manager.delete_link(space, random_edge[0], random_edge[1], collision_manager)   
        
    #     chain_list_valid.remove(chain_obj)




def find_specific_edges(G, y_ex, high, low):
    # This function checks if the subgraph satisfies the specific conditions


    result_edges = []
    # Iterate through each edge
    
    test_edges = [(u, v) for u, v in G.edges() if G.degree(u) <= 2 or G.degree(v) <= 2]
    test_edges = [(u, v) for u, v in test_edges if u not in y_ex and v not in y_ex]
    
    for edge in test_edges:
        
        
        # Temporarily remove the edge
        G.remove_edge(*edge)
        # Check if the graph is now disconnected into exactly two components
        if nx.number_connected_components(G) == 2:
            # Check each component to see if it meets the criteria
            components = list(nx.connected_components(G))
            # Ensure one component satisfies each condition
            
            x,y = G.subgraph(components[0]), G.subgraph(components[1])
            
            edge
            
            
            if ((len(x.nodes()) <= high and len(x.nodes()) >=low) and all(degree <= 2 for _, degree in x.degree())) and set(y_ex).issubset(y.nodes()):
                if edge[0] in x:
                    result_edges.append((edge, 0))
                else:
                    result_edges.append((edge, 1))
            
            if ((len(y.nodes()) <= high and len(y.nodes()) >=low) and all(degree <= 2 for _, degree in y.degree())) and set(y_ex).issubset(x.nodes()):
                if edge[0] in y:
                    result_edges.append((edge, 0))
                else:
                    result_edges.append((edge, 1))                 
            
            

        # Add the edge back to the graph
        G.add_edge(*edge)

    return result_edges



def fission_manager_peri(chain_manager, space, collision_manager, exclude_balls):
    
    # get peri chain
    chain_list_valid = chain_manager.get_chain_list()
    peri_chain = [x for x in chain_list_valid if x.get_goal().goal_type == 'peri'][0]

    
    
    peri_graph_copy = peri_chain.G.copy()

    
    special_edges = find_specific_edges(peri_graph_copy, exclude_balls, 9, 3)


    
    if len(special_edges) == 0:
        return False
    
    prospect_edge = random.choice(special_edges)
    
    
    if prospect_edge[1] == 0:
        peri_body = prospect_edge[0][0]
        init_body = prospect_edge[0][1]
    else:
        peri_body = prospect_edge[0][1]
        init_body = prospect_edge[0][0]
    

    
    chain_manager.special_peri_fission(space, peri_body, init_body, collision_manager)
    
    return True
    