'''
Instantiate a network with a highly constrained topology, in which the market discrepancies 
should be identified by sufficiently sensible algorithms. 

python scripts/eda/network_instance.py  --output similarities
'''
import os 
import numpy as np 
import networkx as nx 

import sys 
sys.path.append('scripts/figures') 
from draw_subgraph import draw_graph 
from random_model import MarketEnum 

import matplotlib.pyplot as plt 

# Documentation 
import argparse 
from typing import List 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Instantiate a highly topologically constrained network.') 
    parser.add_argument('--output', type=str, required=True, help='The directory to which the list of edges will be written.') 
    return parser.parse_args() 

def generate_network_communities(directory: str): 
    '''
    Generate the highly topologically constrained network. 
    '''
    if not os.path.exists(directory): 
        os.mkdir(directory) 
    graph = nx.DiGraph() 
    suppliers = np.arange(22) 
    traders = suppliers[-1] + 1 + np.arange(2) 
    consumers = traders[-1] + 1 + np.arange(16) 

    graph.add_nodes_from(suppliers) 
    graph.add_nodes_from(traders) 
    graph.add_nodes_from(consumers) 
    
    attributes = dict() 
    attributes.update({supplier:MarketEnum.SUPPLIER for supplier in suppliers}) 
    attributes.update({trader:MarketEnum.TRADER for trader in traders}) 
    attributes.update({consumer:MarketEnum.CONSUMER for consumer in consumers}) 
    
    nx.set_node_attributes(graph, attributes, 'type') 
    for inx, supplier in enumerate(suppliers): 
        if inx > len(suppliers) * .8: 
            graph.add_edge(supplier, traders[1])
        else: 
            graph.add_edge(supplier, traders[0]) 
    p = .1 
    for inx, consumer in enumerate(consumers):
        unf = np.random.uniform() 
        if unf < p: 
            graph.add_edge(traders[1], consumer) 
        elif unf <= 9 * p: 
            graph.add_edge(traders[0], consumer) 
        else: 
            graph.add_edge(traders[1], consumer) 
            graph.add_edge(traders[0], consumer) 
    edge_weight = lambda edge: 1 if (edge[0] == traders[0] or edge[1] == traders[0]) else .5 
    edges = {edge:edge_weight(edge) for edge in graph.edges} 
    nx.set_edge_attributes(graph, edges, 'weight') 
    # Compute the coordinates to draw the graph 
    coordinates = dict()    # Consolidate the nodes' coordinates
    height = 18             # The height of the horizontally disposed nodes 
    coordinates.update({supplier:(.1, i * height / len(suppliers)) for i, supplier \
            in enumerate(suppliers)}) 
    coordinates.update({trader:(.3, i * height / len(traders)) for i, trader \
            in enumerate(traders)}) 
    coordinates.update({consumer:(.5, i * height / len(consumers)) for i, consumer \
            in enumerate(consumers)})  
    
    draw_graph( 
            graph, 
            initial_size=len(suppliers), 
            max_nodes=graph.number_of_nodes(), 
            output_dir=directory, 
            fname='communities.png', 
            pos=coordinates
    ) 
    nx.write_graphml(graph, f'{directory}/communities.graphml') 

    # Update the nodes' attributes to capture the networks' underlying dynamics 
    nx.relabel_nodes(graph, {traders[1]: consumers[-1]+1, traders[0]: consumers[-1]+2}, copy=False) 
    coordinates[consumers[-1]+1] = coordinates[traders[0]] 
    coordinates[consumers[-1]+2] = coordinates[traders[1]] 
    draw_graph( 
            graph, 
            initial_size=len(suppliers), 
            max_nodes=graph.number_of_nodes(), 
            output_dir=directory, 
            fname='communities_ii.png', 
            pos=coordinates
    ) 
    nx.write_graphml(graph, f'{directory}/communities_ii.graphml') 
    return graph 

def generate_networks_similarity(directory: str): 
    '''
    Generate networks amenable to the temporal identification of similar nodes. 
    '''
    # 
    # 3 +++ 5 ++++++++ 8 
    #              +
    # 2 +++ 4 ++++ 6  
    #       + 
    # 1 +++++
    pos = { 
            1: (.1, .1), 
            2: (.1, .3), 
            3: (.1, .5), 
            4: (.3, .2), 
            5: (.3, .4), 
            6: (.5, .2), 
            7: (.5, .4), 
    } 
    graph = nx.DiGraph() 
    graph.add_edge(1, 4) 
    graph.add_edge(2, 4) 
    graph.add_edge(3, 5) 
    graph.add_edge(4, 6) 
    graph.add_edge(5, 7) 
    graph.add_edge(5, 6) 
    attributes = { 
            1: MarketEnum.SUPPLIER, 
            2: MarketEnum.SUPPLIER, 
            3: MarketEnum.SUPPLIER, 
            4: MarketEnum.TRADER, 
            5: MarketEnum.TRADER, 
            6: MarketEnum.CONSUMER, 
            7: MarketEnum.CONSUMER 
        } 
    nx.set_node_attributes(graph, attributes, name='type') 
    draw_graph( 
            graph, 
            initial_size=3, 
            max_nodes=graph.number_of_nodes(), 
            output_dir=directory, 
            fname='network_i.png', 
            with_labels=True, 
            pos=pos
    ) 
    nx.write_graphml(graph, f'{directory}/similarity_i.graphml') 
    
    # Update the nodes' names 
    graph = nx.DiGraph() 
    graph.add_edge(1, 8) 
    graph.add_edge(2, 8) 
    graph.add_edge(3, 9) 
    graph.add_edge(9, 7) 
    graph.add_edge(9, 6) 
    graph.add_edge(8, 6) 
    attributes[8] = attributes[4] 
    attributes[9] = attributes[5] 
    pos[8] = pos[4] 
    pos[9] = pos[5] 
    nx.set_node_attributes(graph, attributes, name='type')  
    nx.write_graphml(graph, f'{directory}/similarity_ii.graphml') 
    draw_graph( 
            graph, 
            initial_size=3, 
            max_nodes=graph.number_of_nodes(), 
            output_dir=directory, 
            fname='network_ii.png', 
            with_labels=True, 
            pos=pos 
    ) 
if __name__ == '__main__': 
    args = parse_args() 
    generate_network_communities(directory=args.output) 
    generate_networks_similarity(directory=args.output) 

