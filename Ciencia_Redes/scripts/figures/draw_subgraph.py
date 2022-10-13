""" 
The network is large; to display it, we should sample appropriate nodes. 

python scripts/figures/draw_subgraph.py --source graphs 

Also, 

python scripts/figures/draw_subgraph.py --source graphs \
                                        --fname wood_years_2018 

to draw the subgraph for a specific data set. 
""" 
import os 
import glob 

import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D 

from networkx.drawing import spring_layout 

# Documentation 
from typing import List, Dict 
from tqdm import tqdm 
import argparse 

def parse_args(): 
    '''
    Parse the command line parameters. 
    ''' 
    parser = argparse.ArgumentParser(description='Sample nodes with a BFS-like algorithm to draw the network.') 
    parser.add_argument('--source', type=str, required=True, help='The directory in which the folders for each data are available.') 
    parser.add_argument('--fname', type=str, default=None, help='Choose a folder within `source` to generate the images.')  
    # Parse the command line parameters 
    args = parser.parse_args() 
    # Return the parsed parameters
    return args 

def draw_network( 
        graph: nx.Graph, 
        fname: str, 
        types: List[str], 
        colors: Dict[str, str], 
        attr: str, 
        pos: np.ndarray=None, 
        with_labels: bool=False
    ): 
    """ 
    Write a network (with labels) for the graph `graph`. 
    """ 
    colors_map = [colors[node[1][attr]] for node in graph.nodes.data()] 
    nx.draw(graph, 
            node_size=29, 
            node_color=colors_map, 
            pos=pos) 
    if with_labels: 
        nx.draw_networkx_labels(graph, {node:(inx[0], inx[1]+1e-2) for node, inx in pos.items()}) 
    # Write a legend for the Network 
    placeholder = [0] 
    custom_legends = [Line2D(placeholder, 
        placeholder, 
        color="w", 
        markerfacecolor=colors[t], 
        marker="o", 
        markersize=15) for t in types] 
    plt.legend(custom_legends, types) 
    plt.savefig(fname)   
    plt.show() 
    plt.clf() 

def sample_graph( 
        G: nx.DiGraph, 
        types: List[str], 
        samples: List[int], 
        attr: str="type", 
        figure_fname: str="network.png" 
    ): 
    """ 
    Sample, fore ach type `types[i]`, `samples[i]` nodes and return the 
    induced subgraph. 
    """ 
    # Generate samples for each type 
    nodes = list() 
    for i, t in enumerate(types): 
        # G.nodes.data() returns a list of tuples, with 
        # the initial component as the node's identifier and the other as the node's attributes
        tnodes = [node[0] for node in G.nodes.data() if node[1][attr] == t] 
        tnodes = np.random.choice(tnodes, size=samples[i], replace=False) 
        nodes.extend(tnodes) 

    # Compute the induced subgraph (by the nodes in `nodes`) 
    sgraph = G.subgraph(nodes) 
    colors = ["purple", "orange", "violet"] 
    draw_network(sgraph, figure_fname, 
            types=types, 
            colors={t:color for t, color in zip(types, colors)}, 
            attr="type") 
    return sgraph 


def sample_children(nodes: List, children_per_node: int): 
    children = set() 
    for node in nodes: 
        if len(node) < 1: 
            continue 
        inxs = np.random.choice(len(node), replace=False, 
                size=min(len(node), children_per_node)) 
        for c in inxs: 
            children.add(node[c]) 
    return children  

def sample_recursively( 
        G: nx.DiGraph, 
        initial_nodes: List, 
        max_nodes: int, 
        nodes_: List=set() 
    ): 
    """ 
    Sample nodes recursively. 
    """ 
    # Sample a parent for each node 
    nodes = [list(G.successors(node)) for node in initial_nodes] 
    nodes = [node for node in nodes if len(node) >= 1] 
    nodes = sample_children(nodes, children_per_node=2) 
    nodes = set([node for node in nodes if node not in nodes_]) 
    nodes_.update(initial_nodes) 

    if len(nodes) < 1 or len(nodes_) > max_nodes: 
        return nodes_

    nodes_.update(sample_recursively(G=G, 
            initial_nodes=nodes, 
            max_nodes=max_nodes, 
            nodes_=nodes_)) 
    
    return nodes_

def draw_graph(
        G: nx.Graph, 
        output_dir: str, 
        max_nodes: int, 
        initial_size: int, 
        fname: str='network_sample.png', 
        with_labels: bool=False, 
        pos: np.array=None): 
    '''
    Sample a subset of the graph `G`'s nodes and generate a visualization of their induced subgraph. 
    '''
    print('Graph loaded!') 
    initial_nodes = np.random.choice(
            a=[node[0] for node in G.nodes.data('type') if node[1] == "MANEJO"], 
            size=initial_size, 
            replace=False 
    ) 
    
    if max_nodes != G.number_of_nodes(): 
        nodes = sample_recursively( 
                G, 
                initial_nodes=set(initial_nodes.tolist()),
                max_nodes=max_nodes
        ) 
    else: 
        nodes = [n for (n, d) in G.degree if d != int(1e-19)] 
    print('Nodes sampled!') 
    # Generate the subgraph 
    sgraph = G.subgraph(nodes) 
    eps = 3
    unif = np.random.uniform
    generate_pos = { 
            'MANEJO': lambda: np.array([-9, 1]) + np.random.uniform(-eps, eps, size=2), 
            'PTO': lambda: np.array([3, 1]) + np.random.uniform(-eps, eps, size=2), 
            'FINAL': lambda: np.array([9, 1]) + np.random.uniform(-eps, eps, size=2) 
    } 
    if pos is None: 
        pos = spring_layout( 
                sgraph, 
                pos={ 
                    node[0]: generate_pos[node[1]]() for node in sgraph.nodes.data('type')   
                }, 
                k=3, 
                fixed=list(sgraph.nodes) 
        ) 
    print('Coordinates computed!') 
    draw_network( 
            sgraph, 
            fname=os.path.join(output_dir, fname), 
            types=["MANEJO", "PTO", "FINAL"], 
            colors={"MANEJO": "green", "PTO": "blue", "FINAL": "orange"}, 
            attr="type", 
            pos=pos, 
            with_labels=with_labels
        ) 

if __name__ == "__main__": 
    args = parse_args() 
    np.random.seed(42) # Reproducibility 
    for folder in tqdm(os.listdir(args.source)): 
        if (args.fname is not None and folder != args.fname): continue 
        data_dir = os.path.join(args.source, folder) 
        G = nx.read_graphml(
            os.path.join(
                data_dir, 
                "wood.graphml"
            )
        ) 
        draw_graph(G, output_dir=data_dir, max_nodes=199, initial_size=9) 

