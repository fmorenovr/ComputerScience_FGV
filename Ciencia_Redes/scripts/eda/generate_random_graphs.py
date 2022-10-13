'''
Generate random graphs according to an input. 

python scripts/eda/generate_random_graphs.py  \
        --source graphs/wood_years_2018/wood.graphml \
        --output_dir baselines
'''
import os 
import sys 
import numpy as np 
import networkx as nx 
import json 

# Documentation 
import argparse 
from tqdm import tqdm 

sys.path.append('scripts/figures')  
from draw_subgraph import draw_graph 

from random_model import BarabasiAlbert, RandomModel, MarketEnum 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Generate random graphs and compute their summaries.') 
    parser.add_argument('--source', type=str, required=True, help='The `graphml` file.')  
    parser.add_argument('--output_dir', type=str, required=True, help='The directory to write the graphs.') 
    return parser.parse_args() 

def degree2type(graph: nx.DiGraph, node: str): 
    '''
    Compute the type of the node `node` within the graph `graph` according to the MarketEnum 
    attributes. 
    '''
    out_degrees = graph.out_degree 
    in_degrees = graph.in_degree 
    if out_degrees[node] == 0: 
        return MarketEnum.CONSUMER 
    if in_degrees[node] == 0: 
        return MarketEnum.SUPPLIER 
    # A node active as both a supplier and a consumer is a trader 
    return MarketEnum.TRADER 

def generate_configuration_graphs(graph: nx.DiGraph, quantity: int): 
    '''
    Generate graphs configurationally equivalent to `graph`. 
    '''
    nodes = list(graph.nodes) 
    in_degree_sequence = [graph.in_degree[n] for n in nodes] 
    out_degree_sequence = [graph.out_degree[n] for n in nodes] 
    create_using = nx.DiGraph 

    for inx in tqdm(range(quantity)): 
        random = nx.directed_configuration_model( 
                in_degree_sequence=in_degree_sequence, 
                out_degree_sequence=out_degree_sequence, 
                create_using=create_using) 
        yield random 

def generate_albert_barabasi_graphs(graph: nx.DiGraph, quantity: int): 
    '''
    Generate graphs according to the linear preferential attachment rule. 

    The nodes with null in degree correspond to suppliers; with null out degree, 
    to consumers; otherwise, to traders. 

    It would be appropriate to evaluate whether the distribution of the network is randomized if 
    we used a nonlinear attachment model. 
    '''
    model = BarabasiAlbert(graph)           # We should choose a subgraph with important nodes,
                                            # as this needs to scale
    for inx in tqdm(range(quantity)): 
        yield model.sample() 

def summary_random_graphs(graph: nx.DiGraph, output_dir: str, type: str='configuration'): 
    '''
    Summarize the network's density and the variance in the nodes' degrees. 
    '''
    if type == 'configuration': 
        random_graph_generator = generate_configuration_graphs
    elif type == 'ab': 
        random_graph_generator = generate_albert_barabasi_graphs

    transitivities = list() 
    avgdegrees = list() 
    assortativities = list() 
    for random_graph in random_graph_generator(graph, quantity=int(1e3)): 
        transitivities.append(nx.transitivity(random_graph)) 
        avgdegrees.append(2 * random_graph.number_of_edges() / graph.number_of_nodes())
        assortativities.append(nx.degree_assortativity_coefficient(random_graph, x='out', y='in')) 

    # Write a visualization for the graph 
    nx.set_node_attributes( 
        G=random_graph, 
        values={ 
            node:degree2type(graph=random_graph, node=node) for node in random_graph.nodes 
        }, 
        name='type' 
    ) 

    draw_graph(graph=random_graph, output_dir=output_dir, max_nodes=199, initial_size=9) 
    json.dump(
            transitivities, 
            open(os.path.join(output_dir, f'transitivity_{type}.json'), 'w') 
    ) 
    json.dump( 
            avgdegrees, 
            open(os.path.join(output_dir, f'avgdegrees_{type}.json'), 'w') 
    ) 
    json.dump( 
            assortativities, 
            open(os.path.join(output_dir, f'assotativities_{type}.json'), 'w') 
    ) 
    
if __name__ == '__main__': 
    args = parse_args() 
    if not os.path.exists(args.output_dir): 
        os.mkdir(args.output_dir) 
    graph = nx.read_graphml(args.source)
    summary_random_graphs(graph, output_dir=args.output_dir, type='configuration') 
    summary_random_graphs(graph, output_dir=args.output_dir, type='ab') 
        
