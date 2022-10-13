'''
Implant Newman's algorithm to identify communities within a graph; the heuristic is that 
a oligopoly is conformed by a collection of locally important players within the trader networks.

The main measures of importance are the clustering coefficient (local) and betweenness (global). 
A waning local clustering coefficient corresponds to a high control of the information flow within the network. 
A large betweenness effectively implies global control for the market; the directedness is important. 

python scripts/eda/monopolies.py    --source graphs/wood_years_2018/wood.graphml \
                                    --output graphs/wood_years_2018/figures \
'''
import os 
import numpy as np 
import networkx as nx 

import sys 
sys.path.append('scripts/figures') 
from figures import logarithm_binning, linear_binning 
from random_model import MarketEnum 

# Algorithms 
from networkx.algorithms import community 
from collections import OrderedDict 

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors 
import json 

# Documentation 
import argparse 
from typing import List, Dict 
from tqdm import tqdm 

np.random.seed(42) 
plt.style.use(os.environ['MPL_STYLE']) 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Identify locally and globally crucial players.')
    parser.add_argument('--source', type=str, required=True, help='The graphml source file.') 
    parser.add_argument('--output', type=str, required=True, help='The folder to which the text files will be written.') 
    # parser.add_argument('--measures', type=str, default='betweenness', help='The centrality measure for inducing a order relation within the set of nodes.') 
    return parser.parse_args() 

def empirical_distribution(quantities: Dict[str, float], scale: str, nbins: int): 
    '''
    Compute the (binned) empirical distribution for the estimated clustering quantities. 
    '''
    # Identify the data's boundaries 
    values = list(quantities.values()) 
    if scale == 'log': 
        bins = np.logspace(np.log2(min(values)), np.log2(max(values)), num=nbins, base=2) 
        width = None 
    elif scale == 'linear': 
        bins = np.linspace(min(values), max(values), num=nbins)
        width = bins[-1] - bins[-2] 
    counts, edges = np.histogram(values, bins=bins) 
    empirical_dist = counts.astype(float) / counts.sum() 
    support = (edges[:-1] + edges[1:]) / 2 
    return support, empirical_dist, width  

class GirvanNewman(object): 
    '''
    Methods to implement the Girvan-Newman algorithm. 
    '''
    @staticmethod 
    def hierarchy_branching(hierarchy: List, depth: int): 
        '''
        Choose a branch for the hierarchy `hierarchy`. 
        '''
        import itertools 
        iterations = itertools.takewhile(lambda c: len(c) <= depth, hierarchy) 
        for communities in iterations: 
            cluster = communities 
        return cluster 
    
    @staticmethod 
    def sample_connected_component(graph: nx.DiGraph, suppliers: List, consumers: List): 
        '''
        Choose uniformly at random a connected component of the graph for the Girvan-Newman algorithm. 
        
        The lists `sources` and `targets` correspond to the components of the subsequently computed edge 
        current flow centrality. 
        '''
        connected_components = list(nx.connected_components(graph)) 
        inx = np.random.randint(len(connected_components)) 
        comp = graph.subgraph(connected_components[inx]) 
        sources = suppliers.intersection(comp.nodes) 
        targets = consumers.intersection(comp.nodes) 
        return sources, targets, comp 

    @staticmethod 
    def most_valuable_edge(graph):  
        if not nx.is_connected(graph): 
            # Sample a connected component uniformly at random 
            sources, targets, comp = GirvanNewman.sample_connected_component(
                    graph=graph, 
                    suppliers=suppliers, 
                    consumers=consumers
                ) 
        else: 
            sources = suppliers 
            targets = consumers 
            comp = graph
        centrality = nx.edge_current_flow_betweenness_centrality_subset( 
                G=comp, 
                sources=sources, 
                targets=targets, 
                weight='weight' 
        ) 
        # print(sources, targets) 
        return max(centrality, key=centrality.get) 


def dictators(community: List, graph: nx.DiGraph, community_size: int, directory: str, community_inx: int): 
    '''
    Capture the dictators: nodes with either a highly discrepant degree or highly discrepant 
    clustering coefficient. 
    '''
    if len(community) < community_size: 
        return 
    # Compute the subgraph induced by this community 
    community_graph = graph.subgraph(community) 
    community_graph = community_graph.to_undirected() 
    clustering_coefficients = nx.clustering(community_graph)    # Clustering for node
    support, empirical_dist, width = empirical_distribution(clustering_coefficients, scale='linear', nbins=9) 
    plt.bar(
        support, 
        empirical_dist, 
        color='g', 
        edgecolor='gray', 
        width=width) 
    plt.title('The local clustering coefficient distribution.') 
    plt.xlabel('Clustering coefficint') 
    plt.ylabel('Density') 
    plt.savefig(f'{directory}/dictators{community_inx}.png') 
    # plt.show() 

def voracious(community: List, graph: nx.DiGraph, community_size: int, directory: str, community_inx: int): 
    '''
    Identify the players within the community `community` with a discrepant 
    participation in the transactions. 
    '''
    if len(community) < community_size: 
        return 
    # Capture the subgraph induced by the community 
    community_graph = graph.subgraph(community) 
    degrees = [d for (n, d) in community_graph.degree if graph.nodes.data('type')[n] == MarketEnum.TRADER] 
    empirical_dist = np.bincount(degrees) 
    support = np.argwhere(empirical_dist != 0) 
    plt.scatter(support, empirical_dist[support]) 
    plt.title('The degree distribution within a community-induced subgraph for traders.') 
    plt.xlabel('Degree') 
    plt.ylabel('Density') 
    plt.savefig(f'{directory}/voracious{community_inx}.png') 
    # plt.show() 

def update_weighted_betweenness(betweenness: Dict[str, int], paths: List, graph: nx.DiGraph): 
    '''
    Update the dictionary `betweenness` with the data at `paths`. 
    '''
    attributes = nx.get_node_attributes(graph, 'type') 
    for path in paths: 
        for inx in range(1, len(path)): 
            if attributes[path[inx]] != MarketEnum.TRADER: continue 
            try: 
                weight = graph.get_edge_data(path[inx-1], path[inx])['weight']  
            except KeyError: 
                weight = 1 
            betweenness[path[inx]] += weight 

def main_players(graph: nx.DiGraph, directory: str): 
    '''
    Identify the main players within the graph `graph`; the importance metric equals the 
    volume of the paths between the suppliers and the consumers that traverse each node. 
    '''
    suppliers = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.SUPPLIER] 
    traders = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.TRADER] 
    consumers = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.CONSUMER] 
    betweenness = {trader:int(1e-19) for trader in traders} 
    connected_components = nx.weakly_connected_components(graph) 
    for component in tqdm(connected_components): 
        connected_graph = graph.subgraph(component) 
        if len(component) < 4: 
            # Statistical noise 
            continue 
        print(f'Computing random walk betweenness centrality for component of length {len(component)}') 
        random_walk_betweenness = nx.approximate_current_flow_betweenness_centrality(
                connected_graph.to_undirected(), 
                weight='weight', 
                solver='full', 
                seed=42) 
        print('Assembling this metric') 
        for node in random_walk_betweenness: 
            if node not in betweenness: continue 
            betweenness[node] += random_walk_betweenness[node] 
    # It is computationally exhaustive 
    # for supplier in tqdm(suppliers): 
        # paths = nx.all_simple_paths(graph, source=supplier, target=consumers) 
        # update_weighted_betweenness(betweenness, paths, graph=graph) 
    volumes = {trader:volume for (trader, volume) in betweenness.items() if volume > 0} 
    scale = 'linear'
    print('Computing empirical distribution for the betweenness centrality') 
    support, empirical_dist, width = empirical_distribution(quantities=volumes, scale=scale, nbins=32) 
    volumes = np.argwhere(empirical_dist != 0) 
    print('Generating a scatter plot with this distribution') 
    plt.scatter(support[volumes], empirical_dist[volumes]) 
    plt.title('The distribution of the volume traded by the traders.') 
    plt.xlabel('Volume') 
    plt.ylabel('Density') 
    plt.savefig(f'{directory}/main_players.png') 
    plt.xscale(scale) 
    plt.ylim(0, max(empirical_dist)*1.1) 
    # plt.show() 
    plt.clf() 
    json.dump(betweenness, open(f'{directory}/main_players.json', 'w')) 
    return betweenness 

def bridges(graph: nx.DiGraph, directory: str): 
    ''' 
    Compute the bridges within the Market; they conform highly crucial nodes which, 
    if extracted, would crucially disrupt the market structure. 
    '''
    # `Intermediação` is the characterization of `Betweenness` 
    betweenness = nx.betweenness_centrality(graph) 
    support, empirical_dist, width = empirical_distribution(quantities=betweenness, scale='linear', nbins=19)  
    plt.scatter(support[1:], empirical_dist[1:])    # The initial quantities are highly biased  
    plt.xlabel('Betweenness') 
    plt.ylabel('Density') 
    plt.title('The distribution of the betweenness centraility within the network.') 
    plt.savefig(f'{directory}/bridges.png') 
    plt.clf() 
    # plt.show() 
        
def draw_communities(graph: nx.DiGraph, communities: List[str]): 
    '''
    Write a graph with the nodes colored according to their communities. 
    '''
    mplcolors = list(mcolors.BASE_COLORS) 
    colors = OrderedDict() 
    for inx, community in enumerate(communities): 
        colors.update({node:mplcolors[inx] for node in community}) 
    # Choose properly the nodes' coordinates 
    eps = 2 
    compute_coordinates = { 
        MarketEnum.SUPPLIER: lambda: np.array([1, 1 + np.random.uniform(-eps, eps)]),  
        MarketEnum.TRADER: lambda: np.array([3, 1 + np.random.uniform(-eps, eps)]), 
        MarketEnum.CONSUMER: lambda: np.array([5, 1 + np.random.uniform(-eps, eps)])  
    } 
    coordinates = {node:compute_coordinates[attribute]() for (node, attribute) in graph.nodes.data('type')} 
    nx.draw( 
        graph, 
        pos=coordinates, 
        node_color=[colors[node] for node in graph.nodes], 
        with_labels=True
    ) 
    plt.show() 

def merge_boundaries(graph: nx.Graph): 
    '''
    Merge the boundaries of a graph, as assigned by the attributes `MarketEnum.SUPPLER` and `MarketEnum.CONSUMER`. 
    The objective is that the merged nodes would mitigate the statistical fluctuations of the communities' 
    identification algorithms. 
    '''
    suppliers = [node for (node, attribute) in graph.nodes.data('type') if attribute == MarketEnum.SUPPLIER] 
    consumers = [node for (node, attribute) in graph.nodes.data('type') if attribute == MarketEnum.CONSUMER] 
    
    merged_supplier = 'MERGED_SUPPLIER' 
    merged_consumer = 'MERGED_CONSUMER' 

    merged_supplier_edges = dict() 
    merged_consumer_edges = dict() 
    # Compute the neighbors of each supplier and identify the edge's weight 
    for supplier in tqdm(suppliers): 
        edges = graph.out_edges(supplier)  
        for edge in edges: 
            supplier, trader = edge
            if trader not in merged_supplier_edges: merged_supplier_edges[trader] = int(1e-19) 
            merged_supplier_edges[trader] += graph.edges[edge]['weight'] 
    for consumer in tqdm(consumers): 
        edges = graph.in_edges(consumer) 
        for edge in edges: 
            trader, consumer = edge 
            if trader not in merged_consumer_edges: merged_consumer_edges[trader] = int(1e-19) 
            merged_consumer_edges[trader] += graph.edges[edge]['weight'] 
    graph.remove_nodes_from(suppliers) 
    graph.remove_nodes_from(consumers) 
     
    graph.add_node(merged_supplier) 
    graph.add_node(merged_consumer) 
    
    nx.set_node_attributes(
        graph, 
        values={merged_supplier: MarketEnum.SUPPLIER, merged_consumer: MarketEnum.CONSUMER}, 
        name='type')
    for (target, weight) in merged_supplier_edges.items(): 
        graph.add_edge(merged_supplier, target, weight=weight) 
    for (target, weight) in merged_consumer_edges.items(): 
        graph.add_edge(target, merged_consumer, weight=weight) 
     
    # Return the merged graph 
    return graph 

def locally_dense_markets(graph: nx.DiGraph, directory: str): 
    '''
    Identify locally dense markets through the identification of communities with the 
    Girvan-Newman's algorithm. 
    '''
    largest_connected_component = max(nx.weakly_connected_components(graph), key=len) 
    graph = graph.subgraph(largest_connected_component)
    merged_graph = merge_boundaries(graph.copy()) 
    suppliers = set(node for (node, type) in graph.nodes.data('type') if type == MarketEnum.SUPPLIER) 
    consumers = set(node for (node, type) in graph.nodes.data('type') if type == MarketEnum.CONSUMER) 

    print('Executing Girvan-Newman'"'"' algorithm for community identification') 
    communities = community.louvain_communities(merged_graph, weight='weight', seed=42, resolution=.5) 
    # communities = community.girvan_newman(graph, most_valuable_edge=GirvanNewman.most_valuable_edge) 
    # Identify local markets 
    print('Choosing communities at a prescribed hierarchic depth') 
    print(f'Identified communities: {len(communities)}') 
    print(f'Average quantity of nodes per community: {len(merged_graph.nodes) / len(communities):.2f}') 
    # communities = GirvanNewman.hierarchy_branching(hierarchy=communities, depth=2) 
    if len(graph.nodes) < 99: 
        draw_communities(graph=merged_graph, communities=communities) 
    else: 
        print('The network is tremendously large; we should not attempt to draw it.') 

    for inx, component in tqdm(enumerate(communities)): 
        if len(component) < 1e-2 * len(merged_graph.nodes): 
            continue 
        # print(component) 
        dictators( 
            community=component, 
            graph=merged_graph, 
            community_size=3, 
            directory=directory, 
            community_inx=inx 
        ) 
        voracious( 
            community=component, 
            graph=merged_graph, 
            community_size=3, 
            directory=directory, 
            community_inx=inx 
        ) 


if __name__ == '__main__': 
    args = parse_args() 
    graph = nx.read_graphml(args.source) 
    if not os.path.exists(args.output): 
        os.mkdir(args.output) 
    main_players(graph=graph.copy(), directory=args.output) 
    locally_dense_markets(graph=graph.copy(), directory=args.output) 
    bridges(graph=graph.copy(), directory=args.output) 

