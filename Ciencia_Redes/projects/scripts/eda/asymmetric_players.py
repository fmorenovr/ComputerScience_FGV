'''
Use different criteria to identify the asymmetric players within the customers' market. 

python scripts/eda/asymmetric_players.py    --source graphs/wood_years_2018/wood.graphml \
                                            --output_dir graphs/wood_years_2018/figures
'''
import os 
import numpy as np 
import networkx as nx 

import numpy as np 
from collections import OrderedDict 
import matplotlib.pyplot as plt 
import json 
import pandas as pd 

import sys 
sys.path.append('./scripts/eda')
from random_model import MarketEnum 
from monopolies import empirical_distribution 

import argparse 
from typing import Any, Dict, List 

plt.style.use(os.environ['MPL_STYLE']) 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Evaluate nodes by different metrics, identifying those who are excessively asymmetric.') 
    parser.add_argument('--source', 
            type=str, 
            required=True, 
            help='The `graphml` source of the graph.') 
    parser.add_argument('--stats', 
            type=str, 
            default=None, 
            help='The directory to the precomputed statistics for the graph; optional, although advised.') 
    parser.add_argument('--output_dir', 
            type=str, 
            required=True, 
            help='The directory to which the images and the evaluations will be sent.') 
    parser.add_argument('--target_nodes', 
            type=str, 
            default='PTO', 
            help='The type of the node asymmetry we aim to assess; typically, the `PTO` attribute.') 
    # Parser the command line parameters 
    args = parser.parse_args() 
    # Return the parsed command line parameters 
    return args 

def asymmetry(data: OrderedDict, threshold: float): 
    '''
    Use the absolute standard deviation with respect to the median to identify asymmetric keys 
    within the dictionary `data`. 
    '''
    keys, values = list(data.keys()), list(data.values()) 
    values = np.array(values) 
    median = np.median(values) 
    deviation = np.abs(values - median) / median 
    discrepancies_inxs = np.argwhere(deviation > threshold).reshape(-1)   
    return [keys[inx] for inx in discrepancies_inxs] 

def histogram(data: OrderedDict, logarithmic_binning: bool, fname: str, xlabel: str, ylabel: str): 
    '''
    Write the histogram for the quantities computed at `data`; the objective is to identify 
    (with a perceptual assessment) the existence of discrepant nodes. 
    '''
    values = np.array(list(data.values())) 
    if logarithmic_binning: 
        bins = np.logspace(np.log2(values[values > 0].min()), 
                           np.log2(values.max()), endpoint=True, base=2) 
        plt.xscale('log') 
    else: 
        bins = np.linspace(values.min(), values.max(), endpoint=True) 
    histdata, bin_edges = np.histogram(values, bins=bins) 
    plt.scatter(
        x=(bins[1:] + bins[:-1]) / 2, 
        y=histdata, 
        cmap='gray'
    ) 
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.savefig(fname) 
    plt.cla() 
    # Return the histogram data 
    return histdata 

def evaluate_trader_activity(
        G: nx.DiGraph, 
        threshold: float, 
        output_dir: str, 
        type: str, 
        target_nodes: str=MarketEnum.TRADER): 
    '''
    Evaluate the nodes' activity in the market, which is measured through its (weighted) degree. 
    The alternative is to evaluate its importance, as measured by the local clustering coefficient, 
    and its ubiquity, as characterized by its (unweighted) degree. 

    The threhold is used to evaluate whether a node is excessively discrepant from the median. 
    '''
    assert type in ['activity', 'ubiquity'], '`type` should equal either `activity` or `ubiquity`'
    weights = OrderedDict()                         # Map the node to its activity 
    nodesattr = nx.get_node_attributes(G, 'type')   # Capture the nodes types: supplier, trader, consumer
    for node in G.nodes: 
        if (target_nodes is not None and nodesattr[node] not in target_nodes): continue 
        edges = G.out_edges(node) 
        if type == 'activity': 
            traded_volume = sum([graph.get_edge_data(*edge, default=1)['weight'] for edge in edges])   
        elif type == 'ubiquity': 
            traded_volume = len(edges) 
        if traded_volume == 0: 
            # This node did not bestow itself in a trade in this network 
            continue 
        weights[node] = traded_volume  
    asymmetric_nodes = asymmetry(data=weights, threshold=threshold) 
    histogram(
        data=weights, 
        logarithmic_binning=True, 
        fname=os.path.join(output_dir, f'asymmetric_{type}.png'), 
        xlabel=f'{type}', 
        ylabel='count'
    ) 
        
    # Save the appropriate quantities 
    json.dump(
        {'metric': f'{type}', 'data': asymmetric_nodes}, 
        open(os.path.join(output_dir, f'asymmetric_{type}.json'), 'w')
    ) 

def evaluate_trader_importance(
        G: nx.DiGraph, 
        degree_threshold: float, 
        output_dir: str, 
        target_nodes: str=MarketEnum.TRADER): 
    '''
    Evaluate the trader's importance, as measured through its control of the flow of information within 
    the network. 
    '''
    undirected_graph = G.to_undirected(reciprocal=False) 
    clustering_coefficients = OrderedDict(nx.clustering(undirected_graph)) 
    attributes = nx.get_node_attributes(undirected_graph, 'type') 
    is_target_node = lambda node: undirected_graph.degree[node] > degree_threshold and \
                            (target_nodes is not None and attributes[node] in target_nodes) 
    clustering_coefficients = {node:cc for (node, cc) in clustering_coefficients.items() \
            if is_target_node(node)} 
    histogram( 
        data=clustering_coefficients, 
        logarithmic_binning=False, 
        fname=os.path.join(output_dir, 'asymmetric_importance.png'), 
        xlabel='clustering_coefficient', 
        ylabel='counts'
    ) 
    # Sort the nodes with respect to their clustering coefficients 
    clustering = pd.DataFrame({
        'node': clustering_coefficients.keys(), 
        'cc': clustering_coefficients.values() 
    }) 
    clustering.sort_values(by='cc', inplace=True) 
    clustering.to_csv(index=None, path_or_buf=os.path.join(output_dir, 'asymmetric_importance.csv')) 
    # Return the nodes' clustering coefficients 
    return clustering 

def compute_waste(graph: nx.DiGraph, node: Any, attributes: Dict[Any, float]): 
    '''
    Compute the waste of primary volume for the transactions executed by the node `node`. 
    '''
    predecessors = graph.predecessors(node) 
    involume = [attributes[(p, node)] for p in predecessors] 
    successors = graph.successors(node) 
    outvolume = [attributes[(node, s)] for s in successors] 
    # The waste equals the difference between its out (weighted) degree and its 
    # in (weighted) degree 
    waste = sum(outvolume) - sum(involume) 
    return waste 

def evaluate_trader_waste( 
        graph: nx.DiGraph, 
        output_dir: str 
    ): 
    '''
    Evaluate the volume of traded wood wasted by the trader. It is, in effect, a 
    measure of the efficiency of the market, as we would auspiciously examine a 
    scenario in which each unity of volume is sent to the player. 
    '''
    traders = [node for (node, property) in graph.nodes.data('type') if property == MarketEnum.TRADER] 
    # Compute, for each trade for the trader, the bought volume of wood the 
    # the volume that was sent to a consumer 
    weights = {(u, v): w for (u, v, w) in graph.edges.data('weight')} 
    # print(weights) 
    wastes = {trader:int(1e-19) for trader in traders} 
    for trader in traders: 
        wastes[trader] = compute_waste(
                graph=graph, 
                node=trader, 
                attributes=weights) 
        # print(trader, wastes[trader]) 
    # Compute the distribution of the wastes 
    support, empirical_dist, width = empirical_distribution(wastes, scale='linear', nbins=32) 
    plt.bar(x=support, height=empirical_dist, width=width) 
    plt.title('The distribution of wasted timber for the traders.') 
    plt.xlabel('Wasted volume.') 
    plt.ylabel('Density.') 
    plt.savefig(f'{output_dir}/commodities.png') 
    plt.show() 
    return wastes 

if __name__ == '__main__': 
    args = parse_args() 
    if not os.path.exists(args.output_dir): 
        os.mkdir(args.output_dir) 
    graph = nx.read_graphml(args.source) 
    evaluate_trader_activity(
            graph, 
            threshold=2, 
            type='activity', 
            target_nodes=args.target_nodes, 
            output_dir=args.output_dir) 
    evaluate_trader_activity(
            graph, 
            threshold=2, 
            type='ubiquity', 
            target_nodes=args.target_nodes, 
            output_dir=args.output_dir) 
    evaluate_trader_importance(
            graph, 
            degree_threshold=5, 
            output_dir=args.output_dir, 
            target_nodes=args.target_nodes) 
    evaluate_trader_waste( 
            graph=graph, 
            output_dir=args.output_dir 
    ) 
