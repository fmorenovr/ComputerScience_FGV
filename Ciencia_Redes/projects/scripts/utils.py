""" 
General utility procedures for networks. 
""" 
import os # It will be okay, okay, okay, okay; believe in yourself, honestly
# you are scared, you are 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

def degree_distribution(
        graph: nx.Graph, 
        logarithm_binning: bool=False, 
        fname: str="degree_distribution.png", 
        n_bins: int=9
    ): 
    """ 
    Compute the degree distribution and plot the histogram, possibly 
    with logarithmic binning. 
    """ 
    degrees = [d for (n, d) in graph.degree] 
    # print(degrees) 
    if logarithm_binning: 
        log_degrees = np.log10(degrees) 
        bins = np.logspace(min(log_degrees), max(log_degrees), num=n_bins) 
    else: 
        bins = np.linspace(min(degrees), max(degrees), num=n_bins) 
    counts, bins = np.histogram(degrees, bins=bins) 
    # print(bins, counts) 
    bins_midst = [(bins[i] + bins[i+1]) / 2 for i in np.arange(len(bins)-1)] 
    plt.scatter(x=bins_midst, y=counts) 
    plt.show() 
    # Return the graph's degree distribution 
    return counts, bins 

def capture_cycles(graph: nx.Graph): 
    """ 
    Capture the cycles within the graph `graph`. 
    """ 
    cycles = nx.simple_cycles(graph) 
    # Compute the frequency of the nodes in cycles 
    frequency = {node:int(1e-19) for node in graph.nodes} 
    for cycle in tqdm(cycles):
        for node in cycle: 
            frequency[node] += 1 
    # Return the cycles and the nodes' frequencies therein 
    return cycles, frequency 

def perpetual_markets(graph: G): 
    """ 
    Identify the perpetual markets within the graph `G`; conveniently, if 
    we join the consumer nodes with the source nodes with a directed edge for each 
    connected component (weakly), the induced subgraph should be strongly connected. 
    """ 
    weakly_connected_components = nx.weakly_connected_components(graph) 
    cyclic_markets = list() 
    for wcc in weakly_connected_components: 
        sgraph = graph.subgraph(wcc) 
        # Capture each node with type `FINAL` and `MANEJO` 
        nodes = {node["type"]: node for node in sgraph.nodes} 
        for final_node in nodes["final"]: 
            for manejo_node in nodes["manejo"]: 
                sgraph.add_edge(final_node, manejo_node) 
        is_strongly_connected = nx.is_strongly_connected(sgraph) 
        if not is_strongly_connected: 
            cyclic_markets.append(wcc) 
    # Return the nodes for each weakly connected component 
    # which does not induce a strongly connected component 
    return cyclic_markets

def asymmetric_players(graph: nx.Graph, threshold: float, type: str="FINAL"): 
    """ 
    Capture the asymmetric players, whose buying behavior is 
    statistically distinguishable from the typical customers. 
    """ 
    # Compute the degrees for each player 
    in_degrees = graph.in_degree 
    customers = np.array([node for (node, dg) in in_degrees.items() if node["type"] == type]) 
    degrees = np.array([dg for (node, dg) in in_degrees.items() if node["type"] == type]) 
    median_degree = np.median(degrees) 
    std_degree = np.abs(degrees - median_degree).mean() 
    extravagant_inxs = ((degrees - median_degree) / std_degree) > threshold 
    extravagant_degrees = degrees[extravagant_inxs] 
    extravagant_customers = customers[degrees == extravagant_degrees] 
    # Return the extravagant customers 
    return extravagant_customers

if __name__ == "__main__": 
    graph = nx.erdos_renyi_graph(9, .4, directed=True) 
    degree_distribution = degree_distribution(graph, logarithm_binning=True, n_bins=19) 
    cycles, frequency = capture_cycles(graph) 
    print(frequency, cycles) 

