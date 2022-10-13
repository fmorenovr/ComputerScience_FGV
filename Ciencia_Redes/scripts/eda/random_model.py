'''
Generate a random graph consistent with the Supplier, Trader, Consumer (STC) network topology. 

Alas, it is inappropriate for large graphs. 
'''
import os 
import numpy as np 
import networkx as nx 

from collections import OrderedDict 
import matplotlib.pyplot as plt 

# Documentation 
from typing import List, Dict 
import argparse 
from abc import ABC, abstractmethod 
from tqdm import tqdm 

class RandomGraph(ABC): 

    def __init__(self, graph: nx.DiGraph): 
        '''
        The graph `graph` is the baseline for the random model. 
        '''
        pass 

    @abstractmethod 
    def is_consistent(self, graph: nx.DiGraph): 
        '''
        A criterion to assert that the random graph is constructed. 
        '''
        pass 
    
    @abstractmethod 
    def sample(self): 
        '''
        Sample a graph according to the underlying generative process. 
        '''
        pass 

class MarketEnum(object): 
    SUPPLIER = 'MANEJO' 
    TRADER = 'PTO' 
    CONSUMER = 'FINAL' 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Generate a random graph topologically consistent with a real network.') 
    parser.add_argument('--source', 
            type=str, 
            required=True, 
            help='The `graphml` file with the structural properties of the subsequently generated graphs.') 
    parser.add_argument('--model', 
            type=str, 
            default='degrees', 
            help='The model for the generation of the random graph; either `degrees` or `growth`.') 
    # Parse the command line parameters 
    args = parser.parse_args() 
    # Return the parsed command line parameters 
    return args 

class RandomModel(RandomGraph): 
    ''' 
    A random graph (configurationally) consistent with the STC topology.  
    '''
    
    nodes: List 
    degrees: List 

    def __init__(self, graph: nx.DiGraph): 
        '''
        Constructor method for `RandomModel`.
        '''
        self.nodes = list(graph.nodes) 
        self.in_degrees = [graph.in_degree[n] for n in self.nodes] 
        self.out_degrees = [graph.out_degree[n] for n in self.nodes] 
        self.types = nx.get_node_attributes(graph, 'type') 
        
        self.in_stubs = np.array(self.in_degrees).astype(float) 
        self.out_stubs = np.array(self.out_degrees).astype(float) 

    def is_consistent(self, graph: nx.DiGraph): 
        ''' 
        Assert whether the graph `graph` is consistent with the structural attributes of `self`: 
        in_degrees and out_degrees. 
        '''
        # `plsbl` stands for `plausible`
        return (self.in_stubs == 0).all() and (self.out_stubs == 0).all()
    
    def sample_edge(self, graph: nx.DiGraph): 
        '''
        Sample an edge at random; the probability of a node being choosen is proportional 
        to its current available degrees. 
        '''
        nodes_inxs = np.arange(len(self.nodes)) 
        out_probs = self.out_stubs.copy() 
        in_probs = self.in_stubs.copy()  
        out_probs /= self.out_stubs.sum() 
        in_probs /= self.in_stubs.sum() 
        out_node = np.random.choice(nodes_inxs, p=out_probs) 
        in_node = np.random.choice(nodes_inxs, p=in_probs) 
        # The scenarios in which a node joins itself are not statistically disruptive 
        self.out_stubs[out_node] -= 1 
        self.in_stubs[in_node] -= 1 
        return self.nodes[out_node], self.nodes[in_node]

    def sample(self): 
        '''
        Sample a random network whose degrees are preserved as well as the kind-of-tripartite topology. 
        '''
        graph = nx.DiGraph() 
        graph.add_nodes_from(self.nodes) 
        nx.set_node_attributes(graph, self.types, name='type') 
        
        pbar = tqdm(total=sum(self.in_degrees)) 
        while not self.is_consistent(graph): 
            random_edge = self.sample_edge(graph)  
            graph.add_edge(*random_edge) 
            pbar.update(1) 

        # Return the randomly generated graph 
        return graph 

class BarabasiAlbert(RandomGraph): 
    '''
    Implant the Barábasi-Albet model for linear preferential attachment in networks. 
    '''

    def __init__(self, graph: nx.DiGraph): 
        '''
        Constructor method for this version of the BA algorithm; we use `graph` 
        as the baseline for the quantity of nodes for each type in the utterly networkx. 

        The initial state is a market with a supplier, a trader and a consumer. 
        '''
        self.random_graph = nx.DiGraph
        self.nodes = graph.nodes 

        # This is not a degree preserving algorithm; however, 
        # we should guarantee that the nodes' degrees are consistent: 
        # consumers are not traders, and suppliers are not consumers. 
        self.in_degrees = graph.in_degree
        self.out_degrees = graph.out_degree 

    @property 
    def initial_state(self): 
        '''
        Instantiate an initial state for the BA model; the market starts with 
        a supplier, a costumer and a trader betweem them. 
        '''
        suppliers = BarabasiAlbert.nodes_from_type(self.nodes, MarketEnum.SUPPLIER, 'type') 
        traders = BarabasiAlbert.nodes_from_type(self.nodes, MarketEnum.TRADER, 'type') 
        consumers = BarabasiAlbert.nodes_from_type(self.nodes, MarketEnum.CONSUMER, 'type') 

        supplier = np.random.choice(suppliers) 
        trader = np.random.choice(traders) 
        consumer = np.random.choice(consumers) 
        initial_graph = nx.DiGraph() 
        initial_graph.add_nodes_from([supplier, trader, consumer]) 
        
        nx.set_node_attributes(G=initial_graph, 
                values={
                supplier: MarketEnum.SUPPLIER, 
                trader: MarketEnum.TRADER, 
                consumer: MarketEnum.CONSUMER 
                }, 
                name='type'
        ) 

        initial_graph.add_edge(supplier, trader) 
        initial_graph.add_edge(trader, consumer) 
        # Return the initial state of the graph 
        return initial_graph 
    
    def sample_node(self, graph: nx.DiGraph): 
        '''
        Sample a node to attach to the pool of nodes in `graph`. 
        '''
        p = np.ones_like(self.nodes, dtype=float) 
        mask = np.array([n in graph.nodes for n in self.nodes], dtype=float) 
        p *= (1 - mask)  
        p /= p.sum() 
        node = np.random.choice(self.nodes, p=p)  
        type = self.nodes.data('type')[node]         
        # Return the randomly sampled node 
        return node, type 
    
    def sample_edge(self, graph: nx.DiGraph, attr: str, node: str): 
        '''
        Sample an edge for the graph `graph` accordingly with the linear preferential attachment model. 
        '''
        suppliers = BarabasiAlbert.nodes_from_type(graph.nodes, MarketEnum.SUPPLIER, 'type') 
        traders = BarabasiAlbert.nodes_from_type(graph.nodes, MarketEnum.TRADER, 'type') 
        consumers = BarabasiAlbert.nodes_from_type(graph.nodes, MarketEnum.CONSUMER, 'type') 
        degrees = graph.degree 
        if attr == MarketEnum.SUPPLIER: 
            # Sample a trader 
            trader = BarabasiAlbert.linear_attachment_sample(
                    traders, 
                    degrees, 
                    smoothing=True) 
            return [(node, trader)] 
        elif attr == MarketEnum.TRADER: 
            # Sample a supplier and either a consumer or another trader 
            consumer = BarabasiAlbert.linear_attachment_sample(
                    traders + consumers, 
                    degrees, 
                    smoothing=True) 
            supplier = BarabasiAlbert.linear_attachment_sample(
                    suppliers, 
                    degrees, 
                    smoothing=True) 
            return [(node, consumer), (supplier, node)] 
        elif attr == MarketEnum.CONSUMER: 
            # Sample a trader to attach to the consumer 
            trader = self.linear_attachment_sample(
                    traders, 
                    degrees, 
                    smoothing=True) 
            return [(trader, node)] 
        else: 
            raise ValueError(f'`attr` should be in \
                    {[MarketEnum.SUPPLIER, MarketEnum.TRADER, MarketEnum.CONSUMER]}.') 
    
    @staticmethod 
    def linear_attachment_sample(nodes: List, degrees: Dict[str, int], smoothing: bool=False): 
        '''
        Sample a node from the list `nodes` according to the linear preferential attachment model. 
        '''
        p = np.array([degrees[n] for n in nodes], dtype=float) 
        if smoothing: 
            p += 1          # A Dirichlet prior regularization  
        p /= p.sum()        # Compatibility with NumPy's random choice 
        node = np.random.choice(nodes, p=p) 
        # Return the randomly sampled node 
        return node 

    @staticmethod 
    def nodes_from_type(nodes: nx.DiGraph, attr: str, name: str): 
        '''
        Compute the nodes within `nodes` with the attribute `attr` equal to `name`.  
        '''
        return [n[0] for n in nodes.data(name) if n[1] == attr] 
    
    def is_consistent(self, graph: nx.DiGraph): 
        '''
        Verify whether the randomly generated graph `graph` is consistent with the baseline `self.graph`. 
        '''
        return len(graph) == len(self.nodes) 

    def sample(self): 
        '''
        Sample a graph according to an adapation of the BA model for linear preferential attachment. 
        '''
        graph = self.initial_state 
        while not self.is_consistent(graph):
            node, attr = self.sample_node(graph=graph) 
            edges = self.sample_edge(attr=attr, node=node, graph=graph) 
            nx.set_node_attributes(graph, {node: attr}, name='type') 
            graph.add_edges_from(edges) 
        # Return the random graph 
        return graph 

if __name__ == '__main__': 
    args = parse_args() 
    graph = nx.read_graphml(args.source) 
    model = RandomModel(graph) 
    random_graph = model.sample() 
    nx.draw(random_graph) 
    plt.show() 
    model = BarabasiAlbert(graph) 
    random_graph = model.sample() 
    nx.draw(random_graph) 
    plt.show() 
