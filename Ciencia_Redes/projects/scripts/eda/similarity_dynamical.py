'''
Identify the dynamical similarity between the nodes of different temporal instantiations of a 
complex system. 

python scripts/eda/similarity_dynamical.py \
        --graphs similarities/similarity_i.graphml,similarities/similarity_ii.graphml \
        --output similarities 
'''
import os 
import numpy as np 
import networkx as nx 
import pandas as pd 

import torch 
import torch.nn as nn 

import scipy.optimize 

import matplotlib.pyplot as plt 
import json 

from functools import partial 
# Documentation 
import argparse 
from typing import List, Dict, Tuple
from tqdm import tqdm 

np.random.seed(42)  # Seed for reproducibility 
torch.random.manual_seed(42) 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Compute similar nodes in different networks through a spectral evaluation.') 
    parser.add_argument('--graphs', type=str, required=True, help='A comma separated list of paths for the `graphml` files.') 
    parser.add_argument('--output', type=str, required=True, help='The output directory to which the estimates would be sent.') 
    return parser.parse_args() 

def shared_players(graphs: Tuple[nx.Graph, nx.Graph]): 
    '''
    Compute the players shared by the graphs within the tuple `graphs`. 
    '''
    markets = list(graph.copy() for graph in graphs)  
    markets[:] = map(lambda graph: set(graph.nodes), graphs) 
    intersections = markets[1].intersection(markets[0]) 
    return list(intersections) 

def compute_adjacency_matrix(graph: nx.Graph, shared_nodes: List[str]): 
    '''
    Compute the adjacency matrix for the (undirected) graph `graph`; 
    the initial nodes in the rows and columns indexations are contemplated in `shared_nodes`.
    '''
    nodes = set(graph.nodes) 
    nodes = nodes.difference(shared_nodes) 
    nodes = list(nodes) 
    adj_matrix = nx.adjacency_matrix(graph, nodelist=shared_nodes+nodes, weight=None) 
    adj_matrix = adj_matrix[:, :len(shared_nodes)]   
    return adj_matrix, nodes

def interadjacency_matrix(graphs: Tuple[nx.Graph, nx.Graph]): 
    '''
    Write an adjacency matrix between the graphs within the tuple `graphs`: 
    the component `ij` equals the unity if j, a node within Graph II, is tied to i, 
    a node within Graph I; otherwise, it is a null element. 

    Importantly, notice that, if `j` is not in Graph I, then `ij` is null in the matrix. 

    Actually, the component `ij` of this collective adjacency matrix asseverates 
    the quantity of shared nodes tied to node `i` and node `j`. It is a measure of structural 
    equivalence between graphs. 
    '''
    shared_nodes = shared_players(graphs) 
    # Consistently order the nodes 
    graph_i, graph_ii = graphs 
    adj_i, nodes_i = compute_adjacency_matrix(graph_i, shared_nodes=shared_nodes) 
    adj_ii, nodes_ii = compute_adjacency_matrix(graph_ii, shared_nodes=shared_nodes) 
    adj = adj_i @ adj_ii.T 
    from sklearn.preprocessing import normalize 
    adj = normalize(adj, norm='max', axis=1)   
    # adj /= adj.sum(axis=1) 
    print(adj) 
    return adj, shared_nodes, shared_nodes+nodes_i, shared_nodes+nodes_ii 

def similarity(sigma: np.array, adjacency: np.array, prior: np.array, alpha: float=.9): 
    '''
    Assess the similarity simlarity score between the graphs' nodes, which 
    equals a fixed point of the function 

    sigma \mapsto A \sigma.T A + P, 

    in which P equals a prior similarity. 
    '''
    # Consistency with SciPy, which uses vectors instead of matrices 
    neighbors = alpha * adjacency @ sigma.T @ adjacency + prior 
    return neighbors  

def prior_similarity(
        graphs: Tuple[nx.DiGraph, nx.DiGraph], 
        adjacency: np.array, 
        shared_nodes: List[str]): 
    '''
    Compute the prior similarity between the nodes. 
    '''
    graph_i, graph_ii = graphs 
    prior_sml = adjacency.todense() 
    prior_sml[:len(shared_nodes), :len(shared_nodes)] = 0 
    # prior_sml[len(shared_nodes):, len(shared_nodes):] = adjacency[len(shared_nodes):, len(shared_nodes):] 
    prior_sml[tuple([np.arange(len(shared_nodes))])*2] = 1 
    # Normalize the rows' values 
    for i, row in enumerate(prior_sml): 
        rowsum = row.sum() 
        if rowsum != 0: 
            prior_sml[i, :] /= rowsum 
    # prior_sml /= prior_sml.sum(axis=1) 
    print(prior_sml) 
    return prior_sml 

class Solver(nn.Module): 

    def __init__(
            self, 
            adjacency: float, 
            prior_sml: np.array, 
            regularization: float, 
            device: str='cuda'
        ): 
        super(Solver, self).__init__() 
        initial_value = torch.tensor(prior_sml, dtype=torch.float32) + torch.randn(prior_sml.shape) 
        self.adjacency = torch.tensor(
                adjacency.todense(), 
                requires_grad=False, 
                dtype=torch.float32, 
                device=device) 
        self.prior_sml = torch.tensor(
                prior_sml, 
                requires_grad=False, 
                dtype=torch.float32, 
                device=device) 
        self.act = nn.Sigmoid()  
        
        self.sigma_ = nn.Parameter(
                similarity(sigma=initial_value.to(device), 
                    adjacency=self.adjacency, 
                    prior=self.prior_sml), 
                requires_grad=True 
            ) 
        
        self.similarity = partial(
                similarity, 
                adjacency=self.adjacency, 
                prior=self.prior_sml, 
                alpha=regularization 
        ) 

    def forward(self): 
        sigma = self.similarity(sigma=self.sigma) 
        return sigma 
    
    @property 
    def sigma(self): 
        sigma = self.act(self.sigma_) 
        # sigma /= sigma.sum(axis=1, keepdims=True) 
        return sigma 
    
    def train(self: nn.Module, epochs: int, output_dir: str, pretrained_weights: str=None): 
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4) 
        if pretrained_weights is not None: 
            try: self.load_state_dict(torch.load(pretrained_weights)) 
            except RuntimeError: print(f'The weights at {pretrained_weights} are incompatible.') 
        criterion = nn.MSELoss(reduction='sum') 
        losses = list() 
        for epoch in (pbar := tqdm(range(epochs))): 
            sigma = self() 
            optimizer.zero_grad() 
            loss = criterion(sigma, self.sigma) 
            # print(self.sigma_) 
            pbar.set_description(f'Epoch: {epoch}, loss: {loss.item()}') 
            loss.backward() 
            optimizer.step() 
            losses.append(loss.item()) 
        plt.plot(losses) 
        plt.savefig(f'{output_dir}/training.png') 
        plt.clf() 
        return sigma.detach().cpu() 

    def algorithmic_search(self, iterations: int, initial_value: np.array=None): 
        sigma = torch.ones(self.sigma_.shape, dtype=torch.float32) 
        if initial_value is not None: 
            sigma = initial_value
            if not torch.is_tensor(sigma): 
                sigma = torch.tensor(sigma) 
        for it in (pbar := tqdm(range(iterations))): 
            p = sigma.clone()  
            sigma = self.similarity(sigma=sigma) 
            norm = torch.linalg.norm(p - sigma) 
            # print(norm) 
            pbar.set_description(f'Current distance: {norm}') 
            if norm < 1e-5: 
                return sigma 
        return sigma 

def match_nodes(similarity_matrix: np.array, names: Tuple[List[str], List[str]]): 
    '''
    Match the similar nodes between the graphs according to the names at `names`. 
    '''
    indexes = np.argmax(similarity_matrix, axis=1) 
    matches = list() 
    nodes_i, nodes_ii = names 
    for inx in range(similarity_matrix.shape[0]): 
        if similarity_matrix[inx, indexes[inx]] == 0: 
            matches.append((nodes_i[inx], None)) 
            continue 
        matches.append((nodes_i[inx], nodes_ii[indexes[inx]])) 
    return matches 

def compute_similarity(
        graphs: Tuple[nx.Graph, nx.Graph], 
        pretrained_weights: str=None, 
        output_dir: str='models'): 
    '''
    Compute the similarity between the nodes within both graphs in `graphs`. 
    '''
    adjacency, shared_nodes, nodes_i, nodes_ii = interadjacency_matrix(graphs=graphs) 
    prior_sml = prior_similarity(graphs=graphs, adjacency=adjacency, shared_nodes=shared_nodes) 
    sigma = np.ones(prior_sml.shape, dtype=np.float32) 
    adj_norm = np.linalg.norm(adjacency.todense(), ord=np.inf) 
    alpha = np.divide(1, adj_norm**2 + 1) 
    solver = Solver(adjacency=adjacency, prior_sml=prior_sml, regularization=alpha, device='cpu') 
    sigma = solver.algorithmic_search(iterations=9999, initial_value=sigma)
    # sigma = solver.train(epochs=9999, pretrained_weights=pretrained_weights, output_dir=output_dir) 
    print(
        'Current similarity:', '\n', sigma, '\n', 
        'Prior similarity:', '\n', solver.prior_sml, '\n', 
        'Similarity:', '\n', solver.similarity(sigma) 
    )
    matches = match_nodes(similarity_matrix=sigma, names=(nodes_i, nodes_ii)) 
    print(matches) 
    json.dump(matches, open(f'{output_dir}/matches.json', 'w'))  
    np.save(file=f'{output_dir}/sigma.npy', arr=sigma) 
    torch.save(solver.state_dict(), f'{output_dir}/model.pt') 
    return sigma  

if __name__ == '__main__': 
    args = parse_args() 
    if not os.path.exists(args.output): 
        os.mkdir(args.output) 
    graphs = args.graphs.split(',') 
    print('Graphs:', graphs) 
    assert len(graphs) == 2, ('This is a binary evaluation; a pair of paths should be informed.') 
    graphs[:] = map(lambda graph: nx.read_graphml(graph).to_undirected(), graphs) 
    sml = compute_similarity(graphs, pretrained_weights=f'{args.output}/model.pt', output_dir=args.output) 

