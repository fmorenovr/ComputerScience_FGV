'''
The network is large; we should generate a subgraph to assess the validty of our procedures. 
'''
import os 
import networkx as nx 
import pandas as pd 
import numpy as np 
import re 

# Documentation 
import argparse 
from tqdm import tqdm 
from typing import List 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Generate a subsample of the data set.') 
    parser.add_argument('--csv', type=str, required=True, help="The path to the network's underlying relational data base.") 
    parser.add_argument('--nodes', type=int, default=int(6.5e2), help='The quantity of nodes within the subgraph.') 
    parser.add_argument('--output_csv', type=str, required=True, help='The output file to which the subgraph will be written.') 
    # Parse the command line parameters 
    args = parser.parse_args() 

    # Return the parsed command line parameters 
    return args 

def generate_subgraph(csv: str, nodes: int): 
    '''
    Generate a subgraph from the relational database `csv` with `nodes` nodes. 
    '''
    data = pd.read_csv(csv) 
    # Sample initial nodes of type 'MANEJO'
    initial_nodes = int(nodes * .1) 
    
    is_supplier = data['OrigemID'].str.contains('MANEJO')
    suppliers = data.loc[is_supplier, 'OrigemID'].unique() 

    # Choose randomly a set of suppliers 
    suppliers_inxs = np.random.choice(suppliers.shape[0], size=initial_nodes, replace=False) 
    suppliers = suppliers[suppliers_inxs] 
    
    current_nodes = suppliers.copy()    # The current nodes sampled 
    pbar = tqdm(total=nodes)            # A progress bar 
    batch_nodes = int(nodes * .3)       # Quantity of nodes (neighbors) to sample at each step  
    while len(current_nodes) < nodes: 
        regex = '|'.join([f'(?i){re.escape(identifier)}' for identifier in current_nodes]) 
        is_in_network = data['OrigemID'].str.contains(regex)  
        current_traders = data.loc[is_in_network, 'DestinoID'].values   
        # Choose randomly the traders 
        traders_inxs = np.random.choice(current_traders.shape[0], size=batch_nodes, replace=False) 
        traders_ids = current_traders[traders_inxs] 
        traders_ids = np.unique(traders_ids) 
        current_nodes = np.append(current_nodes, traders_ids) 
        current_nodes = np.unique(current_nodes) 
        pbar.update(len(current_nodes) - pbar.n) 
    # Return the nodes sampled from the data set 
    return data, current_nodes 

def write_csv(nodes: List[str], data: pd.DataFrame, output_csv: str): 
    '''
    Write the subgraph within the data `data` whose nodes are contained in `nodes`. 
    '''
    regex = '|'.join([f'(?i){re.escape(identifier)}' for identifier in nodes]) 
    samples_inxs = data['OrigemID'].str.contains(regex) & data['DestinoID'].str.contains(regex) 
    samples = data[samples_inxs] 
    samples.to_csv(output_csv, index=None)

if __name__ == '__main__': 
    args = parse_args() 
    data, current_nodes = generate_subgraph(csv=args.csv, nodes=args.nodes) 
    write_csv(nodes=current_nodes, data=data, output_csv=args.output_csv) 
    
