""" 
Create a network from the wood transport data. 

python scripts/data/create_network.py   --source years \
                                        --output graphs \
                                        --weights 

Alternatively, 

python scripts/data/create_network.py   --source years \
                                        --output graphs \
                                        --fname years_2018.csv \
                                        --weights 

to instantiate a network for a specific data set. 
""" 
import os 
import pathlib 

import pandas as pd 
import networkx as nx 
import numpy as np 

import argparse 
import glob 

# Documentation 
from tqdm import tqdm 

def parse_args(): 
    '''
    Parse the command line parameters. 
    ''' 
    parser = argparse.ArgumentParser(description='Generate a directed network \
            from the timber flow data set; the condition for the edge \
            is the existence of a transaction between enterprises.') 
    parser.add_argument('--source', 
        type=str, 
        required=True, 
        help='The source of the data, as computed by `consolidate_data.py`.') 
    parser.add_argument('--output', 
            type=str, 
            required=True, 
            help='The directory to which the graphs will be written.') 
    parser.add_argument('--weights', 
            action='store_true', 
            help='Whether to instantiate a directed weighted network.' 
    ) 
    parser.add_argument('--composed_id', 
            action='store_true', 
            help='Whether to use a composed ID or not.' 
    ) 
    parser.add_argument('--fname', 
            type=str, 
            default=None, 
            help='Instead of looking up all files within `source`, check for those with this name.') 
    # Parse the command line parameters 
    args = parser.parse_args() 
    # Return the parsed command line parameters 
    return args 
    
def create_network( 
        data: pd.DataFrame, 
        from_: str, 
        to_: str,   
        weights: str=None, 
        edgelist: str=None 
    ): 
    """ 
    Instantiate a directed network whose nodes start at `from_` and are directed to `to_`. 
    """ 
    data = data[~(data[from_] == data[to_])].copy() # Avoid self-cycle
    from_nodes = data[from_].unique() 
    
    G = nx.DiGraph()

    for from_node in tqdm(from_nodes): 
        from_data = data[data[from_] == from_node].copy()
        # If the `weights` attribute equals None, then the weight within 
        # the directed graph corresponds to the quantity of instances in hich 
        # the nodes `from_node` and `to_node` are therein
        if weights is not None and weights!="Links":
            to_nodes = from_data[[to_, weights]].groupby(to_).sum(weights) 
        else: 
            weights = "Links"
            to_nodes = from_data[to_].value_counts()
            to_nodes = to_nodes.to_frame().reset_index()
            to_nodes.rename(columns = {"DestinoID": "Links", "index": "DestinoID"}, inplace=True)
            to_nodes = to_nodes.reset_index(drop=True).set_index("DestinoID")
        
        # The groupby operation updates the index; this restores the initial column
        to_nodes.reset_index(inplace=True) 
        
        for to_node in to_nodes.to_dict("records"):
            G.add_edge(from_node, to_node[to_], weight=to_node[weights]) 

    # Insert labels for the nodes 
    labels = {node: node[:node.index("-")] for node in G.nodes} 
    nx.set_node_attributes(G, values=labels, name="type") 
 
    # Write the graph to the disk 
    if edgelist is not None: 
        nx.write_graphml(G, edgelist) 
    # Return the graph 
    return G 

def instantiate_yearly_network( 
        source_dir: str, 
        output_dir: str, 
        extensions: str='csv', 
        composed_ids: bool=False, 
        weights: bool=True, 
        fname: str=None): 
    '''
    Instantiate a network with the data for each CSV in `source_dir`. 
    '''
    filenames = glob.glob(os.path.join(source_dir, f'*.{extensions}'))  
    for filename in (pbar := tqdm(filenames)): 
        if (fname is not None and os.path.basename(filename) != fname): continue 
        pbar.set_description(f'Current: {filename}') 
        fpath = pathlib.Path(filename) 
        stem = fpath.stem 
        data = pd.read_csv(filename, sep=",", low_memory=False)  
        if not composed_ids: 
            print("\nUsing simple ID")
            from_ = "OrigemID" 
            to_ = "DestinoID" 
            edgelist_fname = os.path.join(output_dir, 
                f"wood_{stem}", 
                f"wood.graphml"
            ) 
        else: 
            print("\nUsing complex ID")
            data["OrigID"] = data["OrigemID"].map(str) \
                    + "|" + data["MunOrigem"].map(str) \
                    + "|" + data["LatOrigem"].map(str) \
                    + "|" + data["LongOrigem"].map(str) \
                    + "|" + data["NomeOrigem"].map(str) 
            data["DestID"] = data["DestinoID"].map(str) \
                    + "|" + data["MunDestino"].map(str) \
                    + "|" + data["LatDestino"].map(str) \
                    + "|" + data["LongDestino"].map(str) \
                    + "|" + data["NomeDestino"].map(str) 
            from_ = "OrigID" 
            to_ = "DestID" 
            edgelist_fname = os.path.join(output_dir, 
                        f"wood_{stem}", 
                        f"wood_agg.graphml") 
        
        pathlib.Path(edgelist_fname).parent.mkdir(exist_ok=True) 

        create_network( 
                data=data, 
                from_=from_, 
                to_=to_, 
                edgelist=edgelist_fname, 
                weights="Volume" if weights else None
            )

if __name__ == "__main__": 
    args = parse_args() 
    
    source_dir = f"outputs/{args.source}/"
    output_dir = f"outputs/{args.output}/"
    
    pathlib.Path(output_dir).mkdir(parents=True, mode=0o770, exist_ok=True)

    instantiate_yearly_network(
            source_dir=source_dir, 
            output_dir=output_dir,  
            composed_ids=args.composed_id, 
            weights=args.weights, 
            fname=args.fname
    ) 
