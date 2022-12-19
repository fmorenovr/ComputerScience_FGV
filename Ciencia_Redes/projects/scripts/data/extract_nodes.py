'''
Extract nodes which are subject to substantially large statistical fluctuation in a network. 

python scripts/data/extract_nodes.py    --source data \
                                        --output samples \
                                        --threshold 12  
'''
import os 
import numpy as np 
import networkx as nx 
import pandas as pd 
import glob
import pathlib 
import matplotlib.pyplot as plt 

from multiprocessing import Pool 
from functools import partial 

# Documentation 
import argparse 
from typing import List, Tuple, Callable 
from tqdm import tqdm 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Extract highly timid nodes.') 
    parser.add_argument('--source', type=str, required=True, help='The `csv` file.') 
    parser.add_argument('--output', type=str, required=True, help='The output directory.') 
    parser.add_argument('--threshold', type=int, default=12, help='The quantity of transactions a node should do to not be extracted.') 
    parser.add_argument('--fname', 
            type=str, 
            default=None, 
            help='Instead of looking up all files within `source`, check for those with this name.') 
    return parser.parse_args() 

def estimate_transactions(data: pd.DataFrame): 
    '''
    Estimate the quantity of transactions for each player in the market.
    '''
    agents = np.stack([data['OrigemID'].values, data['DestinoID'].values], axis=-1) 
    agents, counts = np.unique(agents, return_counts=True) 
    # Return the quantity of transactions for each agent 
    return agents, counts

def choose_rows(
        row: Tuple[int, pd.Series], 
        condition: Callable[[pd.Series], bool], 
        args: Tuple): 
    '''
    Check if row `row` should be preserved in the data frame. 
    '''
    if condition(row[1], *args): 
        return row[0] 

def is_adequate(row: pd.Series, agents: List[str]): 
    return (row['OrigemID'] in agents) and (row['DestinoID'] in agents) 

def extract_instances(source: str, threshold: int, output_dir: str): 
    '''
    Extract the instances within `source` whose quantity of transactions is at most `threshold`. 
    '''
    data = pd.read_csv(source, low_memory=False) 
    stemfname = pathlib.Path(source).stem 
    agents, counts = estimate_transactions(data) 
    
    new_counts = counts[counts > threshold]
    new_agents = agents[counts > threshold]
    print(new_counts)
    plt.figure(figsize=(20,10))
    print(f'Transactions estimated: [{min(counts)}, {max(counts)}].') 
    print(f'Initial nodes: {len(agents)}.') 
    plt.title(f'Transactions estimated: [{min(new_counts)}, {max(new_counts)}]\nInitial nodes: {len(agents)}, Current nodes: {len(new_agents)}.')
    plt.hist(new_counts, bins=20) 
    plt.xlabel('Number of Transactions', fontsize=16)
    plt.ylabel('Number of Nodes', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{stemfname}_transactions.png')) 
    plt.clf()
    plt.cla()
    plt.close()
    print('Histogram available.') 
    
    empirical_dist = np.bincount(counts) 
    np.save(file=os.path.join(output_dir, f'{stemfname}_empirical_dist.npy'), 
            arr=empirical_dist) 
    
    print(f'Compressed. Current volume of nodes: {len(new_agents)}.') 
    print('Agents identified.') 
    with Pool(os.cpu_count()) as p: 
        is_allowed = partial(choose_rows, 
                condition=is_adequate, 
                args=(new_agents,)) 
        indexes = p.map(is_allowed, data.iterrows()) 
    indexes = [index for index in indexes if index is not None] 
    data = data.loc[indexes, :] 
    data.to_csv(os.path.join(output_dir, f'{stemfname}.csv'), sep=",", index=None) 
    return data

if __name__ == '__main__': 
    args = parse_args() 

    source_dir = f"outputs/{args.source}/"
    output_dir = f"outputs/{args.output}/"
    
    pathlib.Path(output_dir).mkdir(parents=True, mode=0o770, exist_ok=True)

    filenames = glob.glob(os.path.join(source_dir, f'*.csv'))  

    for filename in (pbar := tqdm(filenames)): 

        if (args.fname is not None and os.path.basename(filename) != args.fname): continue 

        print(f"\nFiltering {filename}")

        extract_instances(source=filename, threshold=args.threshold, output_dir=output_dir) 

