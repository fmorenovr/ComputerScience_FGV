'''
Implant the available assessments to a specific data set. 
'''
import os 
import numpy as np 
import networkx as nx 

import subprocess 
import sys
import pathlib 

# Documentation 
import argparse 
from typing import List, Dict  
from tqdm import tqdm 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Implant the available procedures in a data set.') 
    parser.add_argument('--source', type=str, required=True, help='The path to the `graphml`s files.'), 
    parser.add_argument('--output', type=str, required=True, help='The path to which the figures will be written.') 
    return parser.parse_args() 

def forward(sources: List[str], output: str): 
    '''
    Assess the pipeline with the graph at `source`. 
    '''
    scripts = 'scripts'
    for source in sources:   
        fname = pathlib.Path(source).parent
        cout = os.path.join(output, fname.stem) 
        pathlib.Path(cout).mkdir(exist_ok=True, parents=True) 
        print('Generating figures') 
        cmd = f'python {scripts}/figures/figures.py --source {source} --output {cout}/figures --figures all' 
        subprocess.call(cmd, shell=True) 
        print('Identifying asymmetric players') 
        cmd = f'python {scripts}/eda/asymmetric_players.py --source {source} --output_dir {cout}/asymmetries' 
        subprocess.call(cmd, shell=True) 
        print('Completing monopolies') 
        cmd = f'python {scripts}/eda/monopolies.py --source {source} --output {cout}/monopolies' 
        subprocess.call(cmd, shell=True) 
    
    for inx in range(1, len(sources)): 
        fname_i = pathlib.Path(sources[inx-1]).parent 
        fname_ii = pathlib.Path(sources[inx]).parent 
        cout = os.path.join(output, f'similarities_{fname_i.stem}_{fname_ii.stem}') 
        print(cout)  
        pathlib.Path(cout).mkdir(exist_ok=True, parents=True) 
        print('Computing the similarity between a pair of graphs') 
        cmd = f'python {scripts}/eda/similarity_dynamical.py --graphs {sources[inx-1]},{sources[inx]} --output {cout}' 
        subprocess.call(cmd, shell=True) 

if __name__ == '__main__': 
    args = parse_args() 
    forward(sources=args.source.split(','), output=args.output) 
