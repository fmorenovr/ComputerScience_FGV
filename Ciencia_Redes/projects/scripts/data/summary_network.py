""" 
Summarize the network's attributes. 

python scripts/data/summary_network.py  --source graphs \
                                        --output stats 

Alternatively, 

python scripts/data/summary_network.py  --source graphs \
                                        --output stats \
                                        --fname wood_years_2018 

to generate statistics for a specific data set. 
""" 
import os 
import glob 

import numpy as np 
import pandas as pd 
import networkx as nx 
import json 
import pathlib 

import matplotlib.pyplot as plt 

from tqdm import tqdm 
import argparse 

from multiprocessing import Process

def parse_args(): 
    ''' 
    Parse the command line parameters. 
    ''' 
    parser = argparse.ArgumentParser(description='Compute summary statistics for the network.') 
    parser.add_argument('--source', type=str, required=True, help='The directory with the `graphml` files.') 
    parser.add_argument('--output', type=str, required=True, help='The directory to which the summary statistics will be written.') 
    parser.add_argument('--fname', type=str, default=None, help='A specific `graphml` file within `source` from which we will generate the summary statistics.') 
    # Parse the command line parameters 
    args = parser.parse_args() 

    # Return the command line parameters 
    return args 

def quantile( 
        dist: np.ndarray, 
        th: float 
    ): 
    """ 
    Compute the index correspondent to the quantile `th` within the 
    distribution `dist`, in which `dist[i]` equals the probability of a 
    random variable, subjected to this distribution, be equal to `i`. 
    """
    if isinstance(dist, list): 
        dist = np.array(dist) 

    # Compute the cummulative distribution 
    cum_dist = np.cumsum(dist) 
  
    assert np.isclose(cum_dist[-1], 1), \
            "the quantity `dist` ins't a distribution; {sum}".format(sum=cum_dist[-1])  
    
    # Smooth the distribution's support 
    dist = dist[cum_dist < th]  
    dist = dist / dist.sum() 

    # Return the cropped distribution 
    return dist 

def truncated_histogram( 
        data: np.ndarray, 
        th: float, 
        title: str,
        fname: str 
    ): 
    """ 
    Generate a truncated histogram for the data `data`, in which rare values are 
    extirpated, according to the threshold `th` on the cumulative distribution function. 
    """ 
    data_truncated = quantile(data, th=th) 
    
    plt.figure(figsize=(20,10))
    plt.bar( 
            np.arange(len(data_truncated)), 
            data_truncated 
    ) 
    plt.title(title)
    plt.xlabel("Degree", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.grid()
    plt.savefig(fname) 
    # Update the current figure 
    plt.clf()
    plt.cla()
    plt.close()

def summary( 
        input_graphfname: str, 
        output_json: str, 
        output_png: str, 
        source_dir: str, 
        output_dir: str 
    ): 
    """ 
    Capture the list of edges (weighted) from `input_edgelist`; then, 
    generate summary statistics, namely 

    + the number of nodes and of edges, 
    + the edges' density, 
    + the average degree, 
    + the degree standard deviation, and 
    + the degree distribution, 

    and write them to `output_json`. Also, generate visualizations for the 
    network and its degree distribution (a histogram). 
    """ 
    G = nx.read_graphml(os.path.join(source_dir, input_graphfname)) 
    assert type(G) == type(nx.DiGraph()), "The graph's type is inappropriate"  
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True) 

    # Consolidate the data in a dictionary 
    data = dict() 

    data["number_of_nodes"] = G.number_of_nodes() 
    data["number_of_edges"] = G.number_of_edges() 
    max_edges = (G.number_of_nodes()) * (G.number_of_nodes() - 1) / 2
    data["edge_density"] = G.number_of_edges() / max_edges 
    degrees = np.array([node[1] for node in G.degree])  
    data["avg_degree"] = degrees.mean() 
    data["std_degree"] = degrees.std() 
    degree_distribution = np.bincount(degrees) 
    data["degree_distribution"] = (degree_distribution / degree_distribution.sum()).tolist() 
    
    # Compute statistics for in and out degrees 
    in_degrees = np.array([node[1] for node in G.in_degree]) 
    out_degrees = np.array([node[1] for node in G.out_degree]) 
    
    data["in_avg_degree"] = in_degrees.mean() 
    data["out_avg_degree"] = out_degrees.mean() 

    data["in_std_degree"] = in_degrees.std() 
    data["out_std_degree"] = out_degrees.std() 
    in_degree_distribution = np.bincount(in_degrees) 
    out_degree_distribution = np.bincount(out_degrees) 
    data["in_degree_distribution"] = (in_degree_distribution / in_degree_distribution.sum()).tolist() 
    data["out_degree_distribution"] = (out_degree_distribution / out_degree_distribution.sum()).tolist() 

    json.dump(data, 
            open(os.path.join(output_dir, output_json), "w"))  
    # Generate a histogram with the degrees' probabilities 

    # Extract rare degrees 
    truncated_histogram(data["degree_distribution"], th=1 - 1e-2, title="Degree Distribution",
            fname=os.path.join(output_dir, output_png))
    truncated_histogram(data["in_degree_distribution"], th=1 - 1e-2, title="In Degree Distribution",
            fname=os.path.join(output_dir, "in_" + output_png)) 
    truncated_histogram(data["out_degree_distribution"], th=1 - 1e-2, title="Out Degree Distribution",
            fname=os.path.join(output_dir, "out_" + output_png))     
   
    # Write the degrees to a numpy array in the disk 
    np.save( 
            arr=degrees, 
            file=os.path.join(output_dir, "degrees.npy") 
    ) 
    np.save( 
            arr=in_degrees, 
            file=os.path.join(output_dir, "in_degrees.npy") 
    ) 
    np.save( 
            arr=out_degrees, 
            file=os.path.join(output_dir, "out_degrees.npy") 
    ) 
    
    clustering_coefficients = nx.clustering(G) 
    
    json.dump( 
            clustering_coefficients, 
            open(os.path.join(output_dir, 'clustering_coefficients.json'), 'w') 
    ) 
    # p = Process(target=nx.write_gexf, args=(G, os.path.join(data_dir, output_network))) 
    
    # Return the process to draw the network 
    # return p    

if __name__ == "__main__": 
    args = parse_args() 
    source_dir = f"outputs/{args.source}/"
    output_dir = f"outputs/{args.output}/"
    folders = os.listdir(source_dir) 
    for folder in tqdm(folders): 
        if (args.fname is not None and folder != args.fname): continue 
        graph_dir = os.path.join(output_dir, folder) 
        summary( 
            input_graphfname="wood.graphml", 
            output_json="wood_st.json", 
            output_png="wood_dgs.png",
            source_dir=os.path.join(source_dir, folder), 
            output_dir=graph_dir 
        ) 

