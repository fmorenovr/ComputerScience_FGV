'''
Write figures for the slides. 

+   Empirical distribution of centrality measures (degree distribution and local 
    clustering coefficient) 
+   The expected and measured degrees of the traders' nearest neighbors. 
+   The fit of the power law distribution, if appropriate. 

python scripts/figures/figures.py   --source graphs/wood_sample_2018/wood.graphml \
                                    --output graphs/wood_sample_2018/figures \
                                    --figures heatmap,knn,powerlaw,cdf 
'''
import os 
import numpy as np 
import networkx as nx 
import sys 

import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
import pathlib 
import powerlaw 

sys.path.append('scripts/eda') 
from random_model import MarketEnum 
from generate_random_graphs import generate_configuration_graphs 

# Documentation
from tqdm import tqdm 
import argparse 
from typing import List, Dict, Callable  

plt.style.use(os.environ['MPL_STYLE']) 

def parse_args(): 
    '''
    Parse the command line parameters. 
    '''
    parser = argparse.ArgumentParser(description='Parse the command line parameters.') 
    parser.add_argument('--source', type=str, required=True, help='The path to the `graphml` file.') 
    parser.add_argument('--output', type=str, required=True, help='The output directory for the figures.') 
    parser.add_argument('--figures', type=str, default='heatmap,histogram', help='A comma separated list for the figures to draw.') 
    return parser.parse_args() 

def degree2histogram(degrees: Dict, bins: int=None, threshold: int=0): 
    array = np.array(degrees) 
    array = array[:, 1].astype(int) 
    array = array[array >= threshold] 
    if bins is None: 
        iqd = array.max() - array.min() + 1 
        bins_step = max(iqd / 8, 1) 
        bins = np.linspace(array.min(), array.max()+1, endpoint=True, num=int(iqd // bins_step)+1) 
    values, edges = np.histogram(array, bins=bins) 
    return array, values, edges

def degree2array(degrees: Dict): 
    array = np.array(degrees) 
    values = array[:, 1].astype(int) 
    return values 

def joint_distribution(x: Dict, y: Dict, threshold: float): 
    '''
    Compute the joint distribution between the (discrete) random quantities `x` and `y`.  
    '''
    x = degree2array(x) 
    y = degree2array(y) 
    # Chose nodes with appropriate degree 
    inxs = (x >= threshold) & (y >= threshold) 
    x = x[inxs] 
    y = y[inxs] 
    xbins = np.linspace(min(x), max(x)+1, endpoint=True, num=9) 
    ybins = np.linspace(min(y), max(y)+1, endpoint=True, num=9) 
    hist, xedges, yedges = np.histogram2d(x=x, y=y, bins=(xbins, ybins))  
    return hist, xedges, yedges 

def degree_heatmap(graph: nx.DiGraph, fname: str, random_ensemble: nx.DiGraph): 
    '''
    Write a heatmap with the joint (ingoing and outgoing) degree distributions. 
    The image is buffered at `fname`. 
    '''
    # graph.remove_nodes_from([n[0] for n in graph.nodes.data('type') if n[1] != MarketEnum.TRADER]) 
    traders = [node for (node, type) in graph.nodes.data('type') if type == MarketEnum.TRADER] 
    indegrees, indist, inedges = degree2histogram(graph.in_degree(traders), threshold=0) 
    outdegrees, outdist, outedges = degree2histogram(graph.out_degree(traders), threshold=1) 
    # Compute their joint distribution 
    # print(indegrees, outdegrees) 
    hist, xsupport, ysupport = joint_distribution( 
            x=graph.in_degree(traders), 
            y=graph.out_degree(traders), 
            threshold=1 
    ) 
    hist = np.flip(hist, 1) 
    # print(ysupport, xsupport, hist) 
    plt.imshow(hist, cmap='YlGn') 
    plt.xlabel('in-degree') 
    plt.ylabel('out-degree') 
    plt.title('The joint degree distribution for the traders'"'"' induced subgraph') 
    ax = plt.gca() 
    xsupport = [f'{val:.2f}' for val in xsupport if val != str()] 
    ysupport = [f'{val:.2f}' for val in ysupport if val != str()] 
    ax.set_xticks(np.arange(len(xsupport))-.5, labels=xsupport) 
    ax.set_yticks(np.arange(len(ysupport))-.5, labels=ysupport[::-1]) 
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    plt.savefig(fname) 
    plt.show() 

def plot_degree_cdf(graph: nx.DiGraph, nodes: List[str], prefix: str, fname: str): 
    '''
    Write a line plot for the subgraph induced by `nodes`. 
    '''
    degrees = [d for (n, d) in graph.degree(nodes)] 
    degrees, counts = np.unique(degrees, return_counts=True) 
    cdf = [counts[i:].sum() / counts.sum() for i in np.arange(counts.shape[0], like=counts)] 
    plt.plot(degrees, cdf, c='g') 
    plt.xlabel('Degree') 
    plt.ylabel('Complementary CDF') 
    plt.title(f'The CDF for the {prefix}'"'"'s degree distribution') 
    fname, extension = os.path.splitext(fname) 
    plt.savefig(f'{fname}_{prefix}{extension}') 
    plt.show() 
    # plt.clf() 

def degree_cdf(graph: nx.DiGraph, fname: str, random_ensemble: List[nx.DiGraph]): 
    '''
    Write a line plot for the (complementary) degree distribution of the graph graph. 
    It contemplates the suppliers and the consumers. 
    '''
    suppliers = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.SUPPLIER]
    consumers = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.CONSUMER] 
    plot_degree_cdf(graph, suppliers, prefix='suppliers', fname=fname) 
    plot_degree_cdf(graph, consumers, prefix='consumers', fname=fname) 

def update_dictionary(current: Dict, modifier: Dict, func: Callable): 
    for key in modifier: 
        if not key in current: 
            current[key] = func(int(1e-19), modifier[key]) 
        else: 
            current[key] = func(current[key], modifier[key]) 
    return current 

def ensemble_knn(graphs: List[nx.DiGraph]): 
    '''
    Compute the average degree of a node's nearest neighbors in an ensemble of graphs. 
    '''
    # Convert the directed graphs to undirected graphs 
    knns = dict() 
    for graph in graphs: 
        nndegrees = nx.k_nearest_neighbors(graph) 
        update_dictionary(current=knns, modifier=nndegrees, func=np.add)  
    knns = list(knns.items()) 
    knns = np.array(knns).T 
    np.sort(knns, axis=1) 
    degrees, nndegrees = knns 
    nndegrees /= len(graphs) 
    return degrees, nndegrees 

def compute_expected_knn(graph: nx.DiGraph): 
    '''
    Compute the expected KNN score for a random graph, as generated by the Erod-Renyi's model. 
    '''
    degrees = np.array([d for (n, d) in graph.degree]) 
    avgknn = (degrees ** 2).mean() / degrees.mean() 
    return avgknn 

def knn(graph: nx.DiGraph, fname: str, random_ensemble: List[nx.DiGraph]): 
    '''
    Write a scatter plot with the distribution of the nearest neighbors' 
    average degree as a function of the degree. 

    The PNG is written to `fname`. 
    '''
    nodes = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.TRADER] 
    expected_knn = compute_expected_knn(graph) 
    traders = graph.subgraph(nodes).to_undirected()  
    kconnectivity = list(nx.k_nearest_neighbors(traders).items()) 
    degrees, nndegrees = np.array(kconnectivity).T 
    indexes = np.argsort(degrees) 
    nndegrees = nndegrees[indexes]
    degrees = degrees[indexes] 
    baseline_dgs, baseline_nn = ensemble_knn(graphs=random_ensemble) 
    plt.scatter(degrees, nndegrees, c='g', label='Real') 
    plt.scatter(baseline_dgs, baseline_nn, c='violet', label='Random') 
    plt.axhline(expected_knn, linestyle='--', label='Expected KNN (ER)') 
    plt.legend() 
    plt.xlabel('Degree') 
    plt.ylabel('KNN') 
    plt.title('The average degree of a node'"'"' nearest neighbors as a function of the degree') 
    plt.savefig(fname) 
    plt.show() 
    # plt.clf() 

def logarithm_binning(degrees: List[int]): 
    '''
    Compute the empirical distribution of the degrees with a logarithmically binned histogram. 

    Returns both the empirical distribution's support and its values for each datum. 
    '''
    nbins = np.ceil(np.log2(max(degrees))).astype(int)   
    bins = np.logspace(0, nbins, nbins+1, base=2) 
    counts, support = np.histogram(degrees, bins=bins) 
    empirical_dist = counts / counts.sum() 
    support = (support[:-1] + support[1:] - 1) / 2 
    return support, empirical_dist 

def linear_binning(degrees: List[int]): 
    '''
    Compute the (evenly spaced) linear binning for the degrees `degrees`. 

    Return a list with the counts of each degree with a correspondent node in the data set. 
    '''
    counts = np.bincount(degrees) 
    support = np.argwhere(counts != 0) 
    empirical_dist = counts[support].astype(float) 
    empirical_dist /= empirical_dist.sum() 
    return support, empirical_dist 

def plot_degree_distribution(degrees: List[int]): 
    '''
    Plot the degree distribution of the degrees `degrees`. 

    Use both logarithmic and linear binning. 
    '''
    linear_support, linear_dist = linear_binning(degrees) 
    log_support, log_dist = logarithm_binning(degrees) 
    plt.scatter(linear_support, linear_dist, c='gray', alpha=.3, label='linear') 
    plt.scatter(log_support, log_dist, c='g', label='log') 
    ax = plt.gca() 
    ax.set_xscale('log', base=10) 
    ax.set_yscale('log', base=10) 
    plt.xlabel('Degree') 
    plt.ylabel('Density') 
    # plt.clf() 
    return linear_dist, log_dist 

def powerlaw_fit(graph: nx.DiGraph, fname: str, random_ensemble: List[nx.DiGraph]): 
    '''
    Generate a powerlaw fit for the graph `graph`'s degree distribution. 

    The scatter plot PNG is buffered at `fname`. 
    '''
    # Assume that the graph is undirected; this is heuristically plausible, 
    # as we are measuring the interactions within the traders 
    graph = graph.to_undirected() 
    nodes = [n[0] for n in graph.nodes.data('type') if n[1] == MarketEnum.TRADER] 
    degrees = [d for (n, d) in graph.degree(nodes) if d >= 1] 
    support, empirical_dist = logarithm_binning(degrees) 
    fit = powerlaw.Fit(degrees, discrete=True) 
    fit.power_law.plot_pdf(color='g', linestyle='--', label=f'$\\alpha$: {fit.alpha:.2f}, $x_{{min}}$: {fit.xmin:.2f}') 
    linear_dist, log_dist = plot_degree_distribution(degrees) 
    plt.ylim(min(linear_dist)*.9, max(linear_dist)*1.1) 
    plt.legend() 
    plt.title('A power law fit to the PDF underlying the degrees'"'"' distribution.') 
    plt.savefig(fname) 
    # plt.clf() 
    plt.show() 

def make_dir(fname: str, exist_ok=True): 
    pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=exist_ok)  

METHODS = { 
    'heatmap': degree_heatmap, 
    'cdf': degree_cdf, 
    'knn': knn, 
    'powerlaw': powerlaw_fit
} 

if __name__ == '__main__': 
    args = parse_args() 
    graph = nx.read_graphml(args.source) 
    print('Generate random graphs') 
    random_ensemble = list(generate_configuration_graphs(graph, quantity=19)) 
    figures = METHODS.keys() if args.figures == 'all' else args.figures.split(',') 
    for method in figures: 
        fname = os.path.join(args.output, method + '.png') 
        make_dir(fname, exist_ok=True) 
        METHODS[method](graph=graph.copy(), fname=fname, random_ensemble=random_ensemble) 


