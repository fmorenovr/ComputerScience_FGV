# NetworkScience

Wood Transport in Brazil

The data is available at Google Drive. Execute to download:

```
source gen_data.sh 
``` 

to enable local access to it.  

Nextly, instantiate a Conda environemnt 

```
conda create --file environemnt.yml
conda activate networks 
```

and render the algorithms to generate the visualizations through the line 

```
source pipeline.sh 
``` 

(the figures should be generated at distinguishable folders at [evaluate](./evaluate)). 

Check per year summary data for the networks at [graphs](./graphs). 

## Instructions 

Check the folder [scripts](./scripts) and the specific files for instructions on how to execute them. 

+ [`data/consolidate_data.py`](./scripts/data/consolidate_data.py) is responsible to consolidate the data from distinct species. 
+ [`data/extract_nodes.py`](./scripts/data/extract_nodes.py) chooses a subset a statistically stable subset of the data. 
+ [`data/create_network.py`](./scripts/data/create_network.py) instantiates a network with the nodes' attributes. 
+ [`data/summary_network.py`](./scripts/data/summary_network.py) generates statistics for the networks. 

+ [`figures/draw_subgraph.py`](./scripts/figures/draw_subgraph.py) contains procedures to draw a graph with the prescribed nodes' coordinates. 
+ [`figures/figures.py`](./scripts/figures/figures.py) draws multiple figures for the assessment of assortative mixing and power-law behavior for the networks. 

+ [`eda/asymmetric_players.py`](./scripts/eda/asymmetric_players.py) computes the node-level metrics degree and local clustering coefficient to capture discrepancies within the traders. 
+ [`eda/generate_random_graphs.py`](./scripts/eda/generate_random_graphs.py) generates random graphs with the Albert-Barabási's linear preferential attachment models and the (directed) configuration model.
+ [`eda/monopolies.py`](./scripts/eda/monopolies.py) implants Louvain's algorithm to identify the communities and subsequently locally central traders. 
+ [`eda/network_instance.py`](./scripts/eda/network_instance.py) designs topologically constrained networks which are used to heuristically evaluate the algorithms. 
+ [`eda/random_model.py`](./scripts/eda/random_model.py) consolidates modified procedures for the AB and configuration models to generate consistent random graphs. 
+ [`eda/similarity_dynamical.py`](./scripts/eda/similarity_dynamical.py) bestows a spectral method to estimate regular similarity between different temporal instances of the same graph. 

## How to generate the data 

The data was initially partitioned by the wood's species. However, as there is a waning quantity of available species, we should consolidate this data and parition it by year. With this objective, execute 

```
python scripts/data/consolidate_data.py --source data --output years/years.csv
``` 

which will generate the yearly data at the directory `years`.  

With the objective of enforming a network with this data, execute 

``` 
python scripts/data/create_network.py --source years --output graphs
```

we have an additional parameter `--weights`, is used to define the graph links as the number of conexions. Total volume are the links by default; the list of edges correspondent to each CSV at the folder `years` will be written to the folder `graphs`. 

Succeedingly, execute 

```
python scripts/data/summary_network.py --source graphs --output stats 
```

to generate proper (node-level) statistics for the network, including the (in, out and total) degrees' distributions and the local clustering coefficient for each node. 

Finally, we can filter relevant nodes for our analysis. We use a threshold of at least 12 transactions made by each node, just need to run:

```
python scripts/data/extract_nodes.py --source years --output samples --threshold 12  
```

## Visualizations 

Start settings the appropriate environment variables for stylings: 

```
python -c 'import matplotlib.pyplot as plt; print(plt.style.available)' # Check available styles
export MPL_STYLE='seaborn-style' 
```

Execute 

```
python scripts/figures/draw_subgraph.py --source graphs 
```

to generate visualizations of (consistently and sensibly) randomly sampled nodes from the network. 

## A dummy data set 

The networks are excessively large and computationally intricate. Thus, a subnetwork, which reasonably captures the topology of the data set, is appropriate; hence, execute 

```
python scripts/sample_netoworks.py --csv years/year_2019.csv --nodes 199 --output_csv years/sample_debug.csv
```

to instantiate a properly sized data frame at [sample_debug](./years/sample_debug.csv). 

In these settings, I would advise the use of the graph at [wood_sample_debug](./graphs/wood_sample_debug) for the design of the algorithms and their subsequent evaluation at the larger networks.  

## Random Networks 

Sensibly designed models should be evaluated with a reasonable baseline; we thus endeavor in a (directed) degree preserving generative process and in a Barabási-Albert linear preferential attachment model with a characteriscally imposed initial circumstances.
