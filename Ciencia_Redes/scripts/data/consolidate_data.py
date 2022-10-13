""" 
Consolidate the different data frames from distinct species. 

python scripts/data/consolidate_data.py --source data \
                                        --extensions .csv \
                                        --output years/years.csv 
""" 
import os 
import numpy as np 
import pandas as pd 
from glob import glob 
import pathlib 

# Documentation 
from typing import List 
from tqdm import tqdm 
import argparse 

def parse_args(): 
    '''
    Parse the command line parameters. 
    ''' 
    parser = argparse.ArgumentParser(description='Consolidate the data from different species and split them by year.') 
    parser.add_argument('--source', type=str, required=True, help='The path to the data sets.') 
    parser.add_argument('--extensions', type=str, default='.csv', help='The extension of the files.') 
    parser.add_argument('--output', type=str, required=True, help='The output filename in which the consolidated data will be written.') 

    # Parse the command line parameters 
    args = parser.parse_args() 
    
    # Return the parsed command line parameters 
    return args 

def read_csv(
        filename: str, 
        sep: str=",",
        low_memory: bool=False,
        fname_as_column: str=None 
    ): 
    """ 
    Capture a CSV and use the filename as the column. 
    """ 
    data = pd.read_csv(filename, sep=sep, low_memory=low_memory, index_col=False) 

    # Insert another column 
    if fname_as_column is not None: 
        data[fname_as_column] = filename 
    
    # Return the data frame 
    return data 


def consolidate_data(filenames: List[str], to_csv: str=None): 
    """ 
    Join the data frames, combining the values at each column.  
    """ 
    for i, filename in tqdm(enumerate(filenames)): 
        if i < 1: 
            data = read_csv(filename, sep=",", low_memory=False, fname_as_column="Species") 
        else: 
            other = read_csv(filename, sep=",", low_memory=False, fname_as_column="Species") 
            data = pd.concat([data, other], ignore_index=True)

    data.drop(columns=["index", 'Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
    if to_csv is not None: 
        data.to_csv(to_csv, index=None, sep=",") 

    # Return the consolidated data 
    return data 

def split( 
        filename: str, 
        attr: str 
    ): 
    """ 
    Split the dataframe at `filename` according to the attribute `attr`. 
    """ 
    data = pd.read_csv(filename, sep=",", low_memory=False, index_col=False) 
    attr_values = data[attr].unique() 
    
    # Capture the filename's stem and the parent directory 
    path = pathlib.Path(filename) 
    parent = str(path.parent) 
    stem = path.stem 

    # Identify the instances for each attribute's value 
    for val in attr_values: 
        data_attr = data[data[attr] == val] 
        # Write the data to csv 
        data_attr.to_csv( 
                os.path.join(parent, stem + "_{val}".format(val=val) + ".csv"), 
                index=None 
        )

if __name__ == "__main__": 
    args = parse_args() 
    data_dir = args.source   
    output = f"outputs/{args.output}" 
    # Make the directories 
    pathlib.Path(output).parent.mkdir(exist_ok=True, parents=True) 
    filenames = glob(os.path.join(data_dir, f"*{args.extensions}")) 
    consolidate_data(filenames, to_csv=output) 
    split(filename=output, attr="Ano") 



