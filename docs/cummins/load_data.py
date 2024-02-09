"""
Various classes and methods that help represent Cummins data.

Author: Ilias Bilionis

"""


__all__ = ["load_dataset", "load_collection", "collection_to_dataframe"]


from typing import Dict         # Importing 'Dict' type for type hinting
import glob                     # File pattern matching
import os                       # Operating system functions
import tqdm                     # Progress bar visualization
import pandas as pd             # Data manipulation and analysis
from scipy.io import loadmat   # Load MATLAB files


# Function to load a Cummins *.mat file into a pandas DataFrame
def load_dataset(
        m_file : str, 
        resample_rule : Dict = None,
        resample_aggregation = "mean",
        **resample_kwargs
) -> pd.DataFrame:
    # Load the .mat file using scipy's loadmat function
    mat = loadmat(m_file)
    
    # Get the variable names from the .mat file (excluding internal variables)
    var_names = [k for k in mat.keys() if not k[:2] == "__" ]
    var_names.remove("Time")
    
    # Create an empty DataFrame with columns as the variable names and Time as the index
    df = pd.DataFrame(
        columns=var_names,
        index=pd.TimedeltaIndex(data=mat["Time"].flatten(), unit="s", name="Time")
    )
    
    # Populate the DataFrame with data from the .mat file
    for v in var_names:
        df[v] = mat[v].flatten()
    
    # If resampling is requested, apply the resampling to the DataFrame
    if resample_rule is not None:
        df = getattr(df.resample(resample_rule, **resample_kwargs), resample_aggregation)()
    
    # Return the resulting DataFrame
    return df


# Function to load all *.mat files from a folder into a dictionary of pd.DataFrames
def load_collection(folder : str, **kwargs) -> Dict[str, pd.DataFrame]:
    # Get a list of .mat files in the specified folder
    m_files = glob.glob(os.path.join(folder, "*.mat"))
    
    # Create an empty dictionary to store the loaded DataFrames
    out = {}
    
    # Iterate over each .mat file
    for f in tqdm.tqdm(m_files):
        # Load the file using load_dataset and store the DataFrame in the dictionary
        out[os.path.basename(f).split(".")[0]] = load_dataset(f, **kwargs)
    
    # Return the dictionary of DataFrames
    return out


# Function to combine all DataFrames in a collection into a single DataFrame
def collection_to_dataframe(collection : Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Add a "Dataset" column to each DataFrame indicating its key in the collection
    for k in collection.keys():
        collection[k]["Dataset"] = k
    
    # Concatenate all DataFrames in the collection into a single DataFrame
    df = pd.concat(collection.values())
    
    # Reset the index of the resulting DataFrame
    df = df.reset_index()
    
    # Return the combined DataFrame
    return df