import pandas as pd

from typing import List, Dict
 
def remove_empty_records(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Removes rows from the DataFrame where the specified feature (column) has empty or NaN values.

    :param `data`: Input DataFrame from which empty records will be removed.
    :param `feature`: The column name to check for empty or NaN values.
    :return: `DataFrame` with rows containing empty or NaN values in the specified feature removed.
    :raises `ValueError`: If the specified feature is not in the DataFrame.
    """
    # Check if the feature is in the DataFrame columns
    if feature not in data.columns:
        raise ValueError(f"The feature `{feature}` is not in the DataFrame")
    
    try:
        # Identify rows with empty or NaN values in the specified feature
        condition = data[feature].isna() | (data[feature] == '')
        empty_rows = data[condition]


        # Remove rows where the specified feature is empty or NaN and reset the index
        data = data.drop(empty_rows.index).reset_index(drop=True)
        
        # Print the number of removed elements
        print(f"Number of elements that were removed: {empty_rows.shape[0]}")
        
        return data
    except Exception as e:
        raise ValueError(f"Empty records removal error: {e}")

def get_data_criteria(data: pd.DataFrame, features: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows where the specified columns match the provided values.

    :param `data`: Input DataFrame to be filtered.
    :param `features`: Dictionary where keys are column names and values are lists of acceptable values for those columns.
    :return: Filtered `DataFrame` containing only rows that match the specified criteria.
    :raises `ValueError`: If the specified feature is not in the DataFrame.
    """
    # Check if all keys in features are in the DataFrame columns
    if not all(key in data.columns for key in features.keys()):
        missing_keys = [key for key in features.keys() if key not in data.columns]
        raise ValueError(f"The following features are not in the DataFrame: {missing_keys}")

    try:
        # Filter the DataFrame based on the criteria in the features dictionary
        filtered_data = data[data[list(features.keys())].apply(lambda x: all(x[key] in features[key] for key in features), axis=1)]
        
        # Drop the columns specified in the features dictionary from the filtered DataFrame
        filtered_data = filtered_data.drop(labels=list(features.keys()), axis=1)
        
        return filtered_data
    except Exception as e:
        raise ValueError(f"Data get error: {e}")
