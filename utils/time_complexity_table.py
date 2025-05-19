import pandas as pd
import time


def create_time_complexity_table(data):
    """
    Creates a time complexity table as a pandas DataFrame.

    Parameters:
    data (list of dict): Each dict should have keys:
        - 'input_size_max_iter'
        - 'input_size_local_search'
        - 'time_max_iter'
        - 'time_local_search'
        - 'time_total'

    Returns:
    pd.DataFrame: DataFrame with columns ["input_size_max_iter", "input_size_local_search", "time_max_iter (s)", "time_local_search (s)", "time_total (s)"]
    """
    df = pd.DataFrame(data)
    df = df.rename(columns={
        'time_max_iter': 'time_max_iter (s)',
        'time_local_search': 'time_local_search (s)',
        'time_total': 'time_total (s)'
    })
    df = df[["input_size_max_iter", "input_size_local_search", "time_max_iter (s)", "time_local_search (s)", "time_total (s)"]]
    return df


def get_time():
    t_iter = time.time()
    t_local_search = time.time()

    return t_iter, t_local_search
