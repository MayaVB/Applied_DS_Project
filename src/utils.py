import pandas as pd
import numpy as np

def set_seed(random_seed=42):
    np.random.seed(random_seed)  # Set seed for NumPy operations to ensure reproducibility
    #random.seed(random_seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1' # Ensure deterministic operations

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)
