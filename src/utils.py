import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def set_seed(random_seed=42):
    np.random.seed(random_seed)  # Set seed for NumPy operations to ensure reproducibility
    #random.seed(random_seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1' # Ensure deterministic operations

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def print_corr(column1, column2, text):
    # Calculate Pearson correlation and p-value
    pearson_corr, pearson_p_value = pearsonr(column1, column2)
    print(f"Pearson Correlation of {text}: {round(pearson_corr, 2)} (p-value: {pearson_p_value:.4f})")

    # Calculate Spearman correlation and p-value
    spearman_corr, spearman_p_value = spearmanr(column1, column2)
    print(f"Spearman Correlation of {text}: {round(spearman_corr, 2)} (p-value: {spearman_p_value:.4f})")
