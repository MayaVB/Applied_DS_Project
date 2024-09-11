from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def print_correlations_Spearman_and_Pearson(col1, status_code):
    # Calculate Spearman and Pearson correlations
    spearman_corr, spearman_p_value = spearmanr(col1, status_code)
    pearson_corr, pearson_p_value = pearsonr(col1, status_code)

    print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p_value}")
    print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")

# Allowing for an additional text
def print_corr(column1, column2, text):
    # Calculate Pearson correlation and p-value
    pearson_corr, pearson_p_value = pearsonr(column1, column2)
    print(f"Pearson Correlation of {text}: {round(pearson_corr, 2)} (p-value: {pearson_p_value:.4f})")

    # Calculate Spearman correlation and p-value
    spearman_corr, spearman_p_value = spearmanr(column1, column2)
    print(f"Spearman Correlation of {text}: {round(spearman_corr, 2)} (p-value: {spearman_p_value:.4f})")


