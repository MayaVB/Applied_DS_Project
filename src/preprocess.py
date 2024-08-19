import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import sys
from printstatistics import print_correlations_Spearman_and_Pearson


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocess the data: encoding categorical features, and scaling numerical features."""   
    # Create label
    y = df['status'].map({'acquired': 1, 'closed': 0})
    
    # Drop unnecessary columns
    df = df.drop(columns=['status', 'founded_at', 'name', 'id', 'state_code', 'object_id', 'labels', 'closed_at', 'Unnamed: 0', 
                'Unnamed: 6', 'zip_code', 'city', 'closed_at'])
    X = df
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['number']).columns
    
    # OneHotEncode categorical columns
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(X[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Standardize numerical columns
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(X[numerical_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)
    
    # Combine encoded categorical and scaled numerical data
    processed_df = pd.concat([encoded_categorical_df, scaled_numerical_df], axis=1)
    
    # Optionally replace NaN values
    processed_df.fillna(processed_df.mean(), inplace=True)
    
    # Alternatively we can use KNNImputer- TBR
    # knn_imputer = KNNImputer(n_neighbors=5)
    # processed_df = pd.DataFrame(knn_imputer.fit_transform(processed_df), columns=processed_df.columns)
    
    return processed_df, y


def perform_pca(processed_df, n_components=None, thereshold_PCA=0.85):
    """Performs PCA on the processed data and returns the principal components."""
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(processed_df)

    if n_components is None:
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)
        num_components = np.argmax(cumulative_explained_variance >= thereshold_PCA) + 1

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_explained_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Components')
        plt.axhline(y=thereshold_PCA, color='r', linestyle='-')
        plt.axvline(x=num_components - 1, color='r', linestyle='-')
        plt.grid()
        plt.show()
        
        print(f'Minimal number of components to explain {thereshold_PCA} variance: {num_components}')
    
    return principalComponents


def plot_pca_3d(principalComponents, status):
    """Plots the first three PCA components in 3D."""
    pca_df = pd.DataFrame(data=principalComponents, columns=['component_1', 'component_2', 'component_3'])
    pca_df['status'] = status.values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for target in pca_df['status'].unique():
        subset = pca_df[pca_df['status'] == target]
        ax.scatter(subset['component_1'], subset['component_2'], subset['component_3'], label=target, alpha=0.5)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('PCA Visualization (First 3 Components)')
    ax.legend()
    plt.show()


def perform_tsne(processed_df, status):
    """Performs t-SNE on the processed data and returns the t-SNE components."""
    tsne = TSNE(n_components=3, random_state=42)
    tsne_components = tsne.fit_transform(processed_df)

    tsne_df = pd.DataFrame(data=tsne_components, columns=['component_1', 'component_2', 'component_3'])
    tsne_df['status'] = status.values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for target in tsne_df['status'].unique():
        subset = tsne_df[tsne_df['status'] == target]
        ax.scatter(subset['component_1'], subset['component_2'], subset['component_3'], label=target, alpha=0.5)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('t-SNE Visualization (First 3 Components)')
    ax.legend()
    plt.show()

    return tsne_components

def main():
    # Load data
    df = load_data('data/startup_data.csv')

    # Preprocess data
    processed_df = preprocess_data(df)

    # Separate the 'status' column
    status = df['status'].map({'acquired': 1, 'closed': 0})

    # Perform PCA with cumulative variance threshold
    principalComponents_pca = perform_pca(processed_df)

    # Perform PCA with 3 components
    principalComponents_pca_3 = perform_pca(processed_df, n_components=3)

    # Plot the first three PCA components in 3D
    plot_pca_3d(principalComponents_pca_3, status)

    # Perform t-SNE and plot
    tsne_components = perform_tsne(processed_df, status)


if __name__ == "__main__":
    main()