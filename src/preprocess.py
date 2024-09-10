import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import sys
from printstatistics import print_correlations_Spearman_and_Pearson
from utils import load_data
import pandas as pd
from math import radians, cos, sin, asin, sqrt



def process_funding_dates(df):
    # Convert the date columns to datetime
    df['first_funding_at'] = pd.to_datetime(df['first_funding_at'])
    df['last_funding_at'] = pd.to_datetime(df['last_funding_at'])
    
    # Extract year, month, and day for first_funding_at
    df['first_funding_at_year'] = df['first_funding_at'].dt.year
    df['first_funding_at_month'] = df['first_funding_at'].dt.month
    df['first_funding_at_day'] = df['first_funding_at'].dt.day
    
    # Extract year, month, and day for last_funding_at
    df['last_funding_at_year'] = df['last_funding_at'].dt.year
    df['last_funding_at_month'] = df['last_funding_at'].dt.month
    df['last_funding_at_day'] = df['last_funding_at'].dt.day
    
    # Drop the original date columns and 'state_code.1'
    df = df.drop(columns=['first_funding_at', 'last_funding_at', 'state_code.1'])
    
    return df


def feature_log_scaler(df, column):
    df[f'{column}_log'] = np.log(df[column])
    df = df.drop(columns=column)
    return df


def cap_feature(df, column, quantile=0.95):
    """
    Caps the outliers in the specified column of a DataFrame based on the given quantile threshold.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to cap (default is 'relationships').
        quantile (float): The quantile threshold for capping (default is 0.95, i.e., the top 5% of outliers will be capped).
    
    Returns:
        pd.DataFrame: DataFrame with the capped column.
    """
    # Calculate the upper limit based on the specified quantile
    upper_limit = df[column].quantile(quantile)
    
    # Cap the values in the column
    df[f'{column}_capped'] = np.where(df[column] > upper_limit, upper_limit, df[column])
    
    # remove old feature
    df = df.drop(columns=column)
    
    return df


def create_label(df):
    y = df['status'].map({'acquired': 1, 'closed': 0})
    return y


def combine_rares_categories(df, column, threshold=3):
    # Count the number of occurrences
    category_counts = df[column].value_counts()
    
    # Replace rare categories with 'Rare'
    df[column] = df[column].apply(lambda x: x if category_counts[x] >= threshold else 'Rare')   
    
    return df


def remove_outliers(df, column, threshold=3):
    """
    Identifies and removes outliers in the specified column based on the given threshold (default is 3 standard deviations).
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to check for outliers (default is 'avg_participants').
        threshold (int or float): The multiplier of standard deviation for defining outliers (default is 3).
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
        pd.DataFrame: DataFrame containing the outliers that were removed.
    """
    # Calculate mean and standard deviation
    data_mean = df[column].mean()
    data_std = df[column].std()
    
    # Define the cut-off based on the threshold (e.g., 3 standard deviations)
    cut_off = data_std * threshold
    lower_bound = data_mean - cut_off
    upper_bound = data_mean + cut_off
    
    # Identify the outliers
    outliers = df.loc[(df[column] > upper_bound) | (df[column] < lower_bound)]
    
    # Remove the outliers from the DataFrame
    df_cleaned = df.loc[(df[column] <= upper_bound) & (df[column] >= lower_bound)]
    
    return df_cleaned, outliers


def oneHot_encode_columes(df, categorical_columns, remove_feature_names=True):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_columns])
    if remove_feature_names:
        encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))
    else:
        encoded_categorical_df = pd.DataFrame(encoded_categorical)

    return encoded_categorical_df


def standard_scale_columes(df, numerical_columns):
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)
    
    return scaled_numerical_df



def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def min_distance_to_hub(lat, lon, hub_coords):
    distances = [haversine(lat, lon, coords[0], coords[1]) for coords in hub_coords.values()]
    return min(distances)  # Return the smallest distance to any hub

def closest_hub(lat, lon, hub_coords):
    distances = {hub: haversine(lat, lon, coords[0], coords[1]) for hub, coords in hub_coords.items()}
    return min(distances, key=distances.get)  # Return the hub with the smallest distance

def closest_hub(lat, lon, hub_coords):
    distances = {hub: haversine(lat, lon, coords[0], coords[1]) for hub, coords in hub_coords.items()}
    return min(distances, key=distances.get)  # Return the hub with the smallest distance


def add_distance_bin_from_hub_feature(df):
    # Define coordinates for multiple US startup hubs
    main_hub_coords = {
        'San Francisco': (37.7749, -122.4194),
        'New York': (40.7128, -74.0060),
        'Austin': (30.2672, -97.7431),
        'Boston': (42.3601, -71.0589),
        'Los Angeles': (34.0522, -118.2437),
        'Seattle': (47.6062, -122.3321),
        'San Jose': (37.3382, -121.8863)
        }
    
    df['distance_from_hub'] = df.apply(lambda row: min_distance_to_hub(row['latitude'], row['longitude'], main_hub_coords), axis=1)
    
    # Fixed-width binning (e.g., 0-10km, 10-50km, etc.)
    bins = [0, 10, 50, 100, np.inf]  # Define custom distance bins in km
    labels = ['Very Close', 'Moderately Close', 'Far', 'Very Far']
    df['distance_bin'] = pd.cut(df['distance_from_hub'], bins=bins, labels=labels)

    df = df.drop(columns='distance_from_hub')
    df = df.drop(columns=['latitude', 'longitude'])
    
    # optinal:
    # df['closest_hub'] = df.apply(lambda row: closest_hub(row['latitude'], row['longitude'], main_hub_coords), axis=1)
    
    return df


def preprocess_data_classifier(df, useKNNImputer=False, remove_feature_names=True):
    """Preprocess the data: encoding categorical features, and scaling numerical features."""   
    
    df = process_funding_dates(df)
    
    # df = combine_rares_categories(df, column='city', threshold=1)
    
    # df = cap_feature(df, column='avg_participants', quantile=0.98) # !Use with caution – this removes data
    
    df, outliers1 = remove_outliers(df, column='funding_total_usd', threshold=3) # !Use with caution – this removes data
    
    df, outliers2 = remove_outliers(df, column='relationships', threshold=2) # !Use with caution – this removes data
    
    # df = add_distance_bin_from_hub_feature(df)
    
    # df = feature_log_scaler(df, column='funding_total_usd')

    # df = cap_feature(df, column='relationships', quantile=0.95)
    
    # Create label
    y = create_label(df)
    
    # Drop unnecessary columns
    df = df.drop(columns=['status', 'founded_at', 'name', 'id', 'state_code', 'object_id', 'labels', 'closed_at', 'Unnamed: 0', 
                'Unnamed: 6', 'zip_code', 'closed_at'])
    
    # we will create our own category one hotted- remove previous
    df = df.drop(columns=['is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech',
                'is_consulting', 'is_othercategory'])
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_categorical_df = oneHot_encode_columes(df, categorical_columns, remove_feature_names=remove_feature_names)
    
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    scaled_numerical_df = standard_scale_columes(df, numerical_columns)
    
    # numerical columns - MinMax on columns
    # maybe on some numerical columns we want to perform something else... like min-max
    
    # Combine encoded categorical and scaled numerical data
    processed_df = pd.concat([encoded_categorical_df, scaled_numerical_df], axis=1)
    
    # fill missing values
    if useKNNImputer:
        knn_imputer = KNNImputer(n_neighbors=5)
        if not remove_feature_names:
            processed_df.columns = processed_df.columns.astype(str)
        processed_df = pd.DataFrame(knn_imputer.fit_transform(processed_df), columns=processed_df.columns)
    else:
        processed_df.fillna(processed_df.mean(), inplace=True)
        
    # processed_df.to_csv('processed_df.csv', index=False)   
    
    return processed_df, y


def preprocess_data(df, useKNNImputer=False):
    """Preprocess the data: encoding categorical features, and scaling numerical features."""   
    # Create label
    #y = df['status'].map({'acquired': 1, 'closed': 0})

    # Count the occurrences of each city
    city_counts = df['city'].value_counts()

    # Set a threshold (e.g., categories that appear less than 3 times will be considered rare)
    threshold = 2

    # Replace rare categories with 'Other'
    df['city'] = df['city'].apply(lambda x: x if city_counts[x] >= threshold else 'Other')

    # Drop unnecessary columns
    df = df.drop(columns=['status', 'founded_at', 'name', 'id', 'state_code', 'object_id', 'labels', 'closed_at', 'Unnamed: 0', 
                'Unnamed: 6', 'zip_code', 'closed_at'])

    # combine rare cities
    category_counts = df['city'].value_counts()
    threshold = 3 # Set a threshold (e.g., cities that appear less than 3 times will be considered rare)
    df['city'] = df['city'].apply(lambda x: x if category_counts[x] >= threshold else 'Rare')    # Replace rare cities with 'Rare'

    df = df.drop(columns=['is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech',
                'is_consulting', 'is_othercategory'])

    # normalize avg_participants feature
    data_mean = df['avg_participants'].mean()
    data_std = df['avg_participants'].std()
    cut_off = data_std * 3
    lower_bound = data_mean - cut_off
    upper_bound = data_mean + cut_off
    df.loc[(df['avg_participants'] > upper_bound) | (df['avg_participants'] < lower_bound)]

    # normalize funding_total_usd feature
    data_mean = df['funding_total_usd'].mean()
    data_std = df['funding_total_usd'].std()
    cut_off = data_std * 3
    lower_bound = data_mean - cut_off
    upper_bound = data_mean + cut_off
    df.loc[(df['funding_total_usd'] > upper_bound) | (df['funding_total_usd'] < lower_bound)]

    df = df.drop(columns=['city'])

    X = df

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['number']).columns

    # OneHotEncode categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(X[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

    # Standardize numerical columns
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(X[numerical_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)

    # Combine encoded categorical and scaled numerical data
    processed_df = pd.concat([encoded_categorical_df, scaled_numerical_df], axis=1)

    if useKNNImputer:
        knn_imputer = KNNImputer(n_neighbors=5)
        processed_df = pd.DataFrame(knn_imputer.fit_transform(processed_df), columns=processed_df.columns)
    else:
        processed_df.fillna(processed_df.mean(), inplace=True)

    # df.to_csv('df.csv', index=False)

    return processed_df #, y


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
    processed_df = preprocess_data_classifier(df)

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
