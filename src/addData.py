import pandas as pd
import numpy as np
import yfinance as yf
# import tweepy 
import wbdata  # World Bank data API


def add_nasdaq_annual_changes(df):
    # Convert the date column to datetime
    df['founded_at_date'] = pd.to_datetime(df['founded_at'])

    # Extract year, month, and day into separate columns
    df['founded_at_year'] = df['founded_at_date'].dt.year
    df['founded_at_month'] = df['founded_at_date'].dt.month
    df['founded_at_day'] = df['founded_at_date'].dt.day
    
    # Define the range of years we're interested in
    years = np.arange(1984, 2024).tolist()
    
    # Download NASDAQ index data from Yahoo Finance, using monthly intervals from 1984 to 2024
    nasdaq_data = yf.download('^IXIC', start='1984-01-01', end='2024-01-01', interval='1mo')
    
    # Extract the year from the index (date) and add it as a new column
    nasdaq_data['Year'] = nasdaq_data.index.year
    
    # Group the data by year and calculate the annual percentage change in the closing price
    nasdaq_annual = nasdaq_data.groupby('Year')['Close'].last().pct_change().reset_index()
    nasdaq_annual.columns = ['Year', 'NASDAQ_Annual_Change']  # Rename columns for clarity
    
    # Filter the annual data to include only the years in the defined range
    nasdaq_annual = nasdaq_annual[nasdaq_annual['Year'].isin(years)]
    nasdaq_annual.set_index('Year', inplace=True)  # Set 'Year' as the index for easy lookup
    
    # Find the maximum founding year in the DataFrame to determine the range of years to add
    max_year = df['founded_at_year'].max()
    
    # Add a new column for each year relative to the founding year, containing the NASDAQ annual changes
    for year in range(max_year, max_year - 11, -1):
        year_index = max_year - year
        df[f'nasdaq_annual_changes_at_year_{year_index}'] = df['founded_at_year'].apply(
            lambda x: nasdaq_annual.loc[year, 'NASDAQ_Annual_Change'] if year in nasdaq_annual.index else np.nan
        )
    
    return df


def add_economic_indicators(df, indicator_code):
    # Define the fixed range of years to match the first function's range
    years_range = 10
    
    # Convert the date column to datetime
    df['founded_at_date'] = pd.to_datetime(df['founded_at'])

    # Extract year, month, and day into separate columns
    df['founded_at_year'] = df['founded_at_date'].dt.year
    
    # Calculate the start and end year for the World Bank data based on the founding years in the DataFrame
    start_year = df['founded_at_year'].min() - years_range
    end_year = df['founded_at_year'].max()

    # Fetch economic indicator data from the World Bank for the specified range of years
    countries = 'US'
    data = wbdata.get_dataframe({indicator_code: 'Economic_Indicator'}, country=countries)
    
    # Reset index and rename columns for clarity
    data = data.reset_index()
    data.rename(columns={indicator_code: 'Economic_Indicator'}, inplace=True)
    
    # Convert the 'date' column to datetime format to extract the year
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Extract the year from the 'date' column and set it as the index
    data['Year'] = data['date'].dt.year
    data = data.set_index(['Year'])
    
    # Filter the data to include only the years of interest (within the calculated range)
    years_of_interest = np.arange(start_year, end_year + 1)
    data = data.loc[data.index.isin(years_of_interest)]
    
    # Determine the key word to use in column names based on the indicator code
    if 'GDP' in indicator_code:
        key_word = 'GDP'
    elif 'UEM' in indicator_code:
        key_word = 'UEM'
    else:
        key_word = 'UNKNOWN'
    
    # Add a new column for each year relative to the founding year, containing the economic indicator data
    for i in range(years_range + 1):
        df[f'{key_word}_growth_at_year_{i}'] = df['founded_at_year'].apply(
            lambda x: data.loc[x - i, 'Economic_Indicator'] if (x - i) in data.index else np.nan
        )
    
    return df


###########################
# Load the startup data from a CSV file
df = pd.read_csv('data/startup_data.csv')

# Add NASDAQ annual changes to the DataFrame
df = add_nasdaq_annual_changes(df)

# Define the World Bank indicator code for GDP growth and add it to the DataFrame
indicator_code = 'NY.GDP.MKTP.KD.ZG'  # GDP growth (annual %)
df = add_economic_indicators(df, indicator_code)

# Define the World Bank indicator code for the unemployment rate and add it to the DataFrame
indicator_code = 'SL.UEM.TOTL.ZS'  # Unemployment rate, percentage of total labor force
df = add_economic_indicators(df, indicator_code)
