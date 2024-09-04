import pandas as pd
import numpy as np
import yfinance as yf
# import tweepy 
import wbdata  # World Bank data API


def add_nasdaq_annual_changes(df):
    # Years of interest
    years = np.arange(1984, 2024).tolist()

    # Convert the date column to datetime
    df['founded_at_date'] = pd.to_datetime(df['founded_at'])

    # Extract year, month, and day into separate columns
    df['founded_at_year'] = df['founded_at_date'].dt.year

    # Download data for the NASDAQ index
    nasdaq_data = yf.download('^IXIC', start='1984-01-01', end='2024-01-01', interval='1mo')

    # Calculate annual changes
    nasdaq_data['Year'] = nasdaq_data.index.year
    nasdaq_annual = nasdaq_data.groupby('Year')['Close'].last().pct_change().reset_index()
    nasdaq_annual.columns = ['Year', 'NASDAQ_Annual_Change']

    # Filter for the years of interest
    nasdaq_annual = nasdaq_annual[nasdaq_annual['Year'].isin(years)]

    # Set the 'Year' column as the index
    nasdaq_annual.set_index('Year', inplace=True)

    # Create new columns for each year
    df['nasdaq_annual_changes_at_year_0'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x, None))
    df['nasdaq_annual_changes_at_year_1'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-1, None))
    df['nasdaq_annual_changes_at_year_2'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-2, None))
    df['nasdaq_annual_changes_at_year_3'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-3, None))
    df['nasdaq_annual_changes_at_year_4'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-4, None))
    df['nasdaq_annual_changes_at_year_5'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-5, None))
    df['nasdaq_annual_changes_at_year_6'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-6, None))
    df['nasdaq_annual_changes_at_year_7'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-7, None))
    df['nasdaq_annual_changes_at_year_8'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-8, None))
    df['nasdaq_annual_changes_at_year_9'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-9, None))
    df['nasdaq_annual_changes_at_year_10'] = df['founded_at_year'].apply(lambda x: nasdaq_annual['NASDAQ_Annual_Change'].get(x-10, None))
    
    df = df.drop(columns=['founded_at_year', 'founded_at_date'])

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
    
    df = df.drop(columns=['founded_at_year', 'founded_at_date'])
    
    return df

def main():
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
    
if __name__ == "__main__":
    main()
