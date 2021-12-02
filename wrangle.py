import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split


############# Acquire #############
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
def new_zillow_data():
    '''
    This function reads the Bed/Bath count, Finished Sq Ft, Taxable Value, Year Built, and Amount Taxed
    from the 2017 Properties Zillow data from the Codeup SQL server into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                       taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid LIKE '261'
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from the Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df





############# Prepare #############

### Remove outliers
def remove_outliers(df, k, col_list):
    ''' 
    Takes in a df, k, and list of columns returns
    a df with removed outliers
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def prepare(df):
    '''
    Takes a zillow df to handle nulls, remove nulls, change column types,
    drop duplicates, and remove outliers.
    '''

    # run df through remove_outliers function for all columns
    df = remove_outliers(df, 1.5, df.columns.to_list())
    
    # drop duplicates
    df.drop_duplicates()

    # for loop to change dtypes of appropriate columns to int
    for col in df.columns[df.columns != 'taxamount']:
        df[col] = df[col].astype(int)

    return df




############# Split #############

def split_zillow(df):
    '''
    Takes in a zillow dataframe and returns train, validate, test subset dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train, test_size = .3, random_state = 123)
    return train, validate, test


############# Split #############
def wrangle_zillow():
    '''
    Combines all previously defined functions in the module to return
    train, validate, and test datasets.
    '''
    df = get_zillow_data()
    df = prepare(df)
    train, validate, test = split_zillow(df)
    return train, validate, test