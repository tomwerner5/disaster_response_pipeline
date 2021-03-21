import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the disaster reponse data from csv files, and merges them into a 
    dataframe.

    Parameters
    ----------
    messages_filepath : str
        Location/filepath of messages.csv.
    categories_filepath : str
        Location/filepath of categories.csv.

    Returns
    -------
    df : pandas dataframe
        A merged dataframe containing the messages and categories data.

    '''
    # load Datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(left=messages, right=categories, on='id')
    return df


def clean_data(df):
    '''
    Performs preprocessing and cleaning steps on `df` (disaster response data).
    

    Parameters
    ----------
    df : pandas dataframe
        Input data to clean.

    Returns
    -------
    df : pandas dataframe
        Cleaned data.

    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [cat.split('-')[0] for cat in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert columns to numeric
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.to_numeric(categories[column].str[-1])
    
    # Ensure only zeros or 1's
    categories.values[categories.values > 1] = 1
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)

    # drop duplicates, if any
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Save a dataframe to a specified sqlite database.

    Parameters
    ----------
    df : pandas dataframe
        Data to save to sqlite database.
    database_filename : str
        The location/filename of the database.

    Returns
    -------
    None.

    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')  


def main():
    '''
    Performs the tasks of loading the data from csv files, cleaning the data,
    and storing the data in a sqlite database.

    Returns
    -------
    None.

    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()