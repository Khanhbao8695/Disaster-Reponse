# import  necessary libraries 
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(disaster_messages, disaster_categories):
    """
    Loads and merges datasets from 2 filepaths.
    
    Parameters:
    disaster_messages: messages csv file
    disaster_categories: categories csv file
    
    Returns:
    df: dataframe containing disaster_messages and disaster_categories merged
    
    """
    # load datasets
    df_mess= pd.read_csv(disaster_messages)
    df_cat = pd.read_csv(disaster_categories)
    # merge datasets on common id and assign to df
    df = df_mess.merge(df_cat, how ='outer', on =['id'])
    return df

def clean_data(df):
    """
    Cleans the dataframe.
    
    Parameters:
    df: DataFrame
    
    Returns:
    df: Cleaned DataFrame
    
    """
    #create a new columns for our df_cat dataframe 
    df_cat = df['categories'].str.split(';', expand=True)
    # Get the first row (n=1) of the categories dataframe
    row = df_cat.head(1)
    # use this row to retrieve a list of new headers name for our df_cat.
    ## Instead of using the define def function, let's use the lambda function for slicing each string up until the last two characters.
    cat_headers = row.applymap(lambda x: x[:-2]).iloc[0,:]
    # rename the columns of 'categories'
    df_cat.columns = cat_headers
    
    # Use the for loop to loop through columns in df_cat
    for column in df_cat:
        # Assign value of the column = the last character in the column. For example related-1 = 1
        df_cat[column] = df_cat[column].astype(str).str[-1]
        # # convert column from string to numeric
        df_cat[column] = df_cat[column].astype(int)
    # replace related-2 with relate-1 in related column
    df_cat['related'] = df_cat['related'].replace(to_replace=2, value=1)
        
    # drop the old categories column from df dataframe master
    ## Do the drop, enable inplace=True for confirmation
    df.drop('categories', axis=1, inplace=True)
    #  replace the old old with new header names
    # Let's do the concatenate the original dataframe with the new header name from df_cat  
    df = pd.concat([df, df_cat], axis=1)
    # Do the drop with inplace =True to confirm
    df.drop_duplicates(inplace=True)
    return df

    # Save our dataset into SQLite database
def save_data(df, database_filepath):
    """Stores df in a SQLite database."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  


# Master function to do everything about:
def main():
    """Loads data, cleans data, saves data to database"""
    if len(sys.argv) == 4:

        disaster_messages, disaster_categories, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(disaster_messages, disaster_categories))
        df = load_data(disaster_messages, disaster_categories)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the disaster_categories and disaster_messages dataset in csv format '\
              ' as the first and second argument respectively, and input the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
