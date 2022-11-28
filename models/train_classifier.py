# Import necessary libraries
import sys
import pandas as pd
import numpy as np
#Import for interact with SQLie
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
# Import visualization
import matplotlib.pyplot as plt
#import tokenization and ML model package
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#For trainning our models
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
# For reporting 
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Loads data from  our SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Our Features
    Y: Target value
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y

#This function for process text data 
def tokenize(text):
    
    """
    Describe: This is the function to tokenize text
    Input: messages or text data related with messages
    Output: list of words after processing the following steps

    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Builds classifier and tunes model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators': [8, 10],
        'clf__estimator__min_samples_split': [3, 4]
    }
    
    GS = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    
    return GS


def evaluate_model(model, X_test, Y_test):
    """
    Do assessment of model performance and export classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """
    Y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], Y_pred[:, index]))


def save_model(model, model_filepath):
    # Pickle best model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the ML model, trains the model, assess the model, saves the model."""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Assessing model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'store the ML model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()