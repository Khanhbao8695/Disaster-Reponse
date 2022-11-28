# Disaster Response Pipeline Project
Author by Bao Khanh Nguyen

## Table of Contents
- [Disaster Response Pipeline](#disaster-response-pipeline)
  - [Table of Contents](#table-of-contents)
    - [Project Motivation](#project-motivation)
    - [File Descriptions](#file-descriptions)
    - [Components](#components)
      - [1. ETL Pipeline](#1-etl-pipeline)
      - [2. ML Pipeline](#2-ml-pipeline)
      - [3. Flask Web App](#3-flask-web-app)
    - [Instructions:](#instructions)
    - [Acknowledgements](#acknowledgements)

## **Environment Setup**

**Environment**
- OS: Windows 10

- Interpreter: Visual Studio Code

- Python version: Python 3.8+

## Libraries
- Please install all packages using requirements.txt file. YOu could use this command to install: `pip install -r requirements.txt`

### Project Motivation

For this project about analyzing disaster, i would like to apply my data engineering skills to construct a model for an API that could able to categorizes disaster messages. In addition, I also would like to create a create a machine learning pipeline to categorize real-world message sent during disasters so that these messages could be routed to the apporirate disaster relief organization. The project also includes a web app through which an emergency agents could enter a new message and receive classification results in a variety of categories. This web app also has feature to display visualizations of the disaster data. This project's dataset is collected from [Append](https://appen.com/) (Previously was **Figure 8**)


### File Descriptions
app    

| - template    
| |- master.html # main page of our web app    
| |- go.html # web app shows result of classification report
|- run.py # Flask file for running app    


data    

|- disaster_categories.csv # data categories for processing 
|- disaster_messages.csv # data messages for processing    
|- process_data.py # Py file to run our cleaning pipeline   
|- InsertDatabaseName.db # a database where we save our cleaned data     


models   

|- train_classifier.py # machine learning (ML) pipeline     
|- classifier.pkl # where we save our trained model     

|-- README
|-- requirements.txt

### Components
There are three parts need to complete for this project. 

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline:

 - Import the dataset about messages and categories in csv format
 - Merges the two datasets together
 - Cleans the dataset
 - Stores it in a SQLite database
 
A Jupyter notebook `ETL Pipeline` was used to do EDA, and later for processing the `process_data.py`script. 
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

 - Imports data from the SQLite database, which we just saved in part 1
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs classification results on the test set
 - Exports the final model as a pickle file
 - Improve our model
 
A Jupyter notebook `ML Pipeline` was used to do EDA to prepare the train_classifier.py script. 

#### 3. Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification report results in different categories. In this web app, the data would be visualized.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - For running `ETL pipeline` that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - For running `ML pipeline` that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: using `cd app`

3. Run the web app with: `python run.py`

4. Click the `PREVIEW` button to open the homepage
