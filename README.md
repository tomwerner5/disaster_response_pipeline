# Disaster Response Pipeline Project

A web app for sending disaster response tweets to the appropriate organizations

### Table of Contents

1. [Project Motivation](#motivation)
2. [Instructions and Installation](#installation)
3. [File and Project Descriptions](#files)
4. [Licensing and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

Having a data pipeline that could update local agencies about emergencies happening in their area would be invaluable. By analyzing raw text of actual messages sent to these agencies, this project aims to build a machine learning pipeline to process and model the data, as well as predict what disaster category a new message belongs to. A web app (built using Flask) is then used to communicate these results in a simple way. 

## Instructions and Installation <a name="installation"></a>

The data and models for this dashboard are stored in this repository as-is, so to run the application, download or clone the repository and navigate to the app directory (`cd app`). Then, run the `run.py` script using Python (`python run.py`). If retraining or tewaking is desired, simply follow the below instructions to recreate the application.

#### Instructions

0. Install package requirements for Python:
    - numpy
    - pandas
    - sqlalchemy
    - nltk
    - plotly
    - flask
    - sqlite3
    - scikit-learn

1. If you want to use the data that has already been processed in this repository, **skip this step**. Otherwise, run the following commands after navigating to the project's root directory (using your favorite CLI) to set up your database and model. 

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run `cd app` to switch to the app directory.

3. Run the following command in the app's directory to run the web app.
    `python run.py`

    If successful, the terminal will look something like this:
    
    ![Successful Flask Deployment](pics/flask_deployment.png)

4. Go to the address defined in the terminal. It will likely be either `http://0.0.0.0:3001/` or `http://127.0.0.1:3001/`

5. You should see something like the following: 

![Home Page](pics/home_page.png)

On the home page, you can look at the plots or type in a query for the model to predict. 

The first two plots display distributions of the data, the first being the distribution of genre for each messgae, and the second is the the number of messages that appear in each of the target categories. 

The third plot is a T-SNE plot of the messages, which were clustered from the tf-idf matrix using K-Means. The K-Means model used 36 clusters, which is equal to the number of targets. The goal was to try to visualize the clusters in 2D using T-SNE to see if there were any patterns. Underneath the T-SNE plot is a data table. The data table displays the top ten words associated with each cluster centroid. If you hover over a point on the graph, you can see that the cluster that the point belongs to is labeled. Using the data table in combination with the plot, we can try to determine if the clusters are meaningful by seeing which words belong to each cluster.

## File and Project Descriptions <a name="files"></a>

Below is an overview of the directory:

    .
	|-- app
    	|-- templates
    		|-- master.html
        		|-- This is the home page for the application, which contains a
            		query bar and some initial plots
    		|-- go.html (Query prediction and results)
        		|-- After submitting a query, the application is directed here
            		to make a prediction
		|-- run.py
    		|-- Running this script will start the webserver
		|-- tsne.pkl (pickle file for storing T-SNE results)
    		|-- This file is for convenience. It can take a while to run TSNE,
        		so this saves some time by storing the tsne results
	|-- data
		|-- disaster_categories.csv
    		|-- This dataset contains the disaster response categories
        		(target variables)
		|-- disaster_messages.csv (messages dataset)
    		|-- This dataset contains the messages that were collected
		|-- DisasterResponse.db (combined dataset, sqlite file)
    		|-- A sqlite database file containing the combined (and cleaned)
        		data from disaster_messages.csv and disaster_categories.csv
		|-- process_data.py (Preprocessing steps to load into database)
    		|-- This script first loads the data from the csv files in the 
        		directory. Then, it cleans and prepares the target variables
        		for use in an ML model. Once this is completed, the data is
        		exported to DisasterResponse.db
	|-- models
    	|-- classifier.pkl 
        	|-- This object stores the model for convenience in running the app
		|-- train_classifier.py 
    		|-- After tokenizing/normalizing the messages from the sqlite data,
        		it builds the model pipeline which includes tokenization,
        		tf-idf, and grid searching the hyperparameters. Once completed,
        		it exports the results to classifier.pkl
    |-- README.md
    |-- LICENSE
    |-- .gitignore

The three directories (app, data, and models) break up this project into it's
three main components, which are the Flask app, the ETL process, and the model
pipeline.

## Licensing and Acknowledgements<a name="licensing"></a>

This project is provisioned under the MIT License. I would like to acknowledge
and thank Figure Eight for making this data available to me.
