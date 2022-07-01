# ML_NBA_prediction
This repository includes data processing tools ans steps taken to create and ML model for nba match result prediction and Web app to predict the results of nba matches using boosted tree and Deep Neural Nets (DNN). The purpose of this repository is 2-fold:


* First, we want prepare the database and ml model model to make prediction for the result of NBA match. The models to be used boosted tree (XGBoost) and DNN (Tensorflow). The database to be used is Amazon DynamoDB (NoSQL) since the queries we will make key-based queries and I want to make whole process as AWS-native as possible. 

* The second purpose is to deploy web-app that gives nba ml prediction using prepared ML model and database.


Let's start with etl and training process. First is the extraction of the raw data from web sites and api endpoints. I started by using the endpoint for nba stats stats.nba.com. The best documentation I could find on this endpoint is the github repo [nba_api](https://github.com/swar/nba_api) and in particular [documentation page](https://github.com/swar/nba_api/tree/master/docs/nba_api/stats/endpoints) . In the following, etl process consists of python scripts contained in the ml-scripts folder, which will be used to defined ECS task in AWS ECS cluster.

Before moving onto python scripts, we need necessary storage and compute services. For storage, for early steps of etl I decided to store files in csv format in AWS S3. To see the necessary files, we need to look at the bucket name variables in each python script.

In addition s3, dynamodb will be used - needed especially for the web app.

First scripts and task is the extraction of gamelogs. This collects gamelog which contains data about the already played matches. It also updates some configuration settings e.g. season start and end date. Based on this script, we have the task definition to be used in ECS cluster with Apache Airflow.

The following steps will be explained soon
- setting appropriate configs (AWS credentials, configs etc) 
- how to reproduce data processing steps, using automation tools (e.g. Apache Airflow)
- run the app

