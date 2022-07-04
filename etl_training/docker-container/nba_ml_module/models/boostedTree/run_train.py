#!/usr/bin/env python3

# libraries for numerical and data processing
import pandas as pd
import numpy as np


# ML-specific libraries
import xgboost as xgb
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# libraries for visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# helpful util libraries
import random
import json
import os
import joblib
from datetime import datetime, timedelta
from functools import partial


pd.set_option("display.max_columns", 500)

print("Setting config variables...")

TRANSFORMED_DATA_DIR = "transformed_data"
TRANSFORMED_DATA_DIR_v2 = "transformed_data_v2"
GAMELOG_DIR = "gamelog_data"
GAME_ROTATION_DIR = "game_rotation_data"
TEAM_STAT_DIR = "team_stat_data"
PLAYER_STAT_DIR = "player_stat_data"
TRAINING_DIR = "training_data"
TRAINING_DIR_v2 = "training_data_v2"


BUCKET = "mert-kurttutan-nba-project-v1"
BUCKET_TRANSFORMED = "mert-kurttutan-nba-project-transformed-data-v1"




cols_to_drop_v2 = ["team1_TEAM_ID", "team2_TEAM_ID", 
                "team1_L", "team2_W", "team2_L", "team1_PLUS_MINUS", "team1_PTS", 
                "team2_PLUS_MINUS", "team2_PTS", "team2_IS_HOME"]

training_files_v2_sorted = sorted([gamelog_file for gamelog_file in os.listdir(f"../../data/{TRANSFORMED_DATA_DIR_v2}/{TRAINING_DIR_v2}") if ".csv" in gamelog_file])


import subprocess

# Check if gpu can be detected by program
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    is_gpu = True
    
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')
    is_gpu = False



# Reading data
df_train_arr_v2 = []
for idx, train_file in enumerate(training_files_v2_sorted):
    if not ".csv" in train_file:
        continue
    df_gamelog = pd.read_csv(f"../../data/{TRANSFORMED_DATA_DIR_v2}/{TRAINING_DIR_v2}/{train_file}")
    df_gamelog = df_gamelog.drop(columns=cols_to_drop_v2).sort_values(by=["DATE_WITHIN_SEASON"])



    if idx%2 == 0:
      df_gamelog["SEASON"] = idx+1

    else:
      df_gamelog["SEASON"] = idx-1
    
    # Append data of current season into total data
    df_train_arr_v2.append(df_gamelog)


    
print("Processing Data...")

# Get rid of stat of the very early time
# We could not fill this since there is no previos timeline to use
df_train_arr_v2[1] = df_train_arr_v2[1].loc[~df_train_arr_v2[1]["team2_lag08_AST"].isna(), :]


data_v2 = pd.concat(df_train_arr_v2).sort_values(by=["SEASON", "DATE_WITHIN_SEASON"])

# Get index of games that took place after pandemic
pandemic_regular_idx = data_v2["DATE_WITHIN_SEASON"] > 200

# Shift date_within season to adapt to pandemic shift
max_within_season = data_v2.loc[pandemic_regular_idx, "DATE_WITHIN_SEASON"].max()

# Deal with date shift during pandemic
def shift_to(x, end_value: int = 180, max_value: int = max_within_season):
  return x - max_value + end_value

data_v2.loc[pandemic_regular_idx, "DATE_WITHIN_SEASON"] = shift_to(data_v2.loc[pandemic_regular_idx, "DATE_WITHIN_SEASON"])



#### Feature selection by correlation matrix

# Calculate correlation matrix
cor = data_v2.corr() 


#### Get name of the selected features

# Get the absolute value of the correlation
cor_target = abs(cor["team1_W"])

# Select highly correlated features (thresold = 0.2)
relevant_features = cor_target[cor_target>0.05]

# Collect the names of the features
names = [index for index, value in relevant_features.iteritems()]

# Display the results
# print(names, len(names))

names += ["SEASON", "DATE_WITHIN_SEASON"]


# Keep only the selected columns

data_v2 = data_v2[names]



def train_model(model_params, model_bin_name: str):


    
    # Use xgboost gpu implementation if gpu detected in machine
    tree_method = "gpu_hist" if is_gpu else "auto"
    # XGboost model
    model = xgb.XGBClassifier(
        **model_params,
        tree_method=tree_method,
        n_jobs=4)
    
    split_num = 1
    
    total_score = 0

    # Calculate score of model by taking weighted average over 5 splits
    for i in range(split_num):
        

        regular_idx = (data_v2["SEASON"]%2 == 0)


        val_idx = (data_v2["DATE_WITHIN_SEASON"] > 155) 
        train_idx = ~val_idx

        X_train = data_v2.loc[regular_idx&train_idx,:]
        y_train = X_train.pop("team1_W")


        X_val = data_v2.loc[regular_idx&val_idx,:]
        y_val = X_val.pop("team1_W")

        # Apply min-max normalization
        #scaler = MinMaxScaler().fit(X_train)
        #scaler = StandardScaler().fit(X_train)
        #X_train = scaler.transform(X_train)
        #X_val = scaler.transform(X_val)


        model.fit(
            X_train, 
            y_train, 
            #eval_metric=params["eval_metric"], 
            eval_set=[(X_train, y_train), (X_val,y_val)], 
            verbose=params['verbose'])


        # Predictions
        yhat_val = model.predict(X_val)

        # Accuracy
        accuracy = (yhat_val == y_val).sum() / len(y_val)

        
        total_score += accuracy
    
    # Take the averegge
    total_score = total_score / split_num
    
    version = "v1"
    # Save model
    model.save_model(model_bin_name )
    
    print(f"Validation scores is {total_score}")
    

  
  
if __name__ == '__main__':
  
  version="v1"
  study_hyper = joblib.load(f"../../../bins/study_xgboost_hyparam_{version}.hyp")
  best_params = study_hyper.best_params
  best_params["verbose"] = True
  
  print("Started training....")
  
  model_bin_name = f"../../../bins/nba_classifier_{version}.xgb"
  train_model(best_params, model_bin_name)

