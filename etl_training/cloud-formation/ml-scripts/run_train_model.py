import pandas as pd
from pathlib import Path


from typing import List
import argparse
import os
import nba_ml_module.utils as utils


# libraries for numerical and data processing
import pandas as pd
import numpy as np


# ML-specific libraries
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



# helpful util libraries
import json
import os


pd.set_option("display.max_columns", 500)


TRAINING_DIR = "training_data"
ROOT_DATA_DIR = "data"


BUCKET_TRAINING = "mert-kurttutan-nba-ml-project-training"
BUCKET_FILE = "mert-kurttutan-nba-ml-files"
CONFIG_DIR = "config"
MODEL_DIR = "models"


MODEL_FILE = "nba-pred.xgb"
DATA_CONFIG_FILE = "data_config.json"


# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{TRAINING_DIR}/').mkdir(parents=True, exist_ok=True)


def drop_player_stats(df: pd.DataFrame, inplace: bool) -> pd.DataFrame:
  cols_to_drop = [col for col in df.columns if"PStat" in col]

  if inplace:
    df.drop(cols_to_drop, axis=1, inplace=inplace)
    return None
  else:
    return df.drop(cols_to_drop, axis=1, inplace=inplace)


def extract_data_from_s3():

  # Copy config file from s3 bucket
  os.system(f"aws s3 cp s3://{BUCKET_FILE}/{CONFIG_DIR}/{DATA_CONFIG_FILE} {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} >> ~/entrypoint_log.log")

  # Download training data
  os.system(f"aws s3 cp s3://{BUCKET_TRAINING}/{TRAINING_DIR} {ROOT_DATA_DIR}/{TRAINING_DIR} --recursive --exclude \"*\"  --include \"*.csv\"  >> ~/entrypoint_log.log")


def define_config_vars():

  print("Defining config vars...")
  global CONFIG_DICT, TRAINING_FILES_SORTED
  global DF_TRAIN_ARR
    
  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    CONFIG_DICT = json.load(read_content)

  # csv files to be iterated over
  TRAINING_FILES_SORTED = sorted([train_file for train_file in os.listdir(f"./{ROOT_DATA_DIR}/{TRAINING_DIR}") if ".csv" in train_file])

  cols_to_drop = ["team2_IS_HOME", "team1_L", "team2_L", "team2_W", "team1_PTS", "team2_PTS", "team1_PLUS_MINUS", "team2_PLUS_MINUS","team1_L_cum", "team2_L_cum"]
  noise_cols = ["GAME_DATE", "GAME_ID", "team1_TEAM_ID", "team2_TEAM_ID"]


  DF_TRAIN_ARR = []
  for idx, train_file in enumerate(TRAINING_FILES_SORTED):
      if not ".csv" in train_file:
          continue
      df_train = pd.read_csv(f"{ROOT_DATA_DIR}/{TRAINING_DIR}/{train_file}")
      df_train = df_train.drop(columns=cols_to_drop+noise_cols).sort_values(by=["DAY_WITHIN_SEASON"])
      
      df_train["team1_IS_HOME"] = df_train["team1_IS_HOME"].astype("float")
      drop_player_stats(df=df_train, inplace=True)


      df_train["SEASON"] = int(train_file[14:18]) - 2007
      df_train["SEASON"] = int(train_file[14:18]) - 2007
      
      # Append data of current season into total data
      df_train = utils.sort_df_cols(df_train)
      DF_TRAIN_ARR.append(df_train)





def train(df_train_arr: List[pd.DataFrame], is_upload: bool = True, corr_th: float = 0.05, training_season_arr: List[int] = [12, 13], val_season: int = 13):


  # Get rid of stat of the very early time
  # We could not fill this since there is no previos timeline to use
  df_train_arr[1] = df_train_arr[1].loc[~df_train_arr[1]["team2_lag08_AST"].isna(), :]
  df_total = pd.concat(df_train_arr, ignore_index=True).sort_values(by=["SEASON", "IS_REGULAR", "DAY_WITHIN_SEASON"], ascending=[True, False, True])


  # Calculate correlation matrix
  cor = df_total.corr() 


  # Get the absolute value of the correlation
  cor_target = abs(cor["team1_W"])

  # Select highly correlated features (thresold = 0.2)
  relevant_features = cor_target[cor_target>corr_th]

  # Collect the names of the features
  names = [index for index, value in relevant_features.iteritems()]

  names = list(set(names+["SEASON", "DAY_WITHIN_SEASON", "IS_REGULAR"]))


  df_total_v2 = df_total[names]



  regular_idx = df_total_v2["IS_REGULAR"] == 1
  season_select_val = df_total_v2["SEASON"] == val_season & regular_idx
  val_day_idx = int(df_total_v2.loc[season_select_val, "DAY_WITHIN_SEASON"].max()*0.8)

  season_late = df_total_v2["DAY_WITHIN_SEASON"] > val_day_idx
  season_select = df_total_v2["SEASON"].isin(training_season_arr) & regular_idx

  val_idx =  season_select_val & season_late
  train_idx = season_select & ~val_idx

  X_train = df_total_v2.loc[train_idx,:]
  y_train = X_train.pop("team1_W")


  X_val = df_total_v2.loc[val_idx,:]
  y_val = X_val.pop("team1_W")


  print(f"Number training and validation instances: {X_train.shape[0]}, {X_val.shape[0]}")


  selected_cols = list(X_train.columns)


  CONFIG_DICT["selected_cols"] = selected_cols


  params = {
              "max_depth": 5,
              "n_estimators": 256,
              "min_child_weight": 0.8,
              "colsample_bytree": 0.5,
              "subsample": 0.4,
              "eta": 0.07,
              "seed": 43,
              "verbose": True,
              "early_stopping_rounds": 8,
          }

  # XGboost model
  model = xgb.XGBClassifier(
      max_depth=params['max_depth'],
      n_estimators=params['n_estimators'],
      min_child_weight=params['min_child_weight'], 
      colsample_bytree=params['colsample_bytree'], 
      subsample=params['subsample'], 
      eta=params['eta'],
      early_stopping_rounds = params["early_stopping_rounds"],
      #tree_method='gpu_hist',
      seed=params['seed'],
      n_jobs=4)


  model.fit(
      X_train, 
      y_train, 
      #eval_metric=params["eval_metric"], 
      eval_set=[(X_train, y_train), (X_val,y_val)], 
      verbose=params['verbose'], 
      )


  # Predictions
  yhat_val = model.predict(X_val)

  # Accuracy
  accuracy = (yhat_val == y_val).sum() / len(y_val)


  model.save_model(f"{ROOT_DATA_DIR}/{MODEL_FILE}")

  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "w") as outfile:
    json.dump(CONFIG_DICT, outfile)

  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} s3://{BUCKET_FILE}/{CONFIG_DIR}/{DATA_CONFIG_FILE} ")
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{MODEL_FILE} s3://{BUCKET_FILE}/{MODEL_DIR}/{MODEL_FILE} ")


if __name__ == "__main__":

  year_today = 13

  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--year-arr", nargs="+", default=["2014"], help="List of years to process data of ")
  value = parser.parse_args()
  year_arr = [int(year) for year in value.year_arr]

  if year_arr == [-1]:
    print("Current mode is chosen...")
    year_arr = [12, 13]


  val_season = year_arr[-1]
  training_season_arr = year_arr
  extract_data_from_s3()
  define_config_vars()

  train(df_train_arr=DF_TRAIN_ARR, val_season=val_season, training_season_arr=training_season_arr)