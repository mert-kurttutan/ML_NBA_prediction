#!/usr/bin/env python3
import sys
import pandas as pd
import argparse
import json
import os


from pathlib import Path


# Local modules
import nba_ml_module.etl as etl


print("Setting config variables...")


ROOT_DATA_DIR = "./data"
TRANSFORMED_DATA_DIR_v1 = "transformed_data_v1"
TRANSFORMED_DATA_DIR_v2 = "transformed_data_v2"
TRAINING_DATA_DIR = "training_data"

GAMELOG_DIR = "gamelog_data"
GAME_ROTATION_DIR = "game_rotation_data"
TEAM_STAT_DIR = "team_stat_data"
PLAYER_STAT_DIR = "player_stat_data"
TRAINING_DIR = "training_data"

BUCKET_TRANSFORMED_v1 = "mert-kurttutan-nba-ml-project-transformed-data-v1"
BUCKET_TRANSFORMED_v2 = "mert-kurttutan-nba-ml-project-transformed-data-v2"
BUCKET_TRAINING = "mert-kurttutan-nba-ml-project-training"
BUCKET_CONFIG = "mert-kurttutan-nba-ml-files/config"

DATA_CONFIG_FILE = "data_config.json"

# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/').mkdir(parents=True, exist_ok=True)
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/').mkdir(parents=True, exist_ok=True)
Path(f'{ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/').mkdir(parents=True, exist_ok=True)



def year_to_label(year: list):

  return f"{year}-{str(year+1)[-2:]}", f"{year}-{str(year+1)}"



##################   DEFINING IMPORTANT DATAFRAMES    #######################

def extract_data_from_s3(year_arr=None):

  global LABEL_ARR
  LABEL_ARR = None


  # Copy data from s3 bucket

  if year_arr is None:
    os.system(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v1}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive >> ~/entrypoint_log.log")
    os.system(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v2}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive >> ~/entrypoint_log.log")
  
  else:
    LABEL_ARR = [year_to_label(year) for year in year_arr]
    for label in LABEL_ARR:
      os.system(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v1}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive --exclude \"*\"  --include \"*{label[0]}*\"  --include \"*{label[1]}*\" >> ~/entrypoint_log.log ")
      print(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v1}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive --exclude \"*\" --include \"*{label[0]}*\"  --include \"*{label[1]}*\" ")

      os.system(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v2}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2} --recursive --exclude \"*\"  --include \"*{label[0]}*\"  --include \"*{label[1]}*\" >> ~/entrypoint_log.log ")
      print(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v2}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2} --recursive --exclude \"*\" --include \"*{label[0]}*\"  --include \"*{label[1]}*\" ")

  # Copy config file from s3 bucket
  os.system(f"aws s3 cp s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} >> ~/entrypoint_log.log")



def define_config_vars():

  print("Defining config vars...")
  global CONFIG_DICT, GAMELOG_FILES_SORTED, ROTATION_DIR_SORTED, PLAYER_STAT_FILES, TEAM_STAT_FILES
  global YEAR_ARR, DF_GAMELOG_ARR, DF_GAME_ROTATION_ARR, DF_TEAM_STAT_ARR, DF_PLAYER_STAT_ARR
    
  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    CONFIG_DICT = json.load(read_content)

  # csv files to be iterated over
  GAMELOG_FILES_SORTED = sorted([gamelog_file for gamelog_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}") if ".csv" in gamelog_file])
  ROTATION_FILES_SORTED = sorted([rotation_file for rotation_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR}") if ".csv" in rotation_file])
  PLAYER_STAT_FILES = sorted([player_stat_file for player_stat_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{PLAYER_STAT_DIR}") if ".csv" in player_stat_file])
  TEAM_STAT_FILES = sorted([team_stat_file for team_stat_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{TEAM_STAT_DIR}") if ".csv" in team_stat_file])

  # Create season from data_config
  if LABEL_ARR is None: 
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys())]))
  
  else:
    LABEL_ARR_0 = [label[0] for label in LABEL_ARR]
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys()) if elem[:-4] in LABEL_ARR_0]))

  print(f"Here is year_arr: {YEAR_ARR}")

  print("Defining important dataframes...")


  DF_GAMELOG_ARR = []
  for gamelog_file in GAMELOG_FILES_SORTED:
    if not ".csv" in gamelog_file:
      continue
    df_gamelog = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}/{gamelog_file}")#
    
    if df_gamelog.shape[0] > 400:
      df_gamelog["IS_REGULAR"] = 1
    else:
      df_gamelog["IS_REGULAR"] = 0

    
    # Append data of current season into total data
    DF_GAMELOG_ARR.append(df_gamelog)
    
    
    
  DF_TEAM_STAT_ARR = []
  for team_stat_file in TEAM_STAT_FILES:
    if not ".csv" in team_stat_file:
      continue
    df_team_stat = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{TEAM_STAT_DIR}/{team_stat_file}")
			
    # Append data of current season into total data
    DF_TEAM_STAT_ARR.append(df_team_stat)
    
    
  DF_GAME_ROTATION_ARR = []
  for season_label in ROTATION_FILES_SORTED:
    if not ".csv" in gamelog_file:
      continue
    df_game_rotation_season = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR}/{season_label}")
    
    DF_GAME_ROTATION_ARR.append(df_game_rotation_season)
    
    
  DF_PLAYER_STAT_ARR = []
  for player_stat_file in PLAYER_STAT_FILES:
    if not ".csv" in player_stat_file:
      continue

    df_player_stat = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{PLAYER_STAT_DIR}/{player_stat_file}")
    
    # Append data of current season into total data
    DF_PLAYER_STAT_ARR.append(df_player_stat)

    
def get_team_stat_training(is_upload: bool = True, save_array: bool = False):
    
  Path(f'{ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/{TRAINING_DIR}/').mkdir(parents=True, exist_ok=True)
  df_gamelog_merged_arr = []
  for gamelog_file, df_gamelog, df_team_stat in zip(GAMELOG_FILES_SORTED, DF_GAMELOG_ARR, DF_TEAM_STAT_ARR):
    if not ".csv" in gamelog_file:
      continue

    merged_0 = etl.merge_team_stat_to_gamelog(df_gamelog, df_team_stat, 1, False)
    merged_1 = etl.merge_team_stat_to_gamelog(merged_0, df_team_stat, 2, True).astype({"team1_IS_HOME": int,})

    merged_1.to_csv(f"{ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/{TRAINING_DIR}/{gamelog_file}", index=False)


    # Append data of current season into total data
    if save_array:
      df_gamelog_merged_arr.append(merged_1)   

  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/{TRAINING_DIR} s3://{BUCKET_TRAINING}/{TRAINING_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log " )

  return df_gamelog_merged_arr
    
    
    
    
def get_team_player_stat_training(is_upload: bool = True, save_array: bool = False):
    

    
  Path(f'{ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/{TRAINING_DIR}/').mkdir(parents=True, exist_ok=True)
    
  df_gamelog_team_player_stat_merged_arr = []
  for gamelog_file, df_gamelog, df_team_stat, df_player_stat, df_game_rotation in zip(GAMELOG_FILES_SORTED, DF_GAMELOG_ARR, DF_TEAM_STAT_ARR, DF_PLAYER_STAT_ARR, DF_GAME_ROTATION_ARR):

    if not ".csv" in gamelog_file:
      continue

    merged_0 = etl.merge_team_stat_to_gamelog(df_gamelog, df_team_stat, 1, False)
    merged_1 = etl.merge_team_stat_to_gamelog(merged_0, df_team_stat, 2, False)

    merged_final = etl.get_agg_player_stat_to_gamelog(merged_1, df_player_stat, df_game_rotation)
    
    merged_final.to_csv(f"{ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/{TRAINING_DIR}/{gamelog_file}", index=False)

    # Append data of current season into total data
    if save_array:
      df_gamelog_team_player_stat_merged_arr.append(merged_final)
        
  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRAINING_DATA_DIR}/{TRAINING_DIR} s3://{BUCKET_TRAINING}/{TRAINING_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log" )

  return df_gamelog_team_player_stat_merged_arr





if __name__ == '__main__':

  year_today = 2013

  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--year-arr", nargs="+", default=["2014"], help="List of years to process data of ")
  value = parser.parse_args()
  year_arr = [int(year) for year in value.year_arr]

  if year_arr == [-1]:
    print("Current mode is chosen....")
    year_arr = [year_today]

  print(f"Chosen year_arr: {year_arr}")

  extract_data_from_s3(year_arr=year_arr)

  define_config_vars()
  
  
  get_team_player_stat_training()

  
  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}/*")