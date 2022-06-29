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


ROOT_DATA_DIR = "data"
TRANSFORMED_DATA_DIR_v1 = "transformed_data_v1"
TRANSFORMED_DATA_DIR_v2 = "transformed_data_v2"
GAMELOG_DIR = "gamelog_data"
GAME_ROTATION_DIR = "game_rotation_data"
TEAM_STAT_DIR = "team_stat_data"
PLAYER_STAT_DIR = "player_stat_data"

BUCKET_TRANSFORMED_v1 = "mert-kurttutan-nba-ml-project-transformed-data-v1"
BUCKET_TRANSFORMED_v2 = "mert-kurttutan-nba-ml-project-transformed-data-v2"
BUCKET_CONFIG = "mertkurttutan-nba-ml-project-config"

DATA_CONFIG_FILE = "data_config.json"

# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/').mkdir(parents=True, exist_ok=True)
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/').mkdir(parents=True, exist_ok=True)


##################   DEFINING IMPORTANT DATAFRAMES    #######################

def year_to_label(year: list):

  return f"{year}-{str(year+1)[-2:]}", f"{year}-{str(year+1)}"



def extract_data_from_s3(year_arr=None):
  global LABEL_ARR
  LABEL_ARR = None
  
  # Copy data from s3 bucket

  if year_arr is None:
    os.system(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v1}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive")
  else:
    
    LABEL_ARR = [year_to_label(year) for year in year_arr]
    for label in LABEL_ARR:
      os.system(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v1}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive --exclude \"*\"  --include \"*{label[0]}*\"  --include \"*{label[1]}*\" >> ~/entrypoint_log.log ")
      print(f"aws s3 cp s3://{BUCKET_TRANSFORMED_v1}/ {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1} --recursive --exclude \"*\" --include \"*{label[0]}*\"  --include \"*{label[1]}*\" ")

  # Copy config file from s3 bucket
  os.system(f"aws s3 cp s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} >> ~/entrypoint_log.log ")



def define_config_vars():
  print("Defining config vars...")
  global CONFIG_DICT, GAMELOG_FILES_SORTED, ROTATION_DIR_SORTED, PLAYER_STAT_FILES
  global YEAR_ARR, DF_GAMELOG_ARR, DF_GAME_ROTATION_ARR, DF_TEAM_STAT_ARR, DF_PLAYER_STAT_ARR


  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    CONFIG_DICT = json.load(read_content)

	# csv files to be iterated over
  GAMELOG_FILES_SORTED = sorted([gamelog_file for gamelog_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}") if ".csv" in gamelog_file])
  ROTATION_DIR_SORTED = sorted([rotation_file for rotation_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR}") if ".csv" in rotation_file])
  PLAYER_STAT_FILES = sorted([player_stat_file for player_stat_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{PLAYER_STAT_DIR}") if ".csv" in player_stat_file])


  # Create season from data_config
  
  if LABEL_ARR is None: 
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys())]))
  
  else:
    LABEL_ARR_0 = [label[0] for label in LABEL_ARR]
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys()) if elem[:-4] in LABEL_ARR_0]))


  print("Defining important dataframes...")




	# Extracted gamelog dataframe
  DF_GAMELOG_ARR = []
  for gamelog_file in GAMELOG_FILES_SORTED:
    if not ".csv" in gamelog_file:
      continue
    df_gamelog = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}/{gamelog_file}")
			
    # Append data of current season into total data
    DF_GAMELOG_ARR.append(df_gamelog)
			
			

  # Extracted game rotation array
  DF_GAME_ROTATION_ARR = []
  for game_rotation_file in ROTATION_DIR_SORTED:
    if not ".csv" in game_rotation_file:
      continue
    df_game_rotation_season = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR}/{game_rotation_file}")
    DF_GAME_ROTATION_ARR.append(df_game_rotation_season)

			



  # array of tables, each of which contain statistics only for one season type
  DF_TEAM_STAT_ARR = []

  # Iterate over years
  for year in YEAR_ARR:
			
    # Iterate over season types
    for seasonType in ["Pla", "Reg"]:              
									
      df_team_stat = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{TEAM_STAT_DIR}/team_stat_transformed_season{year}_{seasonType}.csv")
      df_team_stat["TEAM_ID_c"] = df_team_stat["TEAM_ID"]
      DF_TEAM_STAT_ARR.append(df_team_stat.groupby("TEAM_ID_c").ffill().reset_index(drop=True).sort_values(by=["GAME_DATE", "TEAM_ID"]).reset_index(drop=True))
			

			



  DF_PLAYER_STAT_ARR = []
  for season_label in PLAYER_STAT_FILES:
    if not ".csv" in season_label:
      continue
    df_player_stat = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{PLAYER_STAT_DIR}/{season_label}")
			
    df_player_stat["PLAYER_ID_c"] = df_player_stat["PLAYER_ID"]
    DF_PLAYER_STAT_ARR.append(df_player_stat.groupby("PLAYER_ID_c").ffill().reset_index(drop=True).sort_values(by=["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True))

    
    
    
def transform_team_stat(is_upload: bool = True, save_array: bool = False):

  # Make sure gamelog diretory exists
  Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{TEAM_STAT_DIR}/').mkdir(parents=True, exist_ok=True)
  
  print("Transforming Teams Stat data...")
  df_arr_team_stat_transformed = []
  for idx in range(1,len(DF_TEAM_STAT_ARR), 2):
    year = YEAR_ARR[(idx-1)//2]
    seasonType = "Reg"
    if idx == 1:
      filled_team_stat = DF_TEAM_STAT_ARR[idx]
    else:
      filled_team_stat_0 = etl.fill_missing_team_stat_regular(DF_TEAM_STAT_ARR[idx], DF_TEAM_STAT_ARR[idx-2], DF_GAMELOG_ARR[idx], 1)
      filled_team_stat = etl.fill_missing_team_stat_regular(filled_team_stat_0, DF_TEAM_STAT_ARR[idx-2], DF_GAMELOG_ARR[idx], 2)
      
    filled_team_stat.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{TEAM_STAT_DIR}/team_stat_transformed_season{year}_{seasonType}.csv", index=False)

    df_arr_team_stat_transformed.append(filled_team_stat)
    
    
    
  df_arr_team_stat_playoff_transformed = []
  for idx in range(len(df_arr_team_stat_transformed)):
    year = YEAR_ARR[idx]
    seasonType = "Pla"
    playoff_start = DF_TEAM_STAT_ARR[idx*2]["GAME_DATE"].min()
    playoff_idx = df_arr_team_stat_transformed[idx]["GAME_DATE"] >= playoff_start
    filled_team_stat_playoff = etl.fill_missing_team_stat_playoff(DF_TEAM_STAT_ARR[idx*2], df_arr_team_stat_transformed[idx].loc[playoff_idx,:])
    filled_team_stat_playoff.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{TEAM_STAT_DIR}/team_stat_transformed_season{year}_{seasonType}.csv", index=False)

    if save_array:
      df_arr_team_stat_playoff_transformed.append(filled_team_stat_playoff)

    
  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{TEAM_STAT_DIR} s3://{BUCKET_TRANSFORMED_v2}/{TEAM_STAT_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log" )

  return df_arr_team_stat_playoff_transformed
      
      
      
def transform_player_stat(is_upload: bool = True, save_array: bool = False):

  # Make sure gamelog diretory exists
  Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{PLAYER_STAT_DIR}/').mkdir(parents=True, exist_ok=True)

  print("Transforming Players Stat data...")
  
  df_arr_player_stat_transformed = []
  for idx in range(1,len(DF_PLAYER_STAT_ARR), 2):
    year = YEAR_ARR[(idx-1)//2]
    seasonType = "Reg"
    if idx == 1:
      filled_player_stat = DF_PLAYER_STAT_ARR[idx]
    else:
      filled_player_stat = etl.fill_missing_player_stat_regular(DF_PLAYER_STAT_ARR[idx], 
                                                                DF_PLAYER_STAT_ARR[idx-2], 
                                                                DF_GAMELOG_ARR[idx], 
                                                                DF_GAME_ROTATION_ARR[idx])
    filled_player_stat.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{PLAYER_STAT_DIR}/player_stat_transformed_season{year}_{seasonType}.csv", index=False)
    
    df_arr_player_stat_transformed.append(filled_player_stat)
  
  
  df_arr_player_stat_playoff_transformed = []
  for idx in range(len(df_arr_player_stat_transformed)):
    year = YEAR_ARR[idx]
    seasonType = "Pla"
    playoff_start = DF_TEAM_STAT_ARR[idx*2]["GAME_DATE"].min()
    playoff_idx = df_arr_player_stat_transformed[idx]["GAME_DATE"] >= playoff_start

    filled_player_stat_playoff = etl.fill_missing_player_stat_playoff(DF_PLAYER_STAT_ARR[idx*2], df_arr_player_stat_transformed[idx].loc[playoff_idx,:])
    filled_player_stat_playoff.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{PLAYER_STAT_DIR}/player_stat_transformed_season{year}_{seasonType}.csv", index=False)

    df_arr_player_stat_playoff_transformed.append(filled_player_stat_playoff)

      
  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/{PLAYER_STAT_DIR} s3://{BUCKET_TRANSFORMED_v2}/{PLAYER_STAT_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log" )


  return df_arr_player_stat_playoff_transformed
      
      

    
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--yearArr", nargs="+", default=["2014"], help="List of years to process data of ")
  value = parser.parse_args()
  year_arr = [int(year) for year in value.yearArr]

  print(f"Chosen year_arr: {year_arr}")
  print("Running transform-v2 script...")

  extract_data_from_s3(year_arr=year_arr)

  define_config_vars()
  
  transform_team_stat()
  
  transform_player_stat()

  
  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}/*")
  
  
  
  
  