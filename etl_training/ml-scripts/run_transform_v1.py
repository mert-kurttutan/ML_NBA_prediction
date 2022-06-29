#!/usr/bin/env python3
import os
import sys
import pandas as pd
import argparse
import json
import os
cwd = os.getcwd()



from datetime import datetime, timedelta

from pathlib import Path


  


# Local modules
import nba_ml_module.utils as utils
import nba_ml_module.etl as etl



print("Setting config variables...")


ROOT_DATA_DIR = "data"
RAW_DATA_DIR = "raw_data"
TRANSFORMED_DATA_DIR_v1 = "transformed_data_v1"
GAMELOG_DIR = "gamelog_data"
GAME_ROTATION_DIR = "game_rotation_data"
TEAM_STAT_DIR = "team_stat_data"
PLAYER_STAT_DIR = "player_stat_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_TRANSFORMED_v1 = "mert-kurttutan-nba-ml-project-transformed-data-v1"
BUCKET_CONFIG = "mertkurttutan-nba-ml-project-config"

DATA_CONFIG_FILE = "data_config.json"


# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/').mkdir(parents=True, exist_ok=True)
Path(f'{ROOT_DATA_DIR}/{RAW_DATA_DIR}/').mkdir(parents=True, exist_ok=True)
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/').mkdir(parents=True, exist_ok=True)

  
################################### EXTRACT GAME LOG DATA   #############################################
  
def year_to_label(year: list):

  return f"{year}-{str(year+1)[-2:]}", f"{year}-{str(year+1)}"



def extract_raw_data_from_s3(year_arr=None):
  global LABEL_ARR
  LABEL_ARR = None
  
  # Copy raw data from s3 bucket

  if year_arr is None:
    os.system(f"aws s3 cp s3://{BUCKET_RAW}/ {ROOT_DATA_DIR}/{RAW_DATA_DIR} --recursive")
  else:
    
    LABEL_ARR = [year_to_label(year) for year in year_arr]
    for label in LABEL_ARR:
      os.system(f"aws s3 cp s3://{BUCKET_RAW}/ {ROOT_DATA_DIR}/{RAW_DATA_DIR} --recursive --exclude \"*\"  --include \"*{label[0]}*\"  --include \"*{label[1]}*\" >> ~/entrypoint_log.log")
      print(f"aws s3 cp s3://{BUCKET_RAW}/ {ROOT_DATA_DIR}/{RAW_DATA_DIR} --recursive --exclude \"*\" --include \"*{label[0]}*\"  --include \"*{label[1]}*\" ")

  # Copy config file from s3 bucket
  os.system(f"aws s3 cp s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} >> ~/entrypoint_log.log")


  



def define_config_vars():
  print("Defining config vars...")
  global CONFIG_DICT, GAMELOG_FILES_SORTED, GAME_ROTATION_RELEVANT_COLS, ROTATION_DIR_SORTED
  global YEAR_ARR


  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    CONFIG_DICT = json.load(read_content)

  GAMELOG_FILES_SORTED = sorted([gamelog_file for gamelog_file in os.listdir(f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{GAMELOG_DIR}")])
  GAME_ROTATION_RELEVANT_COLS = ["PERSON_ID", "PLAYER_FIRST", "PLAYER_LAST", "TEAM_ID"]
  ROTATION_DIR_SORTED = sorted([season_label for season_label in os.listdir(f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{GAME_ROTATION_DIR}")])

  # Create season from data_config
  if LABEL_ARR is None: 
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys())]))

  else:
    LABEL_ARR_0 = [label[0] for label in LABEL_ARR]
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys()) if elem[:-4] in LABEL_ARR_0]))



####################### EXTRACT GAME ROTATION DATA #############################


def extract_gamelog_s3(is_upload: bool = True, save_array: bool = False):
    
  # Make sure gamelog diretory exists
  Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}/').mkdir(parents=True, exist_ok=True)
  
  
  print("Extracting Gamelog data...")
  
  df_gamelog_transformed_arr = []
  for gamelog_file in GAMELOG_FILES_SORTED:
      if not ".csv" in gamelog_file:
          continue
      df_gamelog = pd.read_csv(f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{GAMELOG_DIR}/{gamelog_file}")
      df_gamelog_transformed = etl.transform_gamelog_df(df_gamelog, False)
      df_gamelog_transformed = utils.sort_df_cols(df_gamelog_transformed)
      
      df_gamelog_transformed.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}/{gamelog_file}", index=False)
      
      # Append transformed data of current season into total data
      if save_array:
        df_gamelog_transformed_arr.append(df_gamelog_transformed)
        

    
  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR} s3://{BUCKET_TRANSFORMED_v1}/{GAMELOG_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log " )
    
    
  return df_gamelog_transformed_arr
      






####################### EXTRACT GAME ROTATION DATA #############################




def extract_game_rotation_s3(is_upload: bool = True, save_array: bool = False):
    
  # Make sure game rotation directory exists
  Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR}/').mkdir(parents=True, exist_ok=True)

  print("Extracting Game Rotation data...")
  
  df_game_rotation_arr = []
  for season_label in ROTATION_DIR_SORTED:

      df_game_rotation_season = None
      for file in os.listdir(f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{GAME_ROTATION_DIR}/{season_label}"):
          if not ".csv" in file:
              continue
          game_id = file[file.find("gameId")+6:-4]

          df_game_rotation = pd.read_csv(f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{GAME_ROTATION_DIR}/{season_label}/game_rotation_gameId{game_id}.csv")
          df_game_rotation = df_game_rotation[GAME_ROTATION_RELEVANT_COLS]
          df_game_rotation = df_game_rotation.drop_duplicates(ignore_index=True)
          df_game_rotation["GAME_ID"] = int(game_id)


          # change player_id column name for it to be compatible with other tables
          df_game_rotation = df_game_rotation.rename(columns={"PERSON_ID": "PLAYER_ID"})



          if df_game_rotation_season is None:
              df_game_rotation_season = df_game_rotation

          else:
              df_game_rotation_season = pd.concat([df_game_rotation_season, df_game_rotation], ignore_index=True)

      df_game_rotation_season.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR}/game_rotation_map{season_label}.csv", index=False)
      
      if save_array:
        df_game_rotation_arr.append(df_game_rotation_season)

  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAME_ROTATION_DIR} s3://{BUCKET_TRANSFORMED_v1}/{GAME_ROTATION_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log" )
    

  return df_game_rotation_arr
    
    

########################################## EXTRACT TEAM STATISTICS #############################################


lag_len_arr = ["08", "16", "32", "64", "180"]

relevant_cols_team_stat = ['TEAM_ID', 'GAME_DATE', 'GP', 'W_PCT', 'FG_PCT', 'FT_PCT', 'DAY_WITHIN_SEASON',
                           'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS','PLUS_MINUS',
                           'W_PCT_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK']


# Cols to add tage
cols_to_tag_team_stat = [ 'GP', 'W_PCT', 'FG_PCT', 'FT_PCT', 
                           'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS','PLUS_MINUS',
                           'W_PCT_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK']



def extract_team_stat_s3(is_upload: bool = True, save_array: bool = False):
    
  # Make sure team_statistics directory exists
  Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{TEAM_STAT_DIR}/').mkdir(parents=True, exist_ok=True)

  print("Extracting Team data...")
  
  # array of tables, each of which contain statistics only for one season type
  df_arr_team_stat = []

  # Iterate over years
  for year in YEAR_ARR:

      # Iterate over season types
      for seasonType in ["Pla", "Reg"]:

          # Get start and end date of season
          start_date, end_date = CONFIG_DICT["season_dates"][f"{year}_{seasonType}"]

          if seasonType == "Reg":
              _, end_date = CONFIG_DICT["season_dates"][f"{year}_Pla"]

          start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
          end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
          date_delta = end_date_obj - start_date_obj

          # This will contain all the statisc with all the lagging values
          df_team_stat_transformed = None

          # Iterate over dates within season
          for i in range(date_delta.days + 1):

              # Get current dates
              current_date_obj = start_date_obj + timedelta(days=i)
              current_date = current_date_obj.isoformat()[:10]

              # dataframe that contains data for one date and one lagging value
              df_team_stat = None
              for lag_len in lag_len_arr:

                  # get the file name that contains corresponding table
                  file_name = f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{TEAM_STAT_DIR}/{year}/team_stat_date{current_date}_lagged{lag_len}_seasonType{seasonType}.csv"

                  # at the moment, some of the ranges are uploaded yet
                  # so by pass if not exists
                  if not os.path.exists(file_name):
                      continue

                  # get file, enter corresponding date, extract only relevant columns
                  df_team_stat_lag = pd.read_csv(file_name)
                  df_team_stat_lag['GAME_DATE'] = current_date
                  df_team_stat_lag['DAY_WITHIN_SEASON'] = i
                  df_team_stat_lag = df_team_stat_lag[relevant_cols_team_stat]


                  # add the lagging tag to time-dependent columns, e.g. point average, 3pt percentage
                  col_map_team_stat = utils.add_prefix_arr(cols_to_tag_team_stat, f"lag{lag_len}_")
                  df_team_stat_lag = df_team_stat_lag.rename(columns = col_map_team_stat)

                  # Some of the tables are empty, add only non-empty ones
                  # otherwise, mergin gives undersired column orders
                  if df_team_stat_lag.shape[0] == 0:
                      continue

                  elif df_team_stat is None:
                      df_team_stat = df_team_stat_lag

                  else:
                      # Merge teams into one row where info about two team is contained
                      df_team_stat = df_team_stat.merge(df_team_stat_lag, on=["TEAM_ID", "GAME_DATE", "DAY_WITHIN_SEASON"])



              # Append transformed data of current season into total data
              if df_team_stat is None:
                      continue
              elif df_team_stat_transformed is None:
                  df_team_stat_transformed = df_team_stat
              else:
                  df_team_stat_transformed = pd.concat([df_team_stat_transformed, df_team_stat], ignore_index=True)


          df_team_stat_transformed.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{TEAM_STAT_DIR}/team_stat_transformed_season{year}_{seasonType}.csv", index=False)
          
          
          if save_array:
            df_arr_team_stat.append(df_team_stat_transformed)
          
          
  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{TEAM_STAT_DIR} s3://{BUCKET_TRANSFORMED_v1}/{TEAM_STAT_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log " )
    
    
  return df_arr_team_stat
    

    

#############################   EXTRACT PLAYER STATISTICS   ########################################
    
lag_len_arr = ["08", "16", "32", "64", "180"]

player_stat_relevant_col = ['PLAYER_ID','GAME_DATE', 'GP', 'W_PCT', 'FG_PCT', 'FG3M', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB',
                               'DREB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'DAY_WITHIN_SEASON',
                               'PLUS_MINUS', 'NBA_FANTASY_PTS', 'W_PCT_RANK', 'FGM_RANK', 'FG_PCT_RANK',
                               'FG3M_RANK', 'FG3_PCT_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK','AST_RANK',
                               'TOV_RANK', 'STL_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', ]



cols_to_rename_player_stat = [ 'GP', 'W_PCT', 'FG_PCT', 'FG3M', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB',
                               'DREB', 'AST', 'TOV', 'STL', 'BLK', 'PTS',
                               'PLUS_MINUS', 'NBA_FANTASY_PTS', 'W_PCT_RANK', 'FGM_RANK', 'FG_PCT_RANK',
                               'FG3M_RANK', 'FG3_PCT_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'AST_RANK',
                               'TOV_RANK', 'STL_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK']



def extract_player_stat_s3(is_upload: bool = True, save_array: bool = False):
    
  # Make sure player statistics directory exists
  Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{PLAYER_STAT_DIR}/').mkdir(parents=True, exist_ok=True)

  

  print("Extracting Player data...")

  # array of tables, each of which contain statistics only for one season type
  df_arr_player_stat = []

  # Iterate over years
  for year in YEAR_ARR:

      # Iterate over season types
      for seasonType in ["Pla", "Reg"]:

          # obtain start and end date for one season
          start_date, end_date = CONFIG_DICT["season_dates"][f"{year}_{seasonType}"]

          if seasonType == "Reg":
              _, end_date = CONFIG_DICT["season_dates"][f"{year}_Pla"]
          start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
          end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
          date_delta = end_date_obj - start_date_obj

          # This will contain all the statisc with all the lagging values
          df_player_stat_transformed = None

          # Iterate over dates within season
          for i in range(date_delta.days + 1):

              # Get current date
              current_date_obj = start_date_obj + timedelta(days=i)
              current_date = current_date_obj.isoformat()[:10]

              # dataframe that contains data for one date and one lagging value
              df_player_stat = None

              # Iterate over differnet lag values
              for lag_len in lag_len_arr:

                   # get the file name that contains corresponding table
                  file_name = f"{ROOT_DATA_DIR}/{RAW_DATA_DIR}/{PLAYER_STAT_DIR}/{year}/player_stat_date{current_date}_lagged{lag_len}_seasonType{seasonType}.csv"

                  # at the moment, some of the ranges are uploaded yet
                  # so by pass if not exists
                  if not os.path.exists(file_name):
                      continue

                  # get file, enter corresponding date, extract only relevant columns     
                  df_player_stat_lag = pd.read_csv(file_name)
                  df_player_stat_lag['GAME_DATE'] = current_date
                  df_player_stat_lag['DAY_WITHIN_SEASON'] = i
                  df_player_stat_lag = df_player_stat_lag[player_stat_relevant_col]

                  # add the lagging tag to time-dependent columns, e.g. point average, 3pt percentage
                  col_map_player_stat = utils.add_prefix_arr(cols_to_rename_player_stat, f"lag{lag_len}_")
                  df_player_stat_lag = df_player_stat_lag.rename(columns = col_map_player_stat)

                  # Some of the tables are empty, add only non-empty ones
                  # otherwise, merging gives undersired column orders
                  if df_player_stat_lag.shape[0] == 0:
                      continue
                  elif df_player_stat is None:
                      df_player_stat = df_player_stat_lag
                  else:
                      # Merge teams into one row where info about two team is contained
                      df_player_stat = df_player_stat.merge(df_player_stat_lag, on=["PLAYER_ID", "GAME_DATE", "DAY_WITHIN_SEASON"])


              # Append transformed data of current season into total data
              if df_player_stat is None:
                      continue
              elif df_player_stat_transformed is None:
                  df_player_stat_transformed = df_player_stat
              else:
                  df_player_stat_transformed = pd.concat([df_player_stat_transformed, df_player_stat], ignore_index=True)


          df_player_stat_transformed.to_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{PLAYER_STAT_DIR}/player_stat_transformed_season{year}_{seasonType}.csv", index=False)
          
          
          if save_array:
            df_arr_player_stat.append(df_player_stat_transformed)

  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{PLAYER_STAT_DIR} s3://{BUCKET_TRANSFORMED_v1}/{PLAYER_STAT_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" >> ~/entrypoint_log.log " )
          
  return df_arr_player_stat
          
          
          
          
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--yearArr", nargs="+", default=["2014"], help="List of years to process data of ")
  value = parser.parse_args()
  year_arr = [int(year) for year in value.yearArr]
  
  print(f"Chosen year_arr: {year_arr}")

  print("Running transform-v1 script...")
  

  extract_raw_data_from_s3(year_arr=year_arr)

  define_config_vars()
  
  extract_gamelog_s3(is_upload=True)

  extract_game_rotation_s3(is_upload=True)
  
  extract_team_stat_s3(is_upload=True)
  
  extract_player_stat_s3(is_upload=True)
  
  
  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}/*")
  
