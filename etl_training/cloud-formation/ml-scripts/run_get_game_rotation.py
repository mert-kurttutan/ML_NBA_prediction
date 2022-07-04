#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import pandas as pd

from pathlib import Path


# Local modules
import nba_ml_module.utils as utils
import nba_ml_module.etl as etl



REGULAR_SEASON_LABEL = "Regular Season"
PLAYOFFS_LABEL = "Playoffs"
ROOT_DATA_DIR = "data"
GAME_ROTATION_DIR = "game_rotation_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_CONFIG = "mert-kurttutan-nba-ml-files/config"

DATA_CONFIG_FILE = "data_config.json"


# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{GAME_ROTATION_DIR}/').mkdir(parents=True, exist_ok=True)


def year_to_label(year: list):

  return f"{year}-{str(year+1)[-2:]}", f"{year}-{str(year+1)}"

def extract_data_from_s3(year_arr=None):
  global LABEL_ARR
  LABEL_ARR = None

  LABEL_ARR = [year_to_label(year) for year in year_arr]
  
  # Copy config file from s3 bucket
  os.system(f"aws s3 cp s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} >> ~/entrypoint_log.log ")




def define_config_vars():
  print("Defining config vars...")
  global CONFIG_DICT
  global YEAR_ARR


  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    CONFIG_DICT = json.load(read_content)


  # Create season from data_config
  
  if LABEL_ARR is None: 
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys())]))
  
  else:
    LABEL_ARR_0 = [label[0] for label in LABEL_ARR]
    YEAR_ARR = sorted(set([elem[:-4] for elem in list(CONFIG_DICT["season_dates"].keys()) if elem[:-4] in LABEL_ARR_0]))


  print("Defining important dataframes...")




def run_extract_game_rotation(gameId: str, proxy_config: dict, season_label: str):
  """Extracts player data from for one season and type of season
    Stores the data into player_stat_data directory"""

  df_file_name = f'{ROOT_DATA_DIR}/{GAME_ROTATION_DIR}/{season_label}/game_rotation_gameId{gameId}.csv'

  # Query data only if it does not exist
  if not os.path.exists(df_file_name):
    df_arr = etl.jsonToDataFrame(etl.getRawGameRotation(gameId=gameId, proxy_config=proxy_config))
    df = pd.concat(df_arr)
    df.to_csv(df_file_name)
          


          
          
def run_extract_game_rotation_array(game_id_arr: list, config_dict: dict, season_label: str):
  """Extracts data for entire year, regular season+ playoff games"""
  
  # Make sure parent directory exists
  Path(f'{ROOT_DATA_DIR}/{GAME_ROTATION_DIR}/{season_label}/').mkdir(parents=True, exist_ok=True)

  # proxy-related variables
  proxy_arr = config_dict["proxy_arr"]
  proxy_num = len(proxy_arr)
  proxy_idx = 0
  
    
  for idx,game_id in enumerate(game_id_arr):
    print(f"Query: {idx}")
    # Keep trying different proxies until request is successful
    request_failed = True
    while request_failed:
      # time.sleep(random.randint(1, 3))
      # Change proxy at every trial, both success and failure
      proxy_idx = (proxy_idx + 1) % proxy_num

      try:
        # proxy setting
        proxy_str = proxy_arr[proxy_idx]
        proxy_config = {"http": f"http://{proxy_str}", "https": f"http://{proxy_str}"}

        # extract data
        run_extract_game_rotation(gameId=game_id, proxy_config=proxy_config, season_label=season_label)

        # Update
        request_failed = False
        print("Succeeded in this proxy, going to next query...")

      # Handle only if there is problem with connecting to proxy
      # since this is the only exception type expected to happen
      except requests.exceptions.ProxyError:
        print("Did not succeed in this proxy, Trying another one...")


   



if __name__ == "__main__":

  year_today = 2013
    
  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--year-arr", nargs="+", default=["2014"], help="List of years to process data of ")
  value = parser.parse_args()

  year_arr = [int(year) for year in value.year_arr]

  if year_arr == [-1]:
    print("Current mode is chosen...")
    year_arr = [year_today]
  extract_data_from_s3(year_arr=year_arr)

  define_config_vars()
  
    
    
  for year in year_arr:
    for season_label in sorted([elem for elem in list(CONFIG_DICT["season_game_ids"].keys())]):
      season = utils.get_season_str(year)
      if season in season_label:
        print(season_label)
    
        game_id_arr = CONFIG_DICT["season_game_ids"][season_label]
        print(f"Query for season: {season_label}")
        run_extract_game_rotation_array(game_id_arr=game_id_arr, config_dict=CONFIG_DICT, season_label=season_label)
    
  # Upload directory to s3 bucket
  # Exclude jupyter checkpoint files
  os.system(f"aws s3 cp {ROOT_DATA_DIR}/{GAME_ROTATION_DIR} s3://{BUCKET_RAW}/{GAME_ROTATION_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" " )
  
  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}/{GAME_ROTATION_DIR}/*")
