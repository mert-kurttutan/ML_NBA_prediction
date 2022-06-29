#!/usr/bin/env python3
import os
import sys
import json
import requests
import pandas as pd

from pathlib import Path


# Local modules
import nba_ml_package.utils as utils
import nba_ml_package.etl as etl



REGULAR_SEASON_LABEL = "Regular Season"
PLAYOFFS_LABEL = "Playoffs"
ROOT_DATA_DIR = "data"
GAME_ROTATION_DIR = "game_rotation_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_CONFIG = "mertkurttutan-nba-ml-project-config"
DATA_CONFIG_FILE = "data_config.json"


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
    
  os.system(f"aws s3 cp s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} {ROOT_DATA_DIR}/{DATA_CONFIG_FILE}")
    
  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    config_dict = json.load(read_content)
  
    
    
  config_dict["uploaded_dir"] = []
    
  for season_label in sorted([elem for elem in list(config_dict["season_game_ids"].keys())]):
    
    # Check 1st-level subdirectories (prefixes) 
    # If the current season is already uploaded to S3, skip this season
    if f"{GAME_ROTATION_DIR}/{season_label}" in utils.get_subdirs_s3(BUCKET_RAW):
      print(f"Directory {GAME_ROTATION_DIR}/{season_label} is already uploaded to S3")
      continue
    
    game_id_arr = config_dict["season_game_ids"][season_label]
    print(f"Query for season: {season_label}")
    run_extract_game_rotation_array(game_id_arr=game_id_arr, config_dict=config_dict, season_label=season_label)
    
    # Upload directory to s3 bucket
    # Exclude jupyter checkpoint files
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{GAME_ROTATION_DIR} s3://{BUCKET_RAW}/{GAME_ROTATION_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" " )
    
    # After uploading this directory, delete it
    # Since its presence is not needed anymore
    os.system(f"rm -rf {ROOT_DATA_DIR}/{GAME_ROTATION_DIR}/*")
