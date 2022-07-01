#!/usr/bin/env python3
import os
import sys
import argparse

import json
import requests
import pandas as pd
import numpy as np

from pathlib import Path


# Local modules
import nba_ml_module.utils as utils
import nba_ml_module.etl as etl



REGULAR_SEASON_LABEL = "Regular Season"
PLAYOFFS_LABEL = "Playoffs"
ROOT_DATA_DIR = "data"
GAMELOG_DIR = "gamelog_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_CONFIG = "mertkurttutan-nba-ml-project-config"

DATA_CONFIG_FILE = "data_config.json"


# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{GAMELOG_DIR}/').mkdir(parents=True, exist_ok=True)


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



def update_config(CONFIG_DICT: dict, is_upload: bool):

  for file in os.listdir(f"{ROOT_DATA_DIR}/{GAMELOG_DIR}"):
    if not "csv" in file:
        continue

    # name variables
    year_label = utils.get_year_label(file)
    seasonType = utils.get_season_type(file)
    season_label = f"{year_label}_{seasonType[:3]}"
    gamelog_file_name = f"{ROOT_DATA_DIR}/{GAMELOG_DIR}/{file}"
    
    # Extract data
    # Use converter to preserve leading zeros in GAME_ID column
    # If leading zeros are not preserved, parameters will be in invalid format
    df = pd.read_csv(gamelog_file_name, converters={'GAME_ID': lambda x: str(x)})
        
    # Store season dates into config file
    CONFIG_DICT["season_dates"][season_label] = (df["GAME_DATE"].min(), df["GAME_DATE"].max())
      
    # Store game ids, turn it into list since it is JSON-serializable
    CONFIG_DICT["season_game_ids"][season_label] = np.array(df["GAME_ID"].unique()).tolist()


  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "w") as outfile:
    json.dump(CONFIG_DICT, outfile)


  if is_upload:
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{DATA_CONFIG_FILE} s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} ")


def run_extract_gamelog(season: str, seasonType_arr: list, proxy_config: dict = {}):
  """Extracst player data from for one season and type of season
    Stores the data into player_stat_data directory"""



  for seasonType in seasonType_arr:
    df_file_name = f'{ROOT_DATA_DIR}/{GAMELOG_DIR}/gamelog_season{season}_seasonType{seasonType[:3]}.csv'
      
    # Query data only if it does not exist
    if not os.path.exists(df_file_name):
      df_arr = etl.jsonToDataFrame(etl.getRawGameLog(season=season, seasonType=seasonType, proxy_config=proxy_config))
      for idx, df in enumerate(df_arr):
        df.to_csv(df_file_name)
          
          
          
def run_extract_gamelog_data_entire_year(season: str,  config_dict: dict):
  """Extracts data for entire year, regular season+ playoff games"""
  
  # Make sure parent directory exists
  Path(f'{ROOT_DATA_DIR}/{GAMELOG_DIR}').mkdir(parents=True, exist_ok=True)

  # proxy-related variables
  proxy_arr = config_dict["proxy_arr"]
  proxy_num = len(proxy_arr)
  proxy_idx = 0
  
  
  ##########       Regular Season Part       ########
  
  # Query only regular season since playoffs did not start yet
  seasonType_arr = [REGULAR_SEASON_LABEL]
  
    
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
      run_extract_gamelog(season=season, seasonType_arr=seasonType_arr, proxy_config=proxy_config)
        
      # Update
      request_failed = False
      print("Succeeded in this proxy, going to next query...")

    # Handle only if there is problem with connecting to proxy
    # since this is the only exception type expected to happen
    except requests.exceptions.ProxyError:
      print("Did not succeed in this proxy, Trying another one...")
        
  ############     Playoffs Part    ############
  
  # Query both regular and playoff season
  seasonType_arr = [REGULAR_SEASON_LABEL, PLAYOFFS_LABEL]
  
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
      run_extract_gamelog(season=season, seasonType_arr=seasonType_arr, proxy_config=proxy_config)
        
      # Update
      request_failed = False
      print("Succeeded in this proxy, going to next query...")

    # Handle only if there is problem with connecting to proxy
    # since this is the only exception type expected to happen
    except requests.exceptions.ProxyError:
      print("Did not succeed in this proxy, Trying another one...")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--year-arr", nargs="+", default=["2014"], help="List of years to process data of ")
  #parser.add_argument('--is-upload', action='store_true')
  value = parser.parse_args()

  year_arr = [int(year) for year in value.year_arr]


  extract_data_from_s3(year_arr=year_arr)

  define_config_vars()

  
  for year in year_arr:
      
    season = utils.get_season_str(year)
    run_extract_gamelog_data_entire_year(season, CONFIG_DICT)
      
      
  update_config(CONFIG_DICT=CONFIG_DICT, is_upload=True)


    
  # Upload directory to s3 bucket
  # Exclude jupyter checkpoint files
  os.system(f"aws s3 cp {ROOT_DATA_DIR}/{GAMELOG_DIR} s3://{BUCKET_RAW}/{GAMELOG_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" " )
    
  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}")