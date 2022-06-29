#!/usr/bin/env python3
import os
import sys
import json
import requests

from datetime import datetime, timedelta
from pathlib import Path
  


# Local modules
# Local modules
import nba_ml_module.utils as utils
import nba_ml_module.etl as etl


REGULAR_SEASON_LABEL = "Regular Season"
PLAYOFFS_LABEL = "Playoffs"
ROOT_DATA_DIR = "."
PLAYER_STAT_DIR = "player_stat_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_CONFIG = "mertkurttutan-nba-ml-project-config"

DATA_CONFIG_FILE = "data_config.json"


def run_extract_player_data(date: str, season: str, seasonType_arr: list, proxy_config: dict):
  """Extracst player data from for one season and type of season
    Stores the data into player_stat_data directory"""

    
  # Statistical data from past 8, 16, 32, 64, 180 dates, can be thought of as a hyperparameter
  # 180 days is to capture entire season statistics
  lag_len_arr = [8, 16, 32, 64, 180]

  for lag_len in lag_len_arr:
    for seasonType in seasonType_arr:
      df_file_name = f'{ROOT_DATA_DIR}/{PLAYER_STAT_DIR}/{season}/player_stat_date{date}_lagged{lag_len:02d}_seasonType{seasonType[:3]}.csv'
      # Query data only if it does not exist
      if not os.path.exists(df_file_name):
        df_arr = etl.get_player_data_lagged(date, season, seasonType, lag_len, proxy_config)
        for idx, df in enumerate(df_arr):
          df.to_csv(df_file_name)
          #os.system
          
          
          
def run_extract_player_data_entire_year(season: str,  config_dict: dict):
  """Extracts data for entire year, regular season+ playoff games"""
  
  # Make sure parent directory exists
  Path(f'{ROOT_DATA_DIR}/{PLAYER_STAT_DIR}/{season}').mkdir(parents=True, exist_ok=True)

  # proxy-related variables
  proxy_arr = config_dict["proxy_arr"]
  proxy_num = len(proxy_arr)
  proxy_idx = 0
  
  
  ##########       Regular Season Part       ########
  # Date variables for regular season
  regular_start_date_str, regular_end_date_str = config_dict["season_dates"][f"{season}_{REGULAR_SEASON_LABEL[:3]}"]
  regular_start_date = datetime.strptime(regular_start_date_str, "%Y-%m-%d")
  regular_end_date = datetime.strptime(regular_end_date_str, "%Y-%m-%d")
  regular_delta = regular_end_date - regular_start_date   # returns timedelta
  
  
  # Query only regular season since playoffs did not start yet
  seasonType_arr = [REGULAR_SEASON_LABEL]
  
  for i in range(regular_delta.days + 1):
    
    # Keep trying different proxies until request is successful
    request_failed = True
    while request_failed:
      
      print(f"query: {i}")
      
      # time.sleep(random.randint(1, 3))
      # Change proxy at every trial, both success and failure
      proxy_idx = (proxy_idx + 1) % proxy_num

      try:
        # proxy setting
        proxy_str = proxy_arr[proxy_idx]
        proxy_config = {"http": f"http://{proxy_str}", "https": f"http://{proxy_str}"}
        
        # Current date
        date = regular_start_date + timedelta(days=i)
        
        # extract data
        run_extract_player_data(date=date.isoformat()[:10], season=season, seasonType_arr=seasonType_arr, proxy_config=proxy_config)
        
        # Update
        request_failed = False
        print("Succeeded in this proxy, going to next query...")

      # Handle only if there is problem with connecting to proxy
      # since this is the only exception type expected to happen
      except requests.exceptions.ProxyError:
        print("Did not succeed in this proxy, Trying another one...")
        
  ############     Playoffs Part    ############
  # Date variables for playoffs
  playoff_start_date_str, playoff_end_date_str = config_dict["season_dates"][f"{season}_{PLAYOFFS_LABEL[:3]}"]
  playoff_start_date = datetime.strptime(playoff_start_date_str, "%Y-%m-%d")
  playoff_end_date = datetime.strptime(playoff_end_date_str, "%Y-%m-%d")
  playoff_delta = playoff_end_date - playoff_start_date
  
  
  # Query both regular and playoff season
  seasonType_arr = [REGULAR_SEASON_LABEL, PLAYOFFS_LABEL]
  
  for i in range(playoff_delta.days + 1):
    
    # Keep trying different proxies until request is successful
    request_failed = True
    while request_failed:
      
      print(f"query: {i}")
      
      # time.sleep(random.randint(1, 3))
      # Change proxy at every trial, both success and failure
      proxy_idx = (proxy_idx + 1) % proxy_num

      try:
        # proxy setting
        proxy_str = proxy_arr[proxy_idx]
        proxy_config = {"http": f"http://{proxy_str}", "https": f"http://{proxy_str}"}
        
        # Current date
        date = playoff_start_date + timedelta(days=i)
        
        # extract data
        run_extract_player_data(date=date.isoformat()[:10], season=season, seasonType_arr=seasonType_arr, proxy_config=proxy_config)
        
        # Update
        request_failed = False
        print("Succeeded in this proxy, going to next query...")

      # Handle only if there is problem with connecting to proxy
      # since this is the only exception type expected to happen
      except requests.exceptions.ProxyError:
        print("Did not succeed in this proxy, Trying another one...")


if __name__ == "__main__":
  
  os.system(f"aws s3 cp s3://{BUCKET_CONFIG}/{DATA_CONFIG_FILE} ./{DATA_CONFIG_FILE}")
    
  with open(f"./{DATA_CONFIG_FILE}", "r") as read_content:
    config_dict = json.load(read_content)
    
  # Create season from data_config
  season_arr = sorted(set([elem[:-4] for elem in list(config_dict["season_dates"].keys())]))
    
  
  for season in season_arr:
    
    
    # Check 1st-level subdirectories (prefixes) 
    # If the current season is already uploaded to S3, skip this season
    if f"{PLAYER_STAT_DIR}/{season}" in utils.get_subdirs_s3(BUCKET_RAW):
      print(f"Directory {PLAYER_STAT_DIR}/{season} is already uploaded to S3")
      continue
      
    run_extract_player_data_entire_year(season, config_dict)
    
    # Upload directory to s3 bucket
    # Exclude jupyter checkpoint files
    os.system(f"aws s3 cp {ROOT_DATA_DIR}/{PLAYER_STAT_DIR} s3://{BUCKET_RAW}/{PLAYER_STAT_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" " )
    
    # After uploading this directory, delete it
    # Since its presence is not needed anymore
    os.system(f"rm -rf {ROOT_DATA_DIR}/{PLAYER_STAT_DIR}/*")