#!/usr/bin/env python3
import os
import sys
import argparse
import json
import requests
from typing import List

from datetime import datetime, timedelta
from pathlib import Path
  



# Local modules
import nba_ml_module.utils as utils
import nba_ml_module.etl as etl


REGULAR_SEASON_LABEL = "Regular Season"
PLAYOFFS_LABEL = "Playoffs"
ROOT_DATA_DIR = "data"
TEAM_STAT_DIR = "team_stat_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_CONFIG = "mertkurttutan-nba-ml-project-config"

DATA_CONFIG_FILE = "data_config.json"


# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{TEAM_STAT_DIR}/').mkdir(parents=True, exist_ok=True)


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


def run_extract_team_data(date: str, season: str, seasonType_arr: list, proxy_config: dict):
  """Extracst team data from for one season and type of season
    Stores the data into team_stat_data directory"""

    
  # Statistical data from past 8, 16, 32, 64, 180 dates, can be thought of as a hyperparameter
  # 180 days is to capture entire season statistics
  lag_len_arr = [8, 16, 32, 64, 180]

  for lag_len in lag_len_arr:
    for seasonType in seasonType_arr:
      df_file_name = f'{ROOT_DATA_DIR}/{TEAM_STAT_DIR}/{season}/team_stat_date{date}_lagged{lag_len:02d}_seasonType{seasonType[:3]}.csv'
      # Query data only if it does not exist
      if not os.path.exists(df_file_name):
        df_arr = etl.get_team_data_lagged(date, season, seasonType, lag_len, proxy_config)
        for idx, df in enumerate(df_arr):
          df.to_csv(df_file_name)
          
          

def run_extract_team_data_array(seasonType_arr: List[str],  season: str, start_date_str: str, end_date_str: str, config_dict: dict):
  """Extracts data for entire year, regular season+ playoff games"""
  
  # Make sure parent directory exists
  Path(f'{ROOT_DATA_DIR}/{TEAM_STAT_DIR}/{season}').mkdir(parents=True, exist_ok=True)

  # proxy-related variables
  proxy_arr = config_dict["proxy_arr"]
  proxy_num = len(proxy_arr)
  proxy_idx = 0
  

  start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
  end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
  regular_delta = end_date - start_date   # returns timedelta
  
  
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
        date = start_date + timedelta(days=i)
        
        # extract data
        run_extract_team_data(date=date.isoformat()[:10], season=season, seasonType_arr=seasonType_arr, proxy_config=proxy_config)
        
        # Update
        request_failed = False
        print("Succeeded in this proxy, going to next query...")

      # Handle only if there is problem with connecting to proxy
      # since this is the only exception type expected to happen
      except requests.exceptions.ProxyError:
        print("Did not succeed in this proxy, Trying another one...")   
          
def run_extract_team_data_entire_year(season: str,  config_dict: dict):
  """Extracts data for entire year, regular season+ playoff games"""
  
  # Make sure parent directory exists
  Path(f'{ROOT_DATA_DIR}/{TEAM_STAT_DIR}/{season}').mkdir(parents=True, exist_ok=True)

  
  ##########       Regular Season Part       ########

  # Query only regular season since playoffs did not start yet
  seasonType_arr = [REGULAR_SEASON_LABEL]

  # Date variables for regular season
  regular_start_date_str, regular_end_date_str = config_dict["season_dates"][f"{season}_{REGULAR_SEASON_LABEL[:3]}"]

  run_extract_team_data_array(seasonType_arr=seasonType_arr, season=season, start_date_str=regular_start_date_str, end_date_str=regular_end_date_str, config_dict=config_dict)
  
        
  ############     Playoffs Part    ############

  # Query both regular and playoff season
  seasonType_arr = [REGULAR_SEASON_LABEL, PLAYOFFS_LABEL]
  
  # Date variables for playoffs
  playoff_start_date_str, playoff_end_date_str = config_dict["season_dates"][f"{season}_{PLAYOFFS_LABEL[:3]}"]

  run_extract_team_data_array(seasonType_arr=seasonType_arr, season=season, start_date_str=playoff_start_date_str, end_date_str=playoff_end_date_str, config_dict=config_dict)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--year-arr", nargs="+", default=["-1"], help="List of years to process data of ")
  #parser.add_argument('--is-upload', action='store_true')
  value = parser.parse_args()

  year_arr = [int(year) for year in value.year_arr]


  extract_data_from_s3(year_arr=year_arr)

  define_config_vars()

  if year_arr == [-1]:
    print("Current mode is chosen!!!!")
    date_today = "2016-02-02"
    year = 2016
    season = utils.get_season_str(2015)
    run_extract_team_data_array(seasonType_arr=[REGULAR_SEASON_LABEL],  season=season, start_date_str=date_today, end_date_str=date_today, config_dict=CONFIG_DICT)




  else:
    print("yearly batch mode version")
    

    for year in year_arr:
      for season_label in sorted([elem for elem in list(CONFIG_DICT["season_game_ids"].keys())]):
        season = utils.get_season_str(year)
        if season in season_label:
          print(season_label)
        
          run_extract_team_data_entire_year(season, CONFIG_DICT)
    

    
  # Upload directory to s3 bucket
  # Exclude jupyter checkpoint files
  os.system(f"aws s3 cp {ROOT_DATA_DIR}/{TEAM_STAT_DIR} s3://{BUCKET_RAW}/{TEAM_STAT_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" " )
  
  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}/{TEAM_STAT_DIR}")
