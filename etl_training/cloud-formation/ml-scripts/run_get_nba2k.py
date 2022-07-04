#!/usr/bin/env python3
import os
import sys
import json
import requests
import pandas as pd

from pathlib import Path
  

# Local modules
import nba_ml_module.etl as etl


NBA2K_ENDPOINT = "https://hoopshype.com/nba2k"
ROOT_DATA_DIR = "."
NBA2K_DIR = "nba2k_data"

BUCKET_RAW = "mert-kurttutan-nba-ml-project-raw-data"
BUCKET_CONFIG = "mert-kurttutan-nba-ml-files/config"

DATA_CONFIG_FILE = "data_config.json"



def run_extract_nba2k(season: str, proxy_config: dict = {}):
  """Extracst NBA2K player data from for one season and type of season
    Stores the data into player_stat_data directory"""

  df_file_name = f'{ROOT_DATA_DIR}/{NBA2K_DIR}/nba2k_season{season}.csv'
      
  # Query data only if it does not exist
  if not os.path.exists(df_file_name):
    
    end_point = f"{NBA2K_ENDPOINT}/{season}"
    df = etl.get_nba2k_rating_df(url=end_point, proxy_config=proxy_config)
    print(end_point)
    df.to_csv(df_file_name)
          
          
          
def run_extract_nba2k_array(season_arr: list, config_dict: dict):
  """Extracts data for entire year, regular season+ playoff games"""
  
  # Make sure parent directory exists
  Path(f'{ROOT_DATA_DIR}/{NBA2K_DIR}').mkdir(parents=True, exist_ok=True)

  # proxy-related variables
  proxy_arr = config_dict["proxy_arr"]
  proxy_num = len(proxy_arr)
  proxy_idx = 0
  
    
  for idx,season in enumerate(season_arr):
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
        run_extract_nba2k(season=season)#, proxy_config=proxy_config)

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
    

  season_arr = ["2009-2010", "2010-2011", "2011-2012", "2012-2013", "2013-2014", "2014-2015",
               "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021",
               "2021-2022"]
  run_extract_nba2k_array(season_arr, config_dict)
    
  # Upload directory to s3 bucket
  os.system(f"aws s3 cp {ROOT_DATA_DIR}/{NBA2K_DIR} s3://{BUCKET_RAW}/{NBA2K_DIR}/ --recursive --exclude \".ipynb_checkpoints/*\" " )

  # After uploading this directory, delete it
  # Since its presence is not needed anymore
  os.system(f"rm -rf {ROOT_DATA_DIR}/{NBA2K_DIR}/*")