import awswrangler as wr
import numpy as np
import pandas as pd
from pathlib import Path
import boto3
import os
import json
import botocore
import argparse



import nba_ml_module.utils as utils

TEAM_STAT_DIR = "team_stat_data"
GAMELOG_DIR = "gamelog_data"

ROOT_DATA_DIR = "./data"
TRANSFORMED_DATA_DIR_v1 = "transformed_data_v1"
TRANSFORMED_DATA_DIR_v2 = "transformed_data_v2"


BUCKET_TRANSFORMED_v1 = "mert-kurttutan-nba-ml-project-transformed-data-v1"
BUCKET_TRANSFORMED_v2 = "mert-kurttutan-nba-ml-project-transformed-data-v2"
BUCKET_CONFIG = "mert-kurttutan-nba-ml-files/config"

DATA_CONFIG_FILE = "data_config.json"

# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/').mkdir(parents=True, exist_ok=True)
Path(f'{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v2}/').mkdir(parents=True, exist_ok=True)

def year_to_label(year: list):

  return f"{year}-{str(year+1)[-2:]}", f"{year}-{str(year+1)}"

def weighted(x, cols, w="lag08_GP"):
    return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)

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
  global CONFIG_DICT, GAMELOG_FILES_SORTED, TEAM_STAT_FILES
  global YEAR_ARR, DF_GAMELOG_ARR,  DF_TEAM_STAT_ARR
    
  with open(f"{ROOT_DATA_DIR}/{DATA_CONFIG_FILE}", "r") as read_content:
    CONFIG_DICT = json.load(read_content)

  # csv files to be iterated over
  GAMELOG_FILES_SORTED = sorted([gamelog_file for gamelog_file in os.listdir(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}") if ".csv" in gamelog_file])
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
    df_gamelog = pd.read_csv(f"{ROOT_DATA_DIR}/{TRANSFORMED_DATA_DIR_v1}/{GAMELOG_DIR}/{gamelog_file}")
    
    df_gamelog["SEASON"] = int(gamelog_file[14:18]) - 2007
    
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


if __name__ == "__main__":

  boto3_ses = boto3.Session()

  year_today = 2013

  parser = argparse.ArgumentParser(description='Arguments for mlflow python script')
  parser.add_argument("--year-arr", nargs="+", default=["2014"], help="List of years to process data of ")
  value = parser.parse_args()
  year_arr = [int(year) for year in value.year_arr]

  if year_arr == [-1]:
    print("Current mode is chosen....")
    year_arr = [year_today]

  extract_data_from_s3(year_arr)
  define_config_vars()

  # write gamelog into dynamo db
  total_df_gamelog_stat = pd.concat(DF_GAMELOG_ARR, ignore_index=True).sort_values(["SEASON", "IS_REGULAR", "GAME_DATE"], ascending=[True, False, True])

  selected_cols = ["SEASON", "IS_REGULAR", "GAME_ID", "GAME_DATE", "team1_W", "team1_W_cum", "team1_TEAM_ID", "team2_W", "team2_W_cum", "team2_TEAM_ID"]
  total_df_gamelog_stat = utils.float_to_decimal_df(total_df_gamelog_stat[selected_cols])

  nba_gamelog_table_name = "gamelog-table-vtrial"
  wr.dynamodb.put_df(df=total_df_gamelog_stat, table_name=nba_gamelog_table_name)#, boto3_session=boto3_ses)





  # write team stat into dynamo db
  total_df_team_stat = pd.concat(DF_TEAM_STAT_ARR, ignore_index=True)

  target_cols = list(total_df_team_stat.columns)
  target_cols.remove("GAME_DATE")
  target_cols.remove("TEAM_ID")

  total_df_team_stat = total_df_team_stat.groupby(["GAME_DATE", "TEAM_ID"]).apply(weighted, target_cols).reset_index() 
  total_df_team_stat["TEAM_ID"] = total_df_team_stat["TEAM_ID"].astype(int)

  total_df_team_stat = utils.float_to_decimal_df(total_df_team_stat)

  nba_team_stat_table_name = "team-stat-table-vtrial"
  wr.dynamodb.put_df(df=total_df_team_stat, table_name=nba_team_stat_table_name, boto3_session=boto3)