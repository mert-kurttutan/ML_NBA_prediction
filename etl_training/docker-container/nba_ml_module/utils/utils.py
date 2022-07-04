import boto3
import pandas as pd
from datetime import datetime, timedelta
from typing import Union
from decimal import Decimal


def get_season_str(start_yr: int) -> str:
  
  end_yr = start_yr + 1
  start_yr_str = str(start_yr)
  end_yr_str = str(end_yr)
  
  
  return f"{start_yr_str}-{end_yr_str[-2:]}"

def add_column_prefix(df: pd.DataFrame, prefix: str, inplace: bool)-> pd.DataFrame:

    col_name_map = {}
    for col in df.columns:
        col_name_map[col] = prefix + col
    if inplace:
        df.rename(columns=col_name_map, inplace=inplace)
    else:
        return df.rename(columns=col_name_map, inplace=inplace)



def get_season_type(gamelog_file_name: str):
    if "seasonTypeReg" in gamelog_file_name:
        return "Regular Season"
    elif "seasonTypePla" in gamelog_file_name:
        return "Playoffs"
    else:
        raise ValueError("Wrong file name format!")
        
        
        
        
        
def get_year_label(gamelog_file_name: str):
    start_idx = len("gamelog_season")
    end_idx = gamelog_file_name[start_idx:].find("_")
    
    return gamelog_file_name[start_idx: start_idx+end_idx]
  
  
  
  


def get_subdirs_s3(BUCKET: str):  
  """Returns subdirs (or more accurately sub-prefixes since s3 is not a hierarchical storage system) from s3 bucket"""

  s3_resource = boto3.resource('s3')
  folders = set()

  # Find paths of all non-empty objects (to exclude zero-length 'folder' objects)
  for object in s3_resource.Bucket(BUCKET).objects.all():
      if object.size > 0 and '/' in object.key:
          folders.add(object.key[:object.key.rfind('/')])

  return list(folders)


def sort_df_cols(df: pd.DataFrame) -> pd.DataFrame:
  df = df.reindex(sorted(df.columns), axis=1)
  
  return df



def add_suffix_arr(col_arr: list, suffix: str):
    return {col:col+suffix for col in col_arr}

def add_prefix_arr(col_arr: list, prefix: str):
    return {col:prefix+col for col in col_arr}

def is_home(x: str):
        return not "@" in x
  
  
def str_to_day(start_date:str, date: str):
    current_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    return (current_date - start_date).days



def float_to_dec(x: Union[float, int]) -> Decimal:
  """Convert from int,float to decimal to make data process compatible with dynamodb
  Dynamodb does not support float"""
  return Decimal(str(x))

def float_to_decimal_df(df: pd.DataFrame) -> pd.DataFrame:

    for col_type,col in zip(df.dtypes, df.columns):
        if col_type == "float64":
            df[col] = df[col].apply(float_to_dec)
            
    return df


def decimal_to_float_df(df: pd.DataFrame) -> pd.DataFrame:

    for col_type,col in zip(df.dtypes, df.columns):
        if isinstance(df[col].values[0], Decimal):
            df[col] = df[col].astype(float)
            
    return df
  
