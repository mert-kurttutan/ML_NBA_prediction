from ..utils import float_to_dec, sort_df_cols, add_column_prefix, decimal_to_float_df
from boto3.dynamodb.types import TypeDeserializer
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import botocore



def _deserialize_value(value: Any) -> Any:
    if not pd.isna(value):
        return TypeDeserializer().deserialize(value)
    return value



def nba_match_query(team_ids: List[str], game_date: str, nba_table_name: str, dynamodb_client: botocore.client):


  query = f'SELECT * FROM \"{nba_table_name}\" WHERE ((team1_TEAM_ID = ? AND team2_TEAM_ID = ?) OR (team1_TEAM_ID = ? AND team2_TEAM_ID = ?)) AND GAME_DATE = ? '

  response = dynamodb_client.execute_statement(Statement=query, 
                                Parameters =[{'N': team_ids[0]},{'N': team_ids[1]}, {'N': team_ids[1]},{'N': team_ids[0]}, {'S': game_date}])

  if response["ResponseMetadata"]['HTTPStatusCode'] != 200:
    return -1
    
  else:
    data_dict_arr = response["Items"]

    return data_dict_arr




def query_response_to_df(response: List[Dict]):


  data_dict_c = response[0].copy()
  for key in data_dict_c:
    col_arr = []
    for data_dict in response:
      value = data_dict[key]
      col_arr.append(_deserialize_value(value))
  
    data_dict_c[key] = col_arr

  df = sort_df_cols(pd.DataFrame(data_dict_c))

  return df


def nba_cum_stat_query(team_ids: List[str], game_date: str, nba_table_name: str, dynamodb_client: botocore.client):


  query = f'SELECT * FROM \"{nba_table_name}\" WHERE ((team1_TEAM_ID = ? AND team2_TEAM_ID = ?) OR (team1_TEAM_ID = ? AND team2_TEAM_ID = ?)) AND GAME_DATE < ? '

  response = dynamodb_client.execute_statement(Statement=query, 
                                Parameters =[{'N': team_ids[0]},{'N': team_ids[1]}, {'N': team_ids[1]},{'N': team_ids[0]}, {'S': game_date}])

  if response["ResponseMetadata"]['HTTPStatusCode'] != 200:
    return -1
    
  else:
    data_dict_arr = response["Items"]

    return data_dict_arr



def get_cum_stat(team1_id: int, team2_id:int , game_date: str, nba_table_name: str, dynamodb_client: botocore.client):
    
    data_dict_arr = nba_cum_stat_query([str(team1_id), str(team2_id)], game_date, nba_table_name, dynamodb_client)
    past_matches = query_response_to_df(data_dict_arr)
    past_matches = past_matches.sort_values(by=["GAME_DATE"])
    
    if past_matches.shape[0] < 1:
        current_cum_stat = pd.DataFrame({
                                         "GAME_DATE": game_date,
                                         "team1_W_cum": 0,
                                         "team2_W_cum": 0,
                                         "team1_TEAM_ID": team1_id,
                                         "team2_TEAM_ID": team2_id,
                                        })
    else:
        
        current_cum_stat = past_matches.tail(1).copy().reset_index(drop=True)
        current_cum_stat.loc[:, "team1_W_cum"] += current_cum_stat.loc[:, "team1_W"]
        current_cum_stat.loc[:, "team2_W_cum"] += current_cum_stat.loc[:, "team2_W"]
        current_cum_stat.loc[:, "GAME_DATE"] = game_date
        current_cum_stat.drop(["team1_W", "team2_W", "GAME_ID"], axis=1,inplace=True)
        
    return current_cum_stat

    

def nba_team_stat_query(team_id: str, game_date: str, nba_table_name: str, dynamodb_client: botocore.client):


  query = f'SELECT * FROM \"{nba_table_name}\" WHERE TEAM_ID = ? AND GAME_DATE = ? '

  response = dynamodb_client.execute_statement(Statement=query,
                                Parameters =[{'N': team_id}, {'S': game_date}])

  if response["ResponseMetadata"]['HTTPStatusCode'] != 200:
    return -1
    
  else:
    data_dict_arr = response["Items"]

    return data_dict_arr



def get_X_pred(team1_id: int, team2_id: int, game_date: str, team1_IS_HOME: bool, IS_REGULAR: bool, SEASON: int, nba_team_table_name: str, nba_gamelog_table_name: str, dynamodb_client: botocore.client) -> pd.DataFrame:

    cum_stat = get_cum_stat(team1_id, team2_id, game_date, nba_gamelog_table_name, dynamodb_client)

    data_dict_arr = nba_team_stat_query(team_id=str(team1_id), game_date=game_date, nba_table_name=nba_team_table_name, dynamodb_client=dynamodb_client)
    team1_stat = query_response_to_df(data_dict_arr).drop(["GAME_DATE", "TEAM_ID"], axis=1)
    add_column_prefix(df=team1_stat, prefix="team1_", inplace=True)
    team1_stat.rename(columns={"team1_DAY_WITHIN_SEASON": "DAY_WITHIN_SEASON"}, inplace=True)


    data_dict_arr = nba_team_stat_query(team_id=str(team2_id), game_date=game_date, nba_table_name=nba_team_table_name, dynamodb_client=dynamodb_client)
    team2_stat = query_response_to_df(data_dict_arr).drop(["GAME_DATE", "TEAM_ID", "DAY_WITHIN_SEASON"], axis=1)
    add_column_prefix(df=team2_stat, prefix="team2_", inplace=True)



    X_pred = pd.concat([cum_stat, team1_stat, team2_stat], axis=1)

    X_pred["team1_IS_HOME"] = float(team1_IS_HOME)
    X_pred["IS_REGULAR"] = float(IS_REGULAR)
    X_pred["SEASON"] = SEASON
    X_pred = sort_df_cols(X_pred)
    
    decimal_to_float_df(X_pred)
    
    return X_pred
