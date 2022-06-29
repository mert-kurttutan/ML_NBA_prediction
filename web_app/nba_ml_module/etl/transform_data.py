#!/usr/bin/env python3
import pandas as pd
import numpy as np

from functools import partial

from ..utils import is_home, add_prefix_arr, str_to_day, sort_df_cols



def transform_gamelog_df(df_gamelog: pd.DataFrame, is_home_order: bool = True) -> pd.DataFrame:
    cols_to_drop_0 = ["Unnamed: 0","VIDEO_AVAILABLE", "MIN", "TEAM_ABBREVIATION", "SEASON_ID"]
    
    # Eliminate columns that will lead to data leakage
    # namely any data that resulted in during the game, e.g. percentage of 3 point shots, free throws, etc
    # other than prediction labels
    data_leakage_cols = ['FGM', 'FGA', 'FG_PCT','FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                           'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',]
    
    df_gamelog_c = df_gamelog.drop(cols_to_drop_0+data_leakage_cols, axis=1)
    
    
    # Boolean column to indicate whether the team is home team or not
    df_gamelog_c["IS_HOME"] = df_gamelog_c["MATCHUP"].apply(is_home)

    # Unnecessary column
    df_gamelog_c.drop(["MATCHUP"], axis=1, inplace=True)
    
    if is_home_order:
        sort_col = "IS_HOME"
    else:
        sort_col = "TEAM_ID"
    
    df_gamelog_c.sort_values(by=["GAME_DATE", "GAME_ID", sort_col], inplace=True)
    
    start_date = df_gamelog_c["GAME_DATE"].min()
    # Relative point of match within season (relative to start of the season)
    df_gamelog_c["DAY_WITHIN_SEASON"] = df_gamelog_c["GAME_DATE"].apply(partial(str_to_day, start_date)) 
    
    # Convenient column order
    df_gamelog_c = df_gamelog_c[['GAME_ID', 'GAME_DATE', 'DAY_WITHIN_SEASON', 'TEAM_ID', 'TEAM_NAME', 'WL', 'PTS', 'PLUS_MINUS', "IS_HOME"]]
    
    df_gamelog_c["W"] = (df_gamelog_c["WL"]=="W").astype("int32")
    df_gamelog_c["L"] = (df_gamelog_c["WL"]=="L").astype("int32")
    df_gamelog_c.drop(columns=["WL"], inplace=True)
    
    df1_gamelog = (df_gamelog_c.iloc[::2]).reset_index(drop=True)
    df2_gamelog = (df_gamelog_c.iloc[1::2]).reset_index(drop=True)
 
    
    
    # tag team number to column names
    # first team that appears is the visitor team
    cols = ['TEAM_ID', 'TEAM_NAME', 'WL', "PTS", "PLUS_MINUS", "W", "L", "IS_HOME"]
    col_map1 = add_prefix_arr(col_arr=cols, prefix="team1_")
    col_map2 = add_prefix_arr(col_arr=cols, prefix="team2_")
    df1_gamelog = df1_gamelog.rename(columns = col_map1)
    df2_gamelog = df2_gamelog.rename(columns = col_map2)
    
    df2_gamelog.drop(columns=["GAME_ID", "GAME_DATE", "DAY_WITHIN_SEASON"], inplace=True)
    
    # Merge teams into one row where info about two team is contained
    df3_gamelog = pd.concat([df1_gamelog, df2_gamelog], axis=1)#.T.drop_duplicates().T
    
    
    # Add win/loss for series between two teams
    team_id1_arr = list(df3_gamelog["team1_TEAM_ID"].unique())
    team_id2_arr = list(df3_gamelog["team2_TEAM_ID"].unique())
    #print(df2_gamelog.columns, col_map2)
    
    for team_id1 in team_id1_arr:
        for team_id2 in team_id2_arr:
            matchup_cond = (df3_gamelog["team1_TEAM_ID"] == team_id1) & (df3_gamelog["team2_TEAM_ID"] == team_id2)
            
            #  cumulative W/L for team 1
            df3_gamelog.loc[matchup_cond, "team1_W_cum"] = df3_gamelog.loc[matchup_cond, "team1_W"].rolling(min_periods=1, window=2000).sum().shift(periods=1, fill_value=0)
            df3_gamelog.loc[matchup_cond, "team1_L_cum"] = df3_gamelog.loc[matchup_cond, "team1_L"].rolling(min_periods=1, window=2000).sum().shift(periods=1, fill_value=0)
            
            # cumulative W/L for team 2
            df3_gamelog.loc[matchup_cond, "team2_W_cum"] = df3_gamelog.loc[matchup_cond, "team2_W"].rolling(min_periods=1, window=2000).sum().shift(periods=1, fill_value=0)
            df3_gamelog.loc[matchup_cond, "team2_L_cum"] = df3_gamelog.loc[matchup_cond, "team2_L"].rolling(min_periods=1, window=2000).sum().shift(periods=1, fill_value=0)

    
    return df3_gamelog

  
  
  
  
def fill_missing_team_stat_regular(current_year_stat: pd.DataFrame, prev_year_stat: pd.DataFrame, current_gamelog_data: pd.DataFrame, team_idx: int = 1):
    
    
    current_year_stat_c = current_year_stat.copy()
    prev_year_stat_c = prev_year_stat.copy()
    current_gamelog_data_c = current_gamelog_data.copy()

    current_gamelog_data_c = current_gamelog_data_c.rename(columns={f"team{team_idx}_TEAM_ID": "TEAM_ID"})
    gamelog_team_combined = pd.merge(current_gamelog_data_c, current_year_stat_c, on=["GAME_DATE", "DAY_WITHIN_SEASON", "TEAM_ID"], how="left")
    
    missing_stat_idx = gamelog_team_combined["lag08_W_PCT"].isna()

    missing_stat = gamelog_team_combined.loc[missing_stat_idx, [ "GAME_DATE", "DAY_WITHIN_SEASON", "TEAM_ID"]]
    
    
    current_year_stat_c["pair_idx"] = 1
    prev_year_stat_c["pair_idx"] = 0
    missing_stat["pair_idx"] = 1
    missing_stat["TEAM_ID_c"] = missing_stat["TEAM_ID"]

    df_stat_pair = pd.concat([prev_year_stat_c, current_year_stat_c])
    
    df_stat_pair["TEAM_ID_c"] = df_stat_pair["TEAM_ID"]
    
    
    df_stat_pair_filled = pd.concat([df_stat_pair, missing_stat]).reset_index(drop=True).sort_values(by=["GAME_DATE", "DAY_WITHIN_SEASON", "TEAM_ID"], ignore_index=True).groupby("TEAM_ID_c").ffill().sort_values(by=["GAME_DATE", "DAY_WITHIN_SEASON", "TEAM_ID"], ignore_index=True)
    
    
    return df_stat_pair_filled.loc[df_stat_pair_filled["pair_idx"]==1,:].drop(columns=["pair_idx"]).reset_index(drop=True)
  
  
  
  
def weighted(x, cols, w="lag08_GP"):
    return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)
  
def fill_missing_team_stat_playoff(current_playoff_stat: pd.DataFrame, current_regular_stat: pd.DataFrame):
    
    
    current_playoff_stat_c = current_playoff_stat.copy()
    current_regular_stat_c = current_regular_stat.copy()

    df_stat_pair = pd.concat([current_regular_stat_c, current_playoff_stat_c])
    
    target_cols = list(df_stat_pair.columns)
    target_cols.remove('TEAM_ID')
    target_cols.remove('GAME_DATE')
    target_cols.remove("DAY_WITHIN_SEASON")
    
    
    
    df_stat_pair_filled = df_stat_pair.groupby(["GAME_DATE", "DAY_WITHIN_SEASON", "TEAM_ID"]).apply(weighted, target_cols).reset_index()    
    
    return df_stat_pair_filled
  
  
  
  
  
def fill_missing_player_stat_regular(current_year_stat: pd.DataFrame, 
                                   prev_year_stat: pd.DataFrame, 
                                   current_gamelog_data: pd.DataFrame, 
                                   current_rotation_data: pd.DataFrame,
                                   team_idx: int = 1):
    
    
    current_year_stat_c = current_year_stat.copy()
    prev_year_stat_c = prev_year_stat.copy()
    current_gamelog_data_c = current_gamelog_data.copy()
    current_rotation_data_c = current_rotation_data.copy()
    gamelog_rotation_combined = pd.merge(current_gamelog_data_c, current_rotation_data_c[["PLAYER_ID", "GAME_ID"]], on=["GAME_ID",], how="left")
    
    
    gamelog_player_combined = pd.merge(gamelog_rotation_combined, current_year_stat_c, on=["GAME_DATE", "DAY_WITHIN_SEASON","PLAYER_ID"], how="left")
    
    missing_stat_idx = gamelog_player_combined["lag08_W_PCT"].isna()
    missing_stat = gamelog_player_combined.loc[missing_stat_idx, [ "GAME_DATE", "DAY_WITHIN_SEASON", "PLAYER_ID"]]
    
    
    current_year_stat_c["pair_idx"] = 1
    prev_year_stat_c["pair_idx"] = 0
    missing_stat["pair_idx"] = 1
    missing_stat["PLAYER_ID_c"] = missing_stat["PLAYER_ID"]

    df_stat_pair = pd.concat([prev_year_stat_c, current_year_stat_c])
    
    df_stat_pair["PLAYER_ID_c"] = df_stat_pair["PLAYER_ID"]


    df_stat_pair_filled = pd.concat([df_stat_pair, missing_stat]).reset_index(drop=True).sort_values(by=["GAME_DATE", "DAY_WITHIN_SEASON","PLAYER_ID"], ignore_index=True).groupby("PLAYER_ID_c").ffill().sort_values(by=["GAME_DATE", "DAY_WITHIN_SEASON", "PLAYER_ID"], ignore_index=True)

    return df_stat_pair_filled.loc[df_stat_pair_filled["pair_idx"]==1,:].drop(columns=["pair_idx"]).reset_index(drop=True)
  
  
  
  

  
def fill_missing_player_stat_playoff(current_playoff_stat: pd.DataFrame, current_regular_stat: pd.DataFrame):
    
    
    current_playoff_stat_c = current_playoff_stat.copy()
    current_regular_stat_c = current_regular_stat.copy()

    df_stat_pair = pd.concat([current_regular_stat_c, current_playoff_stat_c])
    
    target_cols = list(df_stat_pair.columns)
    target_cols.remove('PLAYER_ID')
    target_cols.remove('GAME_DATE')
    target_cols.remove('DAY_WITHIN_SEASON')
    
    
    
    df_stat_pair_filled = df_stat_pair.groupby(["GAME_DATE", "DAY_WITHIN_SEASON","PLAYER_ID"]).apply(weighted, target_cols).reset_index()    
    
    return df_stat_pair_filled
  
  
  
  
  
def merge_gamelog_to_rotation(current_gamelog: pd.DataFrame, current_rotation: pd.DataFrame, team_idx: int) -> pd.DataFrame:
    
    current_gamelog_c = current_gamelog.copy()
    current_rotation_c = current_rotation.copy()[["PLAYER_ID", "TEAM_ID", "GAME_ID"]]
    
    
    cols = list(current_rotation_c.columns)
    cols.remove("GAME_ID")
    cols.remove("PLAYER_ID")
    
    
    col_map1 = add_prefix_arr(col_arr=cols, prefix=f"team{team_idx}_")
    col_map2 = add_prefix_arr(col_arr=["PLAYER_ID"], prefix=f"team{team_idx}_PStat_")
    current_rotation_c = current_rotation_c.rename(columns=col_map1).rename(columns=col_map2)
    
    
    df_merged = sort_df_cols(pd.merge(current_gamelog_c, current_rotation_c, on=["GAME_ID", f"team{team_idx}_TEAM_ID"], how="left"))
    
    
    return df_merged
  
  
  
def merge_gamelog_to_rotation_x2(current_gamelog: pd.DataFrame, current_rotation: pd.DataFrame) -> pd.DataFrame:
    
    team_idx_1 = 1
    team_idx_2 = 2
    df_merged_0 = merge_gamelog_to_rotation(current_gamelog, current_rotation, team_idx_1)
    
    df_merged_1 = merge_gamelog_to_rotation(df_merged_0, current_rotation, team_idx_2)
    
    
    return df_merged_1
  
  
  
  
  
    
def merge_team_stat_to_gamelog(current_gamelog: pd.DataFrame, current_team_stat: pd.DataFrame, team_idx: int, is_drop: bool) -> pd.DataFrame:    
    
    cols_to_drop = ["GAME_ID"]
    current_gamelog_c = current_gamelog.copy()
    current_team_stat_c = current_team_stat.copy()
    cols = list(current_team_stat.columns)
    cols.remove("GAME_DATE")
    cols.remove("DAY_WITHIN_SEASON")

    col_map1 = add_prefix_arr(col_arr=cols, prefix=f"team{team_idx}_")

    current_team_stat_c = current_team_stat_c.rename(columns=col_map1)

    df_merged_0 = sort_df_cols(pd.merge(current_gamelog_c, current_team_stat_c, on=["GAME_DATE", "DAY_WITHIN_SEASON", f"team{team_idx}_TEAM_ID"], how="left"))
    
    if is_drop:
        df_merged_0.drop(columns=cols_to_drop, inplace=True)
    
    
    return df_merged_0

    
def merge_player_stat_to_gamelog(current_gamelog_rotation_combined: pd.DataFrame, current_player_stat: pd.DataFrame, team_idx: int, is_drop: bool) -> pd.DataFrame:    
    
    cols_to_drop = ["GAME_ID"]
    current_gamelog_rotation_combined_c = current_gamelog_rotation_combined.copy()
    current_player_stat_c = current_player_stat.copy()
    cols = list(current_player_stat_c.columns)
    cols.remove("GAME_DATE")
    cols.remove("DAY_WITHIN_SEASON")

    col_map1 = add_prefix_arr(col_arr=cols, prefix=f"team{team_idx}_PStat_")

    current_player_stat = current_player_stat.rename(columns=col_map1)

    df_merged_0 = sort_df_cols(pd.merge(current_gamelog_rotation_combined_c, current_player_stat, on=["GAME_DATE", "DAY_WITHIN_SEASON", f"team{team_idx}_PStat_PLAYER_ID"], how="left"))
    
    if is_drop:
        df_merged_0.drop(columns=cols_to_drop, inplace=True)
    
    
    return df_merged_0
  
  
  
  
def get_agg_player_stat_to_gamelog(current_gamelog: pd.DataFrame, 
                                   current_player_stat: pd.DataFrame, 
                                   current_rotation: pd.DataFrame) -> pd.DataFrame:
    
    gamelog_rotation_combined = merge_gamelog_to_rotation_x2(current_gamelog, current_rotation)
    
    team_idx_1 = 1
    team_idx_2 = 2
    merged_0 = merge_player_stat_to_gamelog(gamelog_rotation_combined, current_player_stat, team_idx_1, False)
    merged_1 = merge_player_stat_to_gamelog(merged_0, current_player_stat, team_idx_2, False)
    
    
    # drop na player statistics
    na_players_team1 = merged_1["team1_PStat_lag08_AST_RANK"].isna()
    na_players_team2 = merged_1["team2_PStat_lag08_AST_RANK"].isna()
    
    merged_1 = merged_1.loc[~na_players_team1,:]
    merged_1 = merged_1.loc[~na_players_team2,:]

    gameId_date_map = merged_1[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    last_cols_to_drop = ["team1_PStat_PLAYER_ID", "team2_PStat_PLAYER_ID"]
    merged_1 = merged_1.groupby("GAME_ID").mean().reset_index().drop(columns=last_cols_to_drop)

    merged_1 = pd.merge(merged_1, gameId_date_map, on="GAME_ID", how="left")

    return merged_1
    
    