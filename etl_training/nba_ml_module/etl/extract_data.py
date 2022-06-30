#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np
import bs4

from datetime import datetime, timedelta
from bs4 import BeautifulSoup

###########   Part of Code for stats.nba.com endpoint   #############

nba_stats_headers = {
                    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36',
                    "host": 'stats.nba.com',
                    "cache-control":"max-age=0",
                    "connection": 'keep-alive',
                    "accept-encoding" : "Accepflate, sdch",
                    'accept-language':'he-IL,he;q=0.8,en-US;q=0.6,en;q=0.4',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'accept-language':'he-IL,he;q=0.8,en-US;q=0.6,en;q=0.4',
                    'Host': 'stats.nba.com',
                    'accept-language':'he-IL,he;q=0.8,en-US;q=0.6,en;q=0.4',
                    'Referer': 'https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1&Season=2019-20&SeasonType=Regular%20Season',
                    'Connection': 'keep-alive',
                    'x-nba-stats-origin': 'stats',
                    'x-nba-stats-token': 'true',
                    'Origin': 'https://stats.nba.com'
                    }


def jsonToDataFrame(jsonData):
  """TRansforms json data extracted from nba api into list of dataframes and returns it"""
  # Based on this API, there are two possible options for resultant data
  # resultSet or resultSets

  # list of pandas dataframe
  dataFrame_arr = []

  # Process one resultSet
  if "resultSet" in jsonData.keys():
    rows = jsonData['resultSet']['rowSet']
    columns = jsonData['resultSet']['headers']
    df = pd.DataFrame(rows, columns=columns)
    dataFrame_arr.append(df)

  elif "resultSets" in jsonData.keys():
    for resultSet in jsonData['resultSets']:
      rows = resultSet['rowSet']
      columns = resultSet['headers']
      df = pd.DataFrame(rows, columns=columns)
      dataFrame_arr.append(df)
  
  return dataFrame_arr



def getRawTeamStats(conference="", dateFrom="", dateTo="", division="", gameScope="", gameSegment="", 
                    lastNGames=100, leagueId="00", location="", measureType="Base", month=0, opponentId=0,
                    outcome="", PORound="", PaceAdjust="N", perMode="Totals", period=0, playerExperience="",
                    playerPosition="", plusMinus="N", rank="N", season="2019-20", seasonSegment="", 
                    seasonType="Regular Season", shotClockRange="", starterBench="", teamID="", twoWay="",
                    vsConference="", vsDivision="", headers=nba_stats_headers, proxy_config={}):
  
  """Transforms json data extracted from nba api into list of dataframes and returns it
    Data: Statistics of individual teams, determined by parameters in get request
    Corresponding end point: https://stats.nba.com/stats/leaguedashteamstats"""
  
  # End point from which the data to be extracted
  url = "https://stats.nba.com/stats/leaguedashteamstats"
  
  # Parameters for the end point
  payload = {
        'Conference': conference,
        'DateFrom': dateFrom,
        'DateTo': dateTo,
        'Division': division,
        'GameScope': gameScope,
        'GameSegment': gameSegment,
        'LastNGames': lastNGames,
        'LeagueID': leagueId,
        'Location': location,
        'MeasureType': measureType,
        'Month': month,
        'OpponentTeamID': opponentId,
        'Outcome': outcome,
        'PORound': PORound,
        'PaceAdjust': PaceAdjust,
        'PerMode': perMode,
        'Period': period,
        'PlayerExperience': playerExperience,
        'PlayerPosition': playerPosition,
        'PlusMinus': plusMinus,
        'Rank': rank,
        'Season': season,
        'SeasonSegment': seasonSegment,
        'SeasonType': seasonType,
        'ShotClockRange': shotClockRange,
        'StarterBench': starterBench,
        'TeamID': teamID,
        'TwoWay': twoWay,
        'VsConference': vsConference,
        'VsDivision': vsDivision
      }
  
  # Turn response into json format
  jsonData = requests.get(url, headers=headers, params=payload, proxies=proxy_config).json()
  return jsonData



def getRawPlayerStats(college="", conference="", country="", dateFrom="", dateTo="", division="", draftPick="", draftYear="", 
                      gameScope="", gameSegment="", height="", lastNGames=100, leagueId="00", location="", measureType="Base", 
                      month=0, opponentId=0, outcome="", PORound="", paceAdjust="N", perMode="Totals", period=0, playerExperience="",
                      playerPosition="", plusMinus="N", rank="N", season="2019-20", seasonSegment="", seasonType="Regular Season", 
                      shotClockRange="", starterBench="", teamID="", twoWay="", vsConference="", vsDivision="", weight="", 
                      headers=nba_stats_headers, proxy_config={}):
  
  
  """Transforms json data extracted from nba api into list of dataframes and returns it
    Data: Statistics of individual players, determined by parameters in get request
    Corresponding end point: https://stats.nba.com/stats/leaguedashplayerstats"""
  
  # End point from which the data to be extracted
  url = "https://stats.nba.com/stats/leaguedashplayerstats"
  
  # Parameters for the end point
  payload = {
      'College': college,
      'Conference': conference,
      'Country': country,
      'DateFrom': dateFrom,
      'DateTo': dateTo,
      'Division': division,
      'DraftPick': draftPick,
      'DraftYear': draftYear,
      'GameScope': gameScope,
      'GameSegment': gameSegment,
      'Height': height,
      'LastNGames': lastNGames,
      'LeagueID': leagueId,
      'Location': location,
      'MeasureType': measureType,
      'Month': month,
      'OpponentTeamID': opponentId,
      'Outcome': outcome,
      'PORound': PORound,
      'PaceAdjust': paceAdjust,
      'PerMode': perMode,
      'Period': period,
      'PlayerExperience': playerExperience,
      'PlayerPosition': playerPosition,
      'PlusMinus': plusMinus,
      'Rank': rank,
      'Season': season,
      'SeasonSegment': seasonSegment,
      'SeasonType': seasonType,
      'ShotClockRange': shotClockRange,
      'StarterBench': starterBench,
      'TeamID': teamID,
      'TwoWay': twoWay,
      'VsConference': vsConference,
      'VsDivision': vsDivision,
      'Weight': weight
      }
  
  # Turn response into json format
  jsonData = requests.get(url, headers=headers, params=payload, proxies=proxy_config).json()
  return jsonData
  

def getRawGameLog(counter="0", playerTeam="T", leagueId="00", season="2018-19", seasonType="Regular Season", sort="DATE", direction="ASC", dateFrom="",
                  dateTo="", headers=nba_stats_headers, proxy_config={}):
  
  """Extracts game logs from nba.stats.com and turns in JSON format
    Data: Log of each game, point difference, name of teams, home and visitor, etc
    Correponding end point: https://stats.nba.com/stats/leaguegamelog"""
  
  # End point from which the data is to be extracted
  url = "https://stats.nba.com/stats/leaguegamelog"#
  
  # Parameters for get request
  payload = {
    'Counter': counter,
    'DateFrom': dateFrom,
    'DateTo': dateTo,
    'Direction': direction,
    'LeagueID': leagueId,
    'PlayerOrTeam': playerTeam,
    'Season': season,
    'SeasonType': seasonType,
    'Sorter': sort}

  # Turn into json format
  jsonData = requests.get(url, headers=headers, params=payload, proxies=proxy_config).json()
  return jsonData




def getRawGameRotation(gameId, leagueId="00", headers=nba_stats_headers, proxy_config={}):
  """Extracts game logs from nba.stats.com and turns in JSON format
    Data: Log of each game, point difference, name of teams, home and visitor, etc
    Correponding end point: https://stats.nba.com/stats/gamerotation"""
  
  # End point from which the data is to be extracted
  url = "https://stats.nba.com/stats/gamerotation"
  
  # Parameters for get request
  payload = {
    'GameID': gameId,
    'LeagueID': leagueId,
    }
  
  # Turn into json format
  jsonData = requests.get(url, headers=headers, params=payload, proxies=proxy_config).json()
  return jsonData




def get_team_data_lagged(date: str, season: str, seasonType: str, lag_length: int, proxy_config: dict) -> list:

  today_obj = datetime.strptime(date, "%Y-%m-%d")
  before_day_1 = today_obj - timedelta(days=1)

  before_day_lag_len = today_obj - timedelta(days=1+lag_length)
  print(before_day_1.isoformat()[:10], before_day_lag_len.isoformat()[:10])
  df_arr = jsonToDataFrame(getRawTeamStats(dateTo=before_day_1.isoformat()[:10], 
                                            dateFrom=before_day_lag_len.isoformat()[:10],
                                            season=season, seasonType=seasonType, perMode="PerMinute", proxy_config=proxy_config))

  return df_arr




def get_player_data_lagged(date: str, season: str, seasonType: str, lag_length: int, proxy_config: dict) -> list:

  today_obj = datetime.strptime(date, "%Y-%m-%d")
  before_day_1 = today_obj - timedelta(days=1)

  before_day_lag_len = today_obj - timedelta(days=1+lag_length)
  print(before_day_1.isoformat()[:10], before_day_lag_len.isoformat()[:10])
  df_arr = jsonToDataFrame(getRawPlayerStats(dateTo=before_day_1.isoformat()[:10], 
                                            dateFrom=before_day_lag_len.isoformat()[:10],
                                            season=season, seasonType=seasonType, perMode="PerMinute", proxy_config=proxy_config))

  return df_arr





###########   Part of Code for webscraping   #############

NBA2K_ENDPOINT = "https://hoopshype.com/nba2k"


def get_nba2k_rating_df(url: str = NBA2K_ENDPOINT, proxy_config: dict = {}) -> pd.DataFrame:
  """Scrapes nba2k rating from url and turns into pandas dataframe"""
  
  endpoint = url
  
  # Get response
  page = requests.get(endpoint, proxies=proxy_config)
  
  # Parse the HTML page
  soup = BeautifulSoup(page.text, 'html.parser')
  
  
  # Extract table from 
  # Tag classes are based on HTML elements of the website
  table = soup.find("div", id='content').find("div", id="content-container").find("div", class_="hh-ranking-ranking").find("tbody")

  names_arr = []
  values_arr = []
  
  for i,child in enumerate(table.children):
    
    # Some of the child tags are empty strings
    # Bypass these empty strings
    if isinstance(child, bs4.element.NavigableString):
      continue

    names_arr.append(child.find("td", class_="name").text.strip())
    values_arr.append(child.find("td", class_="value").text.strip())

  # Transform into pandas dataframe
  data_dict = {"name": names_arr, "rating": values_arr}
  return pd.DataFrame.from_dict(data_dict)



