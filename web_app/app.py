from flask import Flask, render_template, request
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import boto3
import nba_ml_module.deploy as deploy

# Initialise the Flask app
app = Flask(__name__)

# Use pickle to load in the pre-trained model
filename = "models/nba-pred.xgb"
# XGboost model
model = xgb.XGBClassifier()
model.load_model(filename)

GAME_DATE = "2013-02-11"
SEASON = 6
IS_REGULAR = True



boto3_ses = boto3.Session()

dynamodb_resource = boto3_ses.resource(service_name="dynamodb")

dynamodb_client = boto3_ses.client(service_name="dynamodb")

nba_gamelog_table_name = "gamelog-table-vtrial"
nba_team_table_name = "team-stat-table-vtrial"



# Opening JSON file
with open('configs/team_name_id_map.json') as json_file:
    team_id_dict = json.load(json_file)["TEAM_ID"]


# Opening JSON file
with open('configs/data_config.json') as json_file:
     data_config = json.load(json_file)


def str_to_bool(word: str) -> bool:
  if word == "True":
    return True
  else: 
    return False



def finish_prediction(team_names, pred_result):
  if pred_result > 0.5:
    winning_team = team_names[0]
    winning_probability = pred_result
  else:
    winning_team = team_names[1]
    winning_probability = 1 - pred_result
  
  return winning_team, winning_probability * 100   # probability percentage


# Set up the main route
@app.route('/', methods=["GET", "POST"])
def main():

    if request.method == "POST":
        # Extract the input from the form
        team1_TEAM_NAME = request.form.get("team1_name")
        team2_TEAM_NAME = request.form.get("team2_name")
        team1_TEAM_ID = int(team_id_dict[team1_TEAM_NAME])
        team2_TEAM_ID = int(team_id_dict[team2_TEAM_NAME])
        IS_HOME = float(str_to_bool(request.form.get("teamhome_name")))
        print(list(request.form.keys()))

        X_pred = deploy.get_X_pred(team1_id=team1_TEAM_ID, team2_id=team2_TEAM_ID,
           game_date=GAME_DATE, team1_IS_HOME=IS_HOME, IS_REGULAR=IS_REGULAR, SEASON=SEASON,
           nba_team_table_name=nba_team_table_name, nba_gamelog_table_name=nba_gamelog_table_name,
           dynamodb_client=dynamodb_client)
        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        pred_probs = model.predict_proba(X_pred[data_config["selected_cols"]])[0]
        result_idx = np.argmax(pred_probs)
        if result_idx == 1:
          winning_team = team1_TEAM_NAME
        else:
          winning_team = team2_TEAM_NAME
        print(pred_probs)

        df = pd.DataFrame({
                "team": np.array([team2_TEAM_NAME, team1_TEAM_NAME]),
                "winning_probability": np.array(pred_probs)
            })
        fig = px.bar(df, x="team", y="winning_probability",
                    barmode='group',height=400, width=500)
                    
                    
        fig.update_layout(barmode='group',bargroupgap=0.4)
        fig.update_layout(title_text="Winning probabilities for the match", title_x=0.5)
                    
                    
        #fig.update_layout(barmode='group', bargap=0.30,bargroupgap=0.0)
        fig.update_layout(barmode='group',bargroupgap=0.4)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    
        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html",
                                     original_input={'team1_name': team1_TEAM_NAME,
                                                     'team2_name': team2_TEAM_NAME,
                                                     'game_date': GAME_DATE},
                                    result= winning_team, team_id_dict=team_id_dict, graphJSON=graphJSON
                                     )
    # If the request method is GET
    return render_template("index.html", team_id_dict=team_id_dict)