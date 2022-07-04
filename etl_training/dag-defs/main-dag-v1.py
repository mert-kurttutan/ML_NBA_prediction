from airflow import DAG
from airflow.operators.python import PythonOperator
import boto3
from datetime import timedelta, datetime
from typing import List


# Client settings
SERVICE_NAME = "ecs"
REGION_NAME = "us-east-1"
DELAY = 6
MAX_ATTEMPTS = 1000


# settings for running task
LAUNCH_TYPE = "FARGATE"
CLUSTER_NAME = "NBA-ML-FLOW-CLUSTER"
PLATFORM_VERSION = "LATEST"
CONTAINER_NAME = "nba-mlflow-container"
NETWORK_CONFIG = {
  'awsvpcConfiguration': {
    'subnets': ['subnet-0387bc990898b9097','subnet-0fcf53881946f87a8'],
    'assignPublicIp': 'ENABLED',
    'securityGroups': ["sg-0e266710159c321dc"]
  }
}

main_command = ['bash', '-c', ('source /entrypoint.sh && set_python_env && get_ml_script' +
                '&& echo ${PYTHONPATH} && python ${MLFLOW_HOME}/ml-scripts/main.py --year-arr ${YEAR_ARR}')]

main_command_0 = ['bash', '-c', ('source /entrypoint.sh && set_python_env ')]

# Args to run_ecs_task function so that they run intended python script/ml dag
TASK_GAMELOG_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command_0,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_get_gamelog.env',
              "description": "get_gamelog"
            }


TASK_GAME_ROTATION_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command_0,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_get_game_rotation.env',
              "description": "get_game_rotation"
            }


TASK_TEAM_STAT_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command_0,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_get_team_stat.env',
              "description": "get_team_stat"
            }

TASK_PLAYER_STAT_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command_0,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_get_player_stat.env',
              "description": "get_player_stat"
            }



TASK_TRANSFORM_V1_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_transform_v1.env',
              "description": "transform_v1"
            }


TASK_TRANSFORM_V2_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_transform_v2.env',
              "description": "transform_v2"
            }


TASK_TRAINING_DATA_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_training_data.env',
              "description": "get_training_data"
            }


TASK_DYNAMODB_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_dynamodb.env',
              "description": "write_to_dynamodb"
            }

TASK_TRAIN_MODEL_KWARGS = {
              "TASK_DEF": 'nba_mlflow_vtrial', 
              "TASK_COMMAND": main_command,
              "TASK_ENV": 'arn:aws:s3:::mert-kurttutan-nba-ml-files/env-files/cfg_train_model.env',
              "description": "train_model"
            }


def run_ecs_task(TASK_DEF: str, TASK_ENV: str, TASK_COMMAND: List[str], description: str):
  
  client = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)
  response = client.run_task(
                      taskDefinition=TASK_DEF,
                      launchType=LAUNCH_TYPE,
                      cluster=CLUSTER_NAME,
                      platformVersion=PLATFORM_VERSION,
                      count=1,
                      networkConfiguration=NETWORK_CONFIG,
                      overrides={
                                  'containerOverrides': [
                                    {
                                      "name": CONTAINER_NAME, 
                                      'command': TASK_COMMAND,
                                      'environmentFiles': [
                                        {
                                          'value': TASK_ENV,
                                          'type': 's3'
                                        },
                                      ]
                                    }
                                  ]
                                })
  print(f"Started running task {description}...")

  task_arn = response["tasks"][0]["containers"][0]["taskArn"]

  # Wait until task is stopped
  waiter = client.get_waiter('tasks_stopped')

  waiter.wait(
              cluster=CLUSTER_NAME,
              tasks=[task_arn],
              WaiterConfig={
                            'Delay': DELAY,
                            'MaxAttempts': MAX_ATTEMPTS
                            }
          )

  print(f"Task {description} is finished ...")
  task_description = client.describe_tasks(
                                    cluster=CLUSTER_NAME,
                                    tasks=[task_arn]
                                    )


one_days_ago = datetime.combine(datetime.today() - timedelta(1),
                                      datetime.min.time())



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': one_days_ago,
    'email': ['kurttutan.mert@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    #'schedule': '@once',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
  }




dag = DAG('nba_main_dag', default_args=default_args)



t_gamelog = PythonOperator(
    task_id = TASK_GAMELOG_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_GAMELOG_KWARGS,
    dag=dag,
)

t_game_rotation = PythonOperator(
    task_id = TASK_GAME_ROTATION_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_GAME_ROTATION_KWARGS,
    dag=dag,
)

t_team_stat = PythonOperator(
    task_id = TASK_TEAM_STAT_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_TEAM_STAT_KWARGS,
    dag=dag,
)

t_player_stat = PythonOperator(
    task_id = TASK_PLAYER_STAT_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_PLAYER_STAT_KWARGS,
    dag=dag,
)



t_transform_v1 = PythonOperator(
    task_id = TASK_TRANSFORM_V1_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_TRANSFORM_V1_KWARGS,
    dag=dag,
)

t_transform_v2 = PythonOperator(
    task_id = TASK_TRANSFORM_V2_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_TRANSFORM_V2_KWARGS,
    dag=dag,
 )
t_training_data = PythonOperator(
    task_id = TASK_TRAINING_DATA_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_TRAINING_DATA_KWARGS,
    dag=dag,
)

t_dynamodb = PythonOperator(
    task_id = TASK_DYNAMODB_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_DYNAMODB_KWARGS,
    dag=dag,
)

t_train_model = PythonOperator(
    task_id = TASK_TRAIN_MODEL_KWARGS["description"],
    python_callable = run_ecs_task,
    op_kwargs = TASK_TRAIN_MODEL_KWARGS,
    dag=dag,
)

t_gamelog >> t_game_rotation


t_game_rotation >> t_transform_v1

t_team_stat >> t_transform_v1

t_player_stat >> t_transform_v1


t_transform_v1 >> t_transform_v2 >> t_training_data

t_transform_v2 >> t_dynamodb

t_training_data >> t_train_model