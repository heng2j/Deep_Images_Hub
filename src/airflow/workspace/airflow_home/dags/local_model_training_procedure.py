from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator


args = {'start_date':datetime.now()
        }

dag = DAG(
          dag_id='local_model_training',
          description='Training Model with Keras locally',
          schedule_interval='0 12 * * *',
          default_args=args,
          catchup=False)

dummy_task = DummyOperator(task_id='dummy_task', dag=dag)


templated_command_1 = """


python ~/Deep_Images_Hub/src/training/requester.py  --label_List {{ dag_run.conf.Labels }} --user_id {{ dag_run.conf.User_id }} 
script_output=$?
echo $script_output


"""


training_request_handler = \
    BashOperator(
        task_id='training_request_processing',
        bash_command=templated_command_1,
        dag=dag)

dummy_task >> training_request_handler