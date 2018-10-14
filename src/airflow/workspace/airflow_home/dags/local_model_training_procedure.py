import datetime as dt
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator



# dummy_task = DummyOperator(task_id='dummy_task', dag=dag)
#


def print_world():
    print('world')


default_args = {
    'owner': 'me',
    'start_date': dt.datetime(2017, 6, 1),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}


with DAG(dag_id='local_model_training',
         description='Training Model with Keras locally',
         default_args=default_args,
         schedule_interval='0 * * * *',
         ) as dag:
    #
    # print_hello = BashOperator(task_id='print_hello',
    #                            bash_command='echo "hello"')
    # sleep = BashOperator(task_id='sleep',
    #                      bash_command='sleep 5')
    # print_world = PythonOperator(task_id='print_world',
    #                              python_callable=print_world)

    templated_command_1 = """

    cd ~/Deep_Images_Hub
    python src/training/requester.py  --label_List Pen Table Lamp --user_id 2
    script_output=$?
    echo $script_output

    """

    training_request_handler = BashOperator(
            task_id='training_request_processing',
            bash_command=templated_command_1)


    templated_command_2 = """

    cd ~/Deep_Images_Hub
    
    python src/training/cnn-keras/train.py --dataset /tmp/Deep_image_hub_Model_Training/dataset --model /tmp/Deep_image_hub_Model_Training/model/sample_model.model --plot /tmp/Deep_image_hub_Model_Training/model/plot.png --labelbin lb.pickle --training_request_number 14

    """

    training_model_in_action = BashOperator(
            task_id='training_model_in_action',
            bash_command=templated_command_2)


    templated_command_3 = """

   cd ~/Deep_Images_Hub

   python src/training/postTraining.py --training_request_number 14  --user_id 2
   """

    post_training_handler = BashOperator(
        task_id='post_training_processing',
        bash_command=templated_command_3)




training_request_handler >> training_model_in_action >> post_training_handler









#
# templated_command_1 = """
#
#
# python ~/Deep_Images_Hub/src/training/requester.py  --label_List {{ dag_run.conf.Labels }} --user_id {{ dag_run.conf.User_id }}
# script_output=$?
# echo $script_output
#

#
# """

#
# training_request_handler = \
#     BashOperator(
#         task_id='training_request_processing',
#         bash_command=templated_command_1,
#         dag=dag)
#
# dummy_task >> training_request_handler