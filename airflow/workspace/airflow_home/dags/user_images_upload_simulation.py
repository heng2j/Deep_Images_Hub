from __future__ import print_function
from airflow.operators import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.models import DAG
from datetime import datetime

args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
}

dag = DAG(
    dag_id='user_images_uploading', default_args=args,
    schedule_interval=None)

def print_context(i):
    print(i)
    return 'print_context has sucess {}'.format(i)

templated_command = """


sh /home/ubuntu/jupyter_config/Deep_Images_Hub/src/producer/auto_upload.sh  {{ src_lables_file }} {{ src_type }}  > /home/ubuntu/jupyter_config/Deep_Images_Hub/src/producer/auto_upload.log &

"""



parent = None
for i in range(500):
    '''
    Generating 10 sleeping task, sleeping from 0 to 9 seconds
    respectively
    '''
    task = \
	BashOperator(
    		task_id='templated',
    		bash_command=templated_command,
    		params = {'src_lables_file' : '~/sample_labels.txt' , 'src_type' : 'test'   },	
		dag=dag)	
	
    if parent:
        task.set_upstream(parent)

    parent = task
