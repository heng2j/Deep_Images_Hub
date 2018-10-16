from __future__ import print_function
from airflow.operators.bash_operator import BashOperator
from airflow.models import DAG
from datetime import datetime

args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
}

dag = DAG(
    dag_id='image_submission_simulation', default_args=args,
    schedule_interval=None)

def print_context(i):
    print(i)
    return 'print_context has sucess {}'.format(i)

templated_command = """

cd ~
sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/Deep_Images_Hub/data/all_labels_For_Simulation.txt "validation"

"""

parent_on_node1 = None
for i in range(120):

    task_on_node1 = \
	BashOperator(
    		task_id='Batch_Image_Submissions',
    		bash_command=templated_command,
		dag=dag)	
	
    if parent_on_node1:
        task_on_node1.set_upstream(parent_on_node1)

    parent_on_node1 = task_on_node1



