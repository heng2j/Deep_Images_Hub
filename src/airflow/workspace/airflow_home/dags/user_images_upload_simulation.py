from __future__ import print_function
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators import PythonOperator
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


peg scp to-rem pySpark-cluster 1 ~/sample_labelsaa /home/ubuntu/sample_labelsaa

peg sshcmd-node pySpark-cluster 1 "touch dummy_from_airflow.txt" 

peg sshcmd-node pySpark-cluster 1 "nohup sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/sample_labelsaa "test"  > ~/Deep_Images_Hub/src/producer/auto_upload.log & " 

"""



# peg ssh pySpark-cluster {{ node_number }}
#

# sh /home/ubuntu/Deep_Images_Hub/src/producer/auto_upload.sh  {{ src_lables_file }} {{ src_type }}  > /home/ubuntu/Deep_Images_Hub/src/producer/auto_upload.log &



Labels_prep = \
BashOperator(
    task_id='Batch_Image_Submissions',
    bash_command=templated_command,
    params = {'node_number': 1, 'sample_file' : 'sample_labelsaa' , 'src_lables_file' : '~/sample_labelsaa' , 'src_type' : 'test'   },
dag=dag)


parent_on_node1 = None
for i in range(120):
    '''
    Generating 10 sleeping task, sleeping from 0 to 9 seconds
    respectively
    '''
    task_on_node1 = \
	BashOperator(
    		task_id='Batch_Image_Submissions',
    		bash_command=templated_command,
    		params = {'node_number': 1, 'sample_file' : 'sample_labelsaa' , 'src_lables_file' : '~/sample_labelsaa' , 'src_type' : 'test'   },
		dag=dag)	
	
    if parent_on_node1:
        task_on_node1.set_upstream(parent_on_node1)

    parent_on_node1 = task_on_node1


