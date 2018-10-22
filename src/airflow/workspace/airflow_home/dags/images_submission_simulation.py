from __future__ import print_function
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.hooks import SSHHook
from airflow.contrib.operators.ssh_execute_operator import SSHExecuteOperator
from airflow.models import DAG
from datetime import datetime



args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
}

dag = DAG(
    dag_id='image_submission_simulation', default_args=args,
    schedule_interval=None)


templated_command_distribute_labels = """


split -l 160 ~/Deep_Images_Hub/data/all_labels.txt ~/sample_labels_validation_


scp ~/sample_labels_validation_ab ubuntu@${Node2}:~/

scp ~/sample_labels_validation_ac ubuntu@${Node3}:~/

scp ~/sample_labels_validation_ad ubuntu@${Node4}:~/



"""

Distribute_labels_to_nodes = BashOperator(
    		task_id='Distribute_labels',
    		bash_command=templated_command_distribute_labels,
		    dag=dag)



templated_command_Node_1= """

sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/sample_labels_validation_aa "validation"


"""

Upload_images_From_Node_1 = SSHExecuteOperator(
    task_id="Upload_images_From_Node_1",
    bash_command=templated_command_Node_1,
    dag=dag)


sshHook_node2 = SSHHook(conn_id='Node_2')


templated_command_Node_2= """

sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/sample_labels_validation_ab "validation"


"""

Upload_images_From_Node_2 = SSHExecuteOperator(
    task_id="Upload_images_From_Node_2",
    bash_command=templated_command_Node_2,
    ssh_hook=sshHook_node2,
    dag=dag)



sshHook_node3 = SSHHook(conn_id='Node_3')


templated_command_Node_3= """

sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/sample_labels_validation_ac "validation"


"""

Upload_images_From_Node_3 = SSHExecuteOperator(
    task_id="Upload_images_From_Node_3",
    bash_command=templated_command_Node_3,
    ssh_hook=sshHook_node3,
    dag=dag)



sshHook_node4 = SSHHook(conn_id='Node_4')

templated_command_Node_4= """

sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/sample_labels_validation_ad "validation"


"""

Upload_images_From_Node_4 = SSHExecuteOperator(
    task_id="Upload_images_From_Node_4",
    bash_command=templated_command_Node_4,
    ssh_hook=sshHook_node4,
    dag=dag)



Distribute_labels_to_nodes >> templated_command_Node_1

Distribute_labels_to_nodes >> templated_command_Node_2

Distribute_labels_to_nodes >> templated_command_Node_3

Distribute_labels_to_nodes >> templated_command_Node_4