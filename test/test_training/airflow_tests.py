import unittest
from datetime import datetime
from airflow import DAG
from airflow.models import TaskInstance
from airflow.operators.bash_operator import BashOperator


class Retrieve_New_Model_Record_ID(unittest.TestCase):

    def test_execute(self):
        dag = DAG(dag_id='local_model_training', start_date=datetime.now())
        task = BashOperator(my_operator_param=10, dag=dag, task_id='training_request_processing')
        ti = TaskInstance(task=task, execution_date=datetime.now())
        result = task.execute(ti.get_template_context())
        # Based on the current database record we should get model record id : 15
        self.assertEqual(result, 15)


suite = unittest.TestLoader().loadTestsFromTestCase(Retrieve_New_Model_Record_ID)
unittest.TextTestRunner(verbosity=2).run(suite)