from abc import ABC, abstractmethod
import logging
from .step_manager import Step


class DagflowCycle:
    """
    Every cicle on a DAG workflow begins with run.
    To implement a workflow you should build all the
    Tasks classes according to your needs and then pass
    them as parameters to the DagFlowCicle Object.
    After that, the workflow will take care of the process
    """

    logger = logging.getLogger(__name__)

    def __init__(self, task_get_data, task_preprocess, task_train, task_model_saver):
        """
        Creates workflow.

        params:

            task_get_data -- Implementation of the TaskGetData class

            task_preprocess -- Implementation of the TaskPreprocess class

            task_train -- Implementation of the TaskTrain class

            task_model_saver -- Implementation of the TaskSaveModel class

        returns:

            The workflow Cycle
        """
        if (
            issubclass(task_get_data.__class__, TaskGetData)
            and issubclass(task_preprocess.__class__, TaskPreprocess)
            and issubclass(task_train.__class__, TaskTrain)
            and issubclass(task_model_saver.__class__, TaskSaveModel)
        ):
            self.step = Step(task_get_data, task_preprocess, task_train, task_model_saver)

    def run(self, step_1=True, step_2=True, step_3=True):
        """
        Runs the workflow cycle you can disable steps as a parameter.
        Will return the Machine Learning Model

        params:

            step_1 -- Default(True) Enables Loading the data from external sources. 
            If false will load from disk, or as defined in load_from_filesystem

            step_2 -- Default(True) Enables the preprocessing of the data. 
            If false will return the original data.

            step_3 -- Default(True) Enables the creation and training of the model. 
            If false will only load model from file system

        returns:
            Machine learning model.
        """
        return self.step.execute_steps(step_1, step_2, step_3)

    def describe(self):
        pass


class TaskGetData(ABC):
    """Abstract class to load data from the specified sources"""

    def __init__(self):
        pass

    @abstractmethod
    def load_from_source(self, *args, **kwargs):
        """
        Loads data from the source
        """

    @abstractmethod
    def load_from_filesystem(self, *args, **kwargs):
        """
        Loads data from filesystem
        """

    @abstractmethod
    def save(self, data, *args, **kwargs):
        """
        Saves data to the repository
        """


class TaskPreprocess(ABC):
    """Abstract class to preprocess the data"""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, data, *args, **kwargs):
        """
        Does Preprocessing on the data, returns dataframe
        """


class TaskTrain(ABC):
    """ Abstract class to train and build the machine learning model"""

    def __init__(self):
        pass

    @abstractmethod
    def build_model(self):
        """
        builds your machine learning algorithm.
        Use this for hiperparemeter tuning and not bloat the run method
        """

    @abstractmethod
    def run(self, model, data):
        """
        Runs the training job
        """


class TaskSaveModel(ABC):
    """Abstract class to save the machine learning model"""

    def __init__(self):
        pass

    @abstractmethod
    def save(self, model):
        """
        Saves the model to the destination output
        """

    @abstractmethod
    def load(self):
        """
        Loads the model from target destination
        """
