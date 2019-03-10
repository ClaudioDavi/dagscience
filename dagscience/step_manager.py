import logging
import os
import configparser
import traceback


class Step():
    """
    Steps to be executed to create the machine learning model
    """
    logger = logging.getLogger(__name__)
    default = {
        "STEPS": {
            "STEP_1": 'ready',
            "STEP_2": 'ready',
            "STEP_3": 'ready'
        }
    }

    def __init__(self, get_data, preprocess, train, model_saver):
        self.data_class = get_data
        self.preprocess_class = preprocess
        self.train_class = train
        self.model_class = model_saver

    def execute_steps(self, step_1=True, step_2=True, step_3=True):
        data = self.step_get_data(step_1)
        data = self.step_preprocess(data, step_2)
        model = self.step_train(data, step_3)
        return model

    def step_get_data(self, execute):
        self.logger.info('=============================================')
        self.logger.info('Step 1: Loading Data')
        if not execute:
            self.logger.info(
                'Step 1: Not getting new data, loading from file system directly')
            return self.data_class.load_from_filesystem()
        else:
            self.logger.info(
                'Step 1: Loading data from original source and saving to filesystem')
            data = self.data_class.load_from_source()
            self.data_class.save(data)
            return data

    def step_preprocess(self, data, execute):
        self.logger.info('=============================================')
        self.logger.info('Step 2: Preprocess')
        if not execute:
            self.logger.info('Step 2: Not preprocessing')
            return data
        else:
            self.logger.info("Step 2: Starting preprocessing step")
            return self.preprocess_class.run(data)

    def step_train(self, data, execute):
        self.logger.info('=============================================')
        self.logger.info('Step 3: Training Model')
        if not execute:
            self.logger.info(
                'Step 3: Not using training, loading model from file system')
            return self.model_class.load()
        else:
            self.logger.info('Step 3: Building and training model')
            model = self.train_class.build_model()
            model = self.train_class.run(model, data)
            self.model_class.save(model)
            return model

    def step_writer(self):
        config = configparser.ConfigParser()
        if os.path.exists('.steps'):
            config.read('.steps')
            print(config.sections)
        else:
            with open('.steps', 'w') as configfile:
                try:
                    config.read_dict(self.default)
                    config.write(configfile)
                except Exception as ex:
                    traceback.print_stack()
