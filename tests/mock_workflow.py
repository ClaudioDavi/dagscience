from dagscience import workflow


class MockGetData(workflow.TaskGetData):
    def __init__(self):
        pass

    def load_from_source(self, *args, **kwargs):
        pass

    def load_from_filesystem(self, *args, **kwargs):
        pass

    def save(self, data, *args, **kwargs):
        pass


class MockPreprocess(workflow.TaskPreprocess):
    def __init(self):
        pass

    def run(self):
        pass


class MockTrain(workflow.TaskTrain):
    def __init__(self):
        pass

    def build_model(self):
        pass

    def run(self, model, data):
        pass


class MockSaveModel(workflow.TaskSaveModel):
    def __init__(self):
        pass

    def save(self, model):
        pass

    def load(self):
        pass
