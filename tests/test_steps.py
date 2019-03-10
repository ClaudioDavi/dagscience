from dagscience.step_manager import Step
from .mock_workflow import MockGetData, MockPreprocess, MockTrain, MockSaveModel
import os
import configparser


class TestStep():
    step = Step(MockGetData(), MockPreprocess(),
                MockTrain(), MockSaveModel())

    def test_step_writer(self):
        self.step.step_writer()
        assert os.path.exists('.steps')
        os.remove('.steps')

    def test_step_writer_sections(self):
        self.step.step_writer()
        sections = configparser.ConfigParser()
        sections.read('.steps')
        assert 'STEPS' in sections.sections()
        os.remove('.steps')
