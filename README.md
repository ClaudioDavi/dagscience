# DAGSCIENCE
#### Machine Learning Engineering Workflow Simplified

When building machine learning workflows I always end up using the same scaffolding for my projects. The data has always some flow that it has to go through before turning into a usable Machine Learning Model.

This soon-to-be package comes to help those in need of a new structure for their engineering workflows.

A Directed Acyclic Graph is a finite graph with no cycles in which data flows in only one direction. If you squint really hard, you can see that most model building in machine learning workflows follow the same rules, data-in/model-out with a lot of in-betweens.

This is a non-opinionated and non-invasive package that relies on pure python to work. I have set as a ground rule so that I can use it in my serverless functions where I always have to optimize for package size.

To use the package for now, you must clone it inside your repository and run: 
`pip install .`

From there is up to you. Here's an example to help you out:

```python
import os
import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sklearn
from sklearn.exceptions import DataConversionWarning


from dagscience import workflow
import joblib
    
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class TaskGetDataConcrete(workflow.TaskGetData):
    def __init__(self):
        pass

    def save(self, data, *args, **kwargs):
        data.to_csv("data.csv", index=False)
        return data

    def load_from_source(self):
        # Loads literacy rate dataset
        data = pd.read_csv("https://query.data.world/s/ohtb5dg6ik6pr7vvylm2yqtwaf5aqs")
        return self.save(data)
        

    def load_from_filesystem(self):
        if os.path.exists("data.csv"):
            return pd.read_csv("data.csv")
        else:
            raise FileNotFoundError()


class TaskPreprocessConcrete(workflow.TaskPreprocess):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, data):
        data.drop(["Country", "Name"], axis=1, inplace=True)
        data.iloc[:, [0, 2, 3, 4, 5]] = sklearn.preprocessing.scale(data.iloc[:, [0, 2, 3, 4, 5]])
        return data.sample(frac=1)


class TaskTrainConcrete(workflow.TaskTrain):
    def __init__(self, *args, **kwargs):
        pass

    def build_model(self, *args, **kwargs):
        return GradientBoostingRegressor()

    def run(self, model, data):
        X = data.iloc[:, [0, 2, 3, 4, 5]]
        model.fit(X=X, y=data["Literacy"])
        return model


class TaskSaveModelConcrete(workflow.TaskSaveModel):
    def __init__(self):
        pass

    def save(self, model):
        joblib.dump(model, "model")

    def load(self):
        if os.path.exists('model'):
            return joblib.load("model")
        else:
            raise FileNotFoundError()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    model = workflow.DagflowCycle(
        task_get_data=TaskGetDataConcrete(),
        task_preprocess=TaskPreprocessConcrete(),
        task_train=TaskTrainConcrete(),
        task_model_saver=TaskSaveModelConcrete(),
    ).run()
    print("Prediction of Literacy Rate: ", model.predict([[-0.466107, -0.445900,	0.794243, -0.257518, -0.714468]])[0])
```

## Roadmap:
* Re-execute step
* Tests
* Upload to pypi
* CI