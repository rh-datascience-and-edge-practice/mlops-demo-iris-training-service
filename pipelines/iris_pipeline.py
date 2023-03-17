"""Example of a pipeline to demonstrate accessing secrets/config maps in a pipeline."""
import os
from typing import NamedTuple
import urllib

from dotenv import load_dotenv

import kfp

import kfp_tekton

import kubernetes

import pandas as pd

load_dotenv(override=True)

kubeflow_endpoint = os.environ["KUBEFLOW_ENDPOINT"]
bearer_token = os.environ["BEARER_TOKEN"]


def data_prep(
    X_train_file: kfp.components.OutputPath(),
    X_test_file: kfp.components.OutputPath(),
    y_train_file: kfp.components.OutputPath(),
    y_test_file: kfp.components.OutputPath(),
):
    from collections import namedtuple
    import pickle

    import pandas as pd

    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    def get_iris_data() -> pd.DataFrame:
        iris = datasets.load_iris()
        data = pd.DataFrame(
            {
                "sepalLength": iris.data[:, 0],
                "sepalWidth": iris.data[:, 1],
                "petalLength": iris.data[:, 2],
                "petalWidth": iris.data[:, 3],
                "species": iris.target,
            }
        )

        print("Initial Dataset:")
        print(data.head())

        return data

    def create_training_set(dataset: pd.DataFrame, test_size: float = 0.3):
        # Features
        X = dataset[["sepalLength", "sepalWidth", "petalLength", "petalWidth"]]
        # Labels
        y = dataset["species"]

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return X_train, X_test, y_train, y_test

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    dataset = get_iris_data()
    X_train, X_test, y_train, y_test = create_training_set(dataset)

    save_pickle(X_train_file, X_train)
    save_pickle(X_test_file, X_test)
    save_pickle(y_train_file, y_train)
    save_pickle(y_test_file, y_test)


def train_model(
    X_train_file: kfp.components.InputPath(),
    y_train_file: kfp.components.InputPath(),
    model_file: kfp.components.OutputPath(),
):
    import pickle

    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)

        return target_object

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    X_train = load_pickle(X_train_file)
    y_train = load_pickle(y_train_file)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    save_pickle(model_file, model)


def validate_model():
    pass


def evaluate_model(
    X_test_file: kfp.components.InputPath(),
    y_test_file: kfp.components.InputPath(),
    model_file: kfp.components.InputPath(),
    mlpipeline_metrics_file: kfp.components.OutputPath("Metrics"),
):
    import json
    import pickle

    from sklearn.metrics import accuracy_score

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)

        return target_object

    X_test = load_pickle(X_test_file)
    y_test = load_pickle(y_test_file)
    model = load_pickle(model_file)

    y_pred = model.predict(X_test)

    accuracy_score_metric = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy_score_metric}")

    metrics = {
        "metrics": [
            {
                "name": "accuracy-score",
                "numberValue": accuracy_score_metric,
                "format": "PERCENTAGE",
            },
        ]
    }

    with open(mlpipeline_metrics_file, "w") as f:
        json.dump(metrics, f)


def upload_model():
    pass


data_prep_op = kfp.components.create_component_from_func(
    data_prep,
    base_image="image-registry.openshift-image-registry.svc:5000/openshift/python:latest",
    packages_to_install=["pandas", "scikit-learn"],
)

train_model_op = kfp.components.create_component_from_func(
    train_model,
    base_image="image-registry.openshift-image-registry.svc:5000/openshift/python:latest",
    packages_to_install=["pandas", "scikit-learn"],
)

evaluate_model_op = kfp.components.create_component_from_func(
    evaluate_model,
    base_image="image-registry.openshift-image-registry.svc:5000/openshift/python:latest",
    packages_to_install=["pandas", "scikit-learn"],
)


@kfp.dsl.pipeline(
    name="Iris Pipeline",
)
def iris_pipeline():
    data_prep_task = data_prep_op()
    train_model_task = train_model_op(
        data_prep_task.outputs["X_train"],
        data_prep_task.outputs["y_train"],
    )
    evaluate_model_task = evaluate_model_op(
        data_prep_task.outputs["X_test"],
        data_prep_task.outputs["y_test"],
        train_model_task.output,
    )


if __name__ == "__main__":
    client = kfp_tekton.TektonClient(
        host=urllib.parse.urljoin(kubeflow_endpoint, "/pipeline"),
        existing_token=bearer_token,
    )
    client.create_run_from_pipeline_func(
        iris_pipeline, arguments={}, experiment_name="iris"
    )
