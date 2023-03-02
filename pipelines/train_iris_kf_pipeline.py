
from kfp import dsl
import kfp.components as components
from kubernetes.client.models import V1EnvVar
from kubernetes import client as k8s_client


def upload_iris_data():
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    from minio import Minio
    from sklearn.model_selection import train_test_split

    # Helper functions
    def load_iris_data():
        iris = datasets.load_iris()
        data = pd.DataFrame(
            {
                "sepal length": iris.data[:, 0],
                "sepal width": iris.data[:, 1],
                "petal length": iris.data[:, 2],
                "petal width": iris.data[:, 3],
                "species": iris.target,
            }
        )
        return data

    def load_minio():
        minio_client = Minio(
            "minio-service.mlops-demo-datascience.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False,
        )
        minio_bucket = "mlpipeline"

        return minio_client, minio_bucket

    ######

    # Get Data from Iris Data Set and push to Minio Storage
    iris_data = load_iris_data()
    minio_client, minio_bucket = load_minio()

    X = iris_data[
        ["sepal length", "sepal width", "petal length", "petal width"]
    ]  # Features
    y = iris_data["species"]  # Labels

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )  # 70% training and 30% test

    # save to numpy file, store in Minio
    np.save("/tmp/x_train.npy", x_train)
    minio_client.fput_object(minio_bucket, "x_train", "/tmp/x_train.npy")

    np.save("/tmp/y_train.npy", y_train)
    minio_client.fput_object(minio_bucket, "y_train", "/tmp/y_train.npy")

    np.save("/tmp/x_test.npy", x_test)
    minio_client.fput_object(minio_bucket, "x_test", "/tmp/x_test.npy")

    np.save("/tmp/y_test.npy", y_test)
    minio_client.fput_object(minio_bucket, "y_test", "/tmp/y_test.npy")


# %%
from typing import NamedTuple


def train_model() -> (
    NamedTuple(
        "Output",
        [("mlpipeline_ui_metadata", "UI_metadata"), ("mlpipeline_metrics", "Metrics")],
    )
):
    import pickle
    import boto3
    import json
    import os
    import numpy as np
    from minio import Minio
    from datetime import date
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score


    # HELPER FUNCTIONS
    def load_minio():
        minio_client = Minio(
            "minio-service.mlops-demo-datascience.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False,
        )
        minio_bucket = "mlpipeline"

        return minio_client, minio_bucket

    def load_model_into_s3(model, fileName, ak, sk):
        bucket_name = "obc-mlops-demo-datascience-iris-model"
        service_point = 'http://s3.openshift-storage.svc.cluster.local'
        s3client = boto3.client('s3',
                            'us-east-1', 
                            endpoint_url=service_point,
                            aws_access_key_id = ak,
                            aws_secret_access_key = sk,
                            use_ssl = True if 'https' in service_point else False,
                            verify = False )
        
        #s3client.upload_file(fileName, bucket_name, fileName)
        s3client.put_object(body=b'bytes'|model, bucket=bucket_name, key=fileName)


    # Create Model and Train
    minio_client, minio_bucket = load_minio()
    minio_client.fget_object(minio_bucket, "x_train", "/tmp/x_train.npy")
    x_train = np.load("/tmp/x_train.npy")

    minio_client.fget_object(minio_bucket, "y_train", "/tmp/y_train.npy")
    y_train = np.load("/tmp/y_train.npy")

    minio_client.fget_object(minio_bucket, "x_test", "/tmp/x_test.npy")
    x_test = np.load("/tmp/x_test.npy")

    minio_client.fget_object(minio_bucket, "y_test", "/tmp/y_test.npy")
    y_test = np.load("/tmp/y_test.npy")

    # Create a Gaussian Classifier
    model = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)


    # save the model to disk
    date = date.today()
    fileName = f'iris-model_{date}'
    #print("dumping model to local dir")
    #pickle.dump(model, open(fileName, 'wb')) 
    # Upload Model into S3
    ## Get creds from k8s secrets
    ak = os.environ["ak"]
    sk = os.environ["sk"]
    load_model_into_s3(model, fileName, ak, sk)
    
    
    # Output accuracy
    accuracy_score_metric = accuracy_score(y_test, y_pred)

    metrics = {
        "metrics": [
            {
                "name": "model_accuracy",
                "numberValue": float(accuracy_score_metric),
                "format": "PERCENTAGE",
            }
        ]
    }
    metadata = {"metadata": [{"placeholder_key": "placeholder_value"}]}

    from collections import namedtuple

    output = namedtuple("output", ["mlpipeline_ui_metadata", "mlpipeline_metrics"])
    return output(json.dumps(metadata), json.dumps(metrics))


# %%
component_upload_iris_data = components.create_component_from_func(
    upload_iris_data,
    base_image="image-registry.openshift-image-registry.svc:5000/mlops-demo-pipelines/iris-training",
)
component_train_model = components.create_component_from_func(
    train_model,
    base_image="image-registry.openshift-image-registry.svc:5000/mlops-demo-pipelines/iris-training",
)


# %%
@dsl.pipeline(name="iris-training-pipeline")
def iris_model_training():
    step1 = component_upload_iris_data()
    step2 = component_train_model()
    step2.add_env_variable(V1EnvVar(
                name="ak",
                value_from=k8s_client.V1EnvVarSource(secret_key_ref=k8s_client.V1SecretKeySelector(
                    name="iris-model",
                    key="AWS_ACCESS_KEY_ID"
                    )
                )
            )
    )
    step2.add_env_variable(V1EnvVar(
                name="sk",
                value_from=k8s_client.V1EnvVarSource(secret_key_ref=k8s_client.V1SecretKeySelector(
                    name="iris-model",
                    key="AWS_SECRET_ACCESS_KEY"
                    )
                )
            )
    )
    step2.after(step1)


# %%
from kfp_tekton.compiler import TektonCompiler

TektonCompiler().compile(iris_model_training, package_path="iris_model_training.yaml")
