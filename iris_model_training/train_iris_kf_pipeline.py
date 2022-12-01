# %%
import pickle
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import kfp
from kfp import dsl
from kfp import compiler
import kfp.components as components
from minio import Minio

# %%
def load_minio():
    minio_client = Minio(
        "minio-service.mlops-demo-datascience.svc.cluster.local:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"

    return minio_client, minio_bucket;

# %%
def load_iris_data():
    iris = datasets.load_iris()
    data=pd.DataFrame({
        'sepal length':iris.data[:,0],
        'sepal width':iris.data[:,1],
        'petal length':iris.data[:,2],
        'petal width':iris.data[:,3],
        'species':iris.target
    })
    data.head()
    return data

# %%
def upload_iris_data():

    from sklearn import datasets
    import pandas as pd
    import numpy as np
    from minio import Minio
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier


    def load_iris_data():
        iris = datasets.load_iris()
        data=pd.DataFrame({
            'sepal length':iris.data[:,0],
            'sepal width':iris.data[:,1],
            'petal length':iris.data[:,2],
            'petal width':iris.data[:,3],
            'species':iris.target
        })
        data.head()
        return data
        
    def load_minio():
        minio_client = Minio(
            "minio-service.mlops-demo-datascience.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )
        minio_bucket = "mlpipeline"

        return minio_client, minio_bucket;

    # Get Data from Iris Data Set and push to Minio Storage
    iris_data = load_iris_data()
    minio_client, minio_bucket = load_minio()

    X=iris_data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
    y=iris_data['species']  # Labels

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01) # 70% training and 30% test
    load_minio()
    # save to numpy file, store in Minio
    np.save("/tmp/x_train.npy",x_train)
    minio_client.fput_object(minio_bucket,"x_train","/tmp/x_train.npy")

    np.save("/tmp/y_train.npy",y_train)
    minio_client.fput_object(minio_bucket,"y_train","/tmp/y_train.npy")

    np.save("/tmp/x_test.npy",x_test)
    minio_client.fput_object(minio_bucket,"x_test","/tmp/x_test.npy")

    np.save("/tmp/y_test.npy",y_test)
    minio_client.fput_object(minio_bucket,"y_test","/tmp/y_test.npy")

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

# %%
from typing import NamedTuple

def train_model() -> NamedTuple('Output', [('mlpipeline_ui_metadata', 'UI_metadata'),('mlpipeline_metrics', 'Metrics')]):

    import numpy as np
    from minio import Minio
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import json
        
    def load_minio():
        minio_client = Minio(
            "minio-service.mlops-demo-datascience.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )
        minio_bucket = "mlpipeline"

        return minio_client, minio_bucket;

    # Create Model and Train
    minio_client, minio_bucket = load_minio()
    minio_client.fget_object(minio_bucket,"x_train","/tmp/x_train.npy")
    x_train = np.load("/tmp/x_train.npy")

    minio_client.fget_object(minio_bucket,"y_train","/tmp/y_train.npy")
    y_train = np.load("/tmp/y_train.npy")

    minio_client.fget_object(minio_bucket,"x_test","/tmp/x_test.npy")
    x_test = np.load("/tmp/x_test.npy")

    minio_client.fget_object(minio_bucket,"y_test","/tmp/y_test.npy")
    y_test = np.load("/tmp/y_test.npy")

    #Create a Gaussian Classifier
    model=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)


    print(x_train)
    model.predict([[5,3,1.6,0.2]])

    filename = '/tmp/iris-model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Upload model to minio
    minio_client.fput_object(minio_bucket, "iris-model", filename)

    # Output metrics

    confusion_matrix_metric = confusion_matrix(y_test, y_pred)
    accuracy_score_metric = accuracy_score(y_test, y_pred)
    classification_report_metric = classification_report(y_test,y_pred)
    
    metrics = {
      'metrics': [{
          'name': 'model_accuracy',
          'numberValue':  float(accuracy_score_metric),
          'format' : "PERCENTAGE"
        }]}
    metadata = {
      'metadata': [{
          'placeholder_key': 'placeholder_value'
        }]}

    from collections import namedtuple
    output = namedtuple('output', ['mlpipeline_ui_metadata', 'mlpipeline_metrics'])
    return output(json.dumps(metadata),json.dumps(metrics))

# %%
component_upload_iris_data = components.create_component_from_func(upload_iris_data,base_image="image-registry.openshift-image-registry.svc:5000/mlops-demo-datascience/s2i-generic-data-science-notebook:v0.0.5")
component_train_model = components.create_component_from_func(train_model,base_image="image-registry.openshift-image-registry.svc:5000/mlops-demo-datascience/s2i-generic-data-science-notebook:v0.0.5")

# %%
@dsl.pipeline(name='iris-training-pipeline')
def iris_model_training():
    step1 = component_upload_iris_data()
    step2 = component_train_model()
    step2.after(step1)

# %%
from kfp_tekton.compiler import TektonCompiler

TektonCompiler().compile(iris_model_training, package_path='iris_model_training.yaml')

# %%
# if __name__ == "__main__":
#     client = kfp.Client(host='ds-pipeline.mlops-demo-datascience.svc.cluster.local:8888')

#     run_directly = 1
    
#     if (run_directly == 1):
#         client.create_run_from_pipeline_func(iris_model_training,arguments=arguments,experiment_name="mlops-demo")
#     else:
#         kfp.compiler.Compiler().compile(iris_model_training,package_path='iris_model_training.yaml')
#         client.upload_pipeline_version(pipeline_package_path='iris_model_training.yaml',pipeline_version_name="0.4",pipeline_name="iris training")


