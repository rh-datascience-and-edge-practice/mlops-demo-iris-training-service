import kfp
from kfp import dsl
from kfp import compiler
import kfp.components as components

from kfp_tekton.compiler import TektonCompiler

def train_model():
    pass

component_train_model = components.create_component_from_func(train_model,base_image="image-registry.openshift-image-registry.svc:5000/mlops-demo-datascience/jupyter-datascience-notebook:py3.8-v1")

@dsl.pipeline(name='iris-training-pipeline')
def iris_training_pipeline():
    step1 = component_train_model()

TektonCompiler().compile(iris_training_pipeline, package_path='iris_model_training.yaml')
