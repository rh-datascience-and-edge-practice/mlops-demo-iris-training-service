# mlops-demo-iris-training-service

## Running locally

### Installing pipenv

Pipenv is a package and virtual environment management tool built on pip and pythyon virtual environments.

```
python -m pip install pipenv
```

### Setting up the Python Virtual Environment

```
# Install the required packages with dev dependencies
pipenv install --dev

# Activate the python virtual environment
pipenv shell
```

### Running Pytest

```
pytest tests/
```

### Running Automated Formatting

Black is an opinionated automated code formatting tool for Python.  Black can help to automatically resolve many pep8 issues.

```
black .
```

### Running Linting

Flake8 is a python pep8 lint tool to help identify common pep8 issues.  Flake8 also has several extensions enabled such as `bandit`, a static code analysis tool for identifying security issues.

```
flake8 .
```

### OCP Pipelines Integration

OCP Pipeline for Iris training is set up manually at the moment. 

1. In the "mlops-demo-datascience" namespace, navigate to _Routes -> ds-pipeline-ui_. This is the UI for managing OCP Pipelines
```
oc project mlops-demo-datascience
oc get routes
```

2. Expose MinIO Service:
```
oc expose svc/minio-service
```

3. Run:
```
python3 iris_model_training/train_iris_kf_pipeline.py
```

4. This generates the Pipeline YAML using the pipeline components defined in the script above. In the OCP Pipelines UI from Step 1 upload the generated "iris_model_training.yaml". Runs and experiments can be executed here.



TODO:
- Automate MinIO route creation - add to mlops config
- Modify train_iris_kf_pipeline.py to create pipeline directly from script instead of generating and manually uploading pipeline yaml
- Update bootstrap.sh to run train_iris_kf_pipeline.py