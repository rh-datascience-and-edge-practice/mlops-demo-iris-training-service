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
