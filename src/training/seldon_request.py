# %% [markdown]
# # Iris Machine Learning Requests
#
# This notebook performs a simple API call to the ML Service and receives an inference result back.
#
# The model is based on the [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) which includes data samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, we can predict which species of Iris the flower belongs.
#
# | Species | Value |
# |---------|-------|
# |Iris-setosa | 0 |
# |Iris-versicolor | 1 |
# |Iris-virginica | 2 |

# %% [markdown]
# ### Install Requirements
#
# Install requests package if it is not already available
import json

from urllib.parse import urljoin

import requests

# %% [markdown]
# ### Request Helper Functions
#
# Load a few helper functions to help interpret the response of the ML Service

# %%
def get_iris_classification(response):
    response_obj = json.loads(response)
    inference_result = response_obj["data"]["ndarray"][0]

    return inference_result


def get_iris_species(classification):
    if classification == 0:
        species = "Iris-setosa"
    elif classification == 1:
        species = "Iris-versicolor"
    elif classification == 2:
        species = "Iris-virginica"

    return species


# %% [markdown]
# ## ML Service Request Examples

# %% [markdown]
# The following contain example requests submitted to the ML Model Service.  Each requests sends the model the data for the four Iris features and the model returns a numeric value representing the result of the species inference.

# %% [markdown]
# Construct the URL for the API call and setup other header values needed for the request.

# %%
# Example Route URL if connecting outside of the cluster
# base_url = "https://seldon-iris-seldon-demo.apps.cluster-kmmjn.kmmjn.sandbox1150.opentlc.com"

base_url = "http://localhost:9000"

# Example service URL when connecting internally on the cluster
# base_url = "http://iris-seldon-example:8000"
predict_url = urljoin(base_url, "/api/v1.0/predictions")
headers = {"Content-Type": "application/json"}

# %% [markdown]
# #### Request Example #1
#
# This request should return a prediction of `0` indicating that this species is `Iris-setosa`.

# %%
X = [[5, 3, 1.6, 0.2]]
columns = ["sepal length", "sepal width", "petal length", "petal width"]

data_obj = json.dumps({"data": {"names": columns, "ndarray": X}})

try:
    r = requests.post(predict_url, headers=headers, data=data_obj)

    classification = get_iris_classification(r.text)
    species = get_iris_species(classification)

    print(f"Classification Prediction: {classification}")
    print(f"Species Prediction: {species}")
    print(f"Http Status Code: {r.status_code}")
    print(f"Raw Json response: {r.text}")
except:
    print("Connection error")

# %% [markdown]
# #### Request Example #2
#
# This request should return a prediction of `1` indicating that this species is `Iris-versicolor`.

# %%
X = [[5.9, 3.0, 5.1, 1.8]]

columns = ["sepal length", "sepal width", "petal length", "petal width"]

data_obj = json.dumps({"data": {"names": columns, "ndarray": X}})

try:
    r = requests.post(predict_url, headers=headers, data=data_obj)

    classification = get_iris_classification(r.text)
    species = get_iris_species(classification)

    print(f"Classification Prediction: {classification}")
    print(f"Species Prediction: {species}")
    print(f"Http Status Code: {r.status_code}")
    print(f"Raw Json response: {r.text}")
except:
    print("Connection error")

# %% [markdown]
# #### Request Example #3
#
# This request should return a prediction of `2` indicating that this species is `Iris-virginica`.

# %%
X = [[7.2, 3.6, 6.1, 2.5]]

columns = ["sepal length", "sepal width", "petal length", "petal width"]

data_obj = json.dumps({"data": {"names": columns, "ndarray": X}})

try:
    r = requests.post(predict_url, headers=headers, data=data_obj)

    classification = get_iris_classification(r.text)
    species = get_iris_species(classification)

    print(f"Classification Prediction: {classification}")
    print(f"Species Prediction: {species}")
    print(f"Http Status Code: {r.status_code}")
    print(f"Raw Json response: {r.text}")
except:
    print("Connection error")
