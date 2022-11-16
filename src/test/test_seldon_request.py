import pytest
import os
from training.seldon_request import get_iris_species


def test_get_iris_species():
    assert get_iris_species(0) == "Iris-setosa"
    assert get_iris_species(1) == "Iris-versicolor"
    assert get_iris_species(2) == "Iris-virginica"
