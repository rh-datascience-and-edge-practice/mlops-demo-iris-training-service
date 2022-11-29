# %%
import pickle

import pandas as pd

import numpy as np

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

# %%
iris = datasets.load_iris()
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()

# %%
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01) # 70% training and 30% test

# %%
#Create a Gaussian Classifier
model=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# %%
print(X_train)

# %%
model.predict([[5,3,1.6,0.2]])

# %%
# save the model to disk
filename = '../models/iris-model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
