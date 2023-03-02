# %%
import pickle
import boto3
import pandas as pd
import numpy as np
from datetime import date
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Uses the creds in ~/.aws/credentials
access_key = <access_key_from_object_bucket_claim>
secret_key = <secret_access_key_from_object_bucket_claim>
#endPoint= 's3.openshift-storage.svc:443' #this endpoint is not needed
service_point = 'http://s3.openshift-storage.svc.cluster.local'
bucketName = <name_of_bucket_from_object_bucket> #can be found in object bucket claim as well 
s3client = boto3.client('s3','us-east-1', endpoint_url=service_point,
                       aws_access_key_id = access_key,
                       aws_secret_access_key = secret_key,
                        use_ssl = True if 'https' in service_point else False,
                       verify = False)
# %%
iris = datasets.load_iris()
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
#data.head()

# %%
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

# %%
#Create a Gaussian Classifier
model=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
print("training model")
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# %%
#print(X_train)

# %%
model.predict([[5,3,1.6,0.2]])

# %%
# save the model to disk
date = date.today()
fileName = f'../temp_models/iris-model_{date}.pkl'
pickle.dump(model, open(fileName, 'wb')) 
print("dumping model to local dir")
print(fileName)
s3client.upload_file(fileName, bucketName, fileName)

# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
