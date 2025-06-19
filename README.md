# Experimental Design
This work is dedicated for experimental design with MLflow

## Installation
To install mlflow, it is simply run the following command in 
your virtual environment such as conda
```
$ pip install mlflow
```

## MLflow and a Local SQLite Database
To use MLflow with a local SQLite database, you need to set 
the environment variable MLFLOW_TRACKING_URI (e.g., sqlite:///mlruns.db). 
This will create a SQLite database file (mlruns.db) in the current directory.
Specify a different path if you want to store the database file in a different location.

```
$ export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# If you are in a notebook

%env MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```
## Logging the program

Now you are ready to start logging your experiment runs. For example, the following code runs training for a scikit-learn 
RandomForest model on the diabetes dataset:

```python

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

```

## View the results

To open the UI in the browser of a local machine, you can use SSH 
port forwarding. If you already have a VS Code window open, the embedded terminal 
automates port forwarding after you type the command. 

Once your training job finishes, you can run the following
command to launch the MLflow UI (You will have to specify the path to SQLite database file with --backend-store-uri option):

```bash
$mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```

Then, navigate to http://localhost:8080 in your browser to view the results.

