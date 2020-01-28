import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer



with mlflow.start_run() as run:
    df = pd.read_csv("data/train.csv")
    all_features = ["x1"]
    categorical_features = ["x1"]
    target = "y"

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', RandomForestRegressor())])

    clf.fit(df[all_features], df[target])


    df_test = pd.read_csv("data/test.csv")
    y_pred = clf.predict(df_test[categorical_features])

    r2 = r2_score(df_test["y"], y_pred)
    print(r2)

    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_metric("r2", r2)
    run_id = run.to_dictionary()["info"]["run_id"]
    model_uri = "runs:/" + run_id + "/model"
    print(model_uri)