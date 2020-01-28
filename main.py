import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import r2_score



with mlflow.start_run() as run:
    df = pd.read_csv("data/train.csv")
    df["x1"] = df["x1"].astype("category")

    dtrain = lgb.Dataset(df[["x1"]], df["y"])

    params= {
       'num_leaves': 10,
       'n_jobs': -1,
       'max_depth': 2,
       'learning_rate': 0.1,
    }
    model = lgb.train(params=params, train_set=dtrain)

    df_test = pd.read_csv("data/test.csv")
    df_test["x1"] = df_test["x1"].astype("category")
    y_pred = model.predict(data=df_test[["x1"]])

    r2 = r2_score(df_test["y"], y_pred)
    print(r2)

    mlflow.lightgbm.log_model(model, "model")
    mlflow.log_metric("r2", r2)
    run_id = run.to_dictionary()["info"]["run_id"]
    model_uri = "runs:/" + run_id + "/model"
    print(model_uri)