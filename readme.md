1. Run the model and save with MLflow
python pipeline.py

2. Launch the MLflow models
mlflow models serve --model-uri runs:/71a633a9c53f4d9984f1475169e33d79/model


3. Query with one new datapoint
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{"columns":["x1"],"data":[["a"]]}'

