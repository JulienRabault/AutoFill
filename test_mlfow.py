import os
import mlflow


mlflow.set_tracking_uri("https://mlflowts.irit.fr")
mlflow.set_experiment("TESTOCCIDATA")

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.89)
