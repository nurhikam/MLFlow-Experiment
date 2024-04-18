import mlflow
mlflow.set_tracking_uri('http://localhost:5000')

exp_id = mlflow.get_tracking_uri('Loan_Prediction')

with mlflow.start_run(run_name='DecisionTreeClass') as run:
    mlflow.set_tag("version", '1.0.0')

mlflow.end_run()

n_estimator=10
criterion='gini'
mlflow.log_param('n_estimator', n_estimator)
mlflow.log_param('criterion', criterion)
mlflow.log_metric('accuaacy', 0.9)
