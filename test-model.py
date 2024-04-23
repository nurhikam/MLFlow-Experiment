import mlflow
logged_model = 'runs:/86c44dfd142040a8b8977f5ea10a2771/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = [[
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.98745,
            360.0,
            1.0,
            2.0,
            8.698
            ]]

print(f"prediction is: {loaded_model.predict(pd.DataFrame(data))}")