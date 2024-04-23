import mlflow
logged_model = 'runs:/2192e50132744b09a33d72c64cb5429c/RandomForestClassifier'

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