import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class utils_tools:

    def persist_model(model, path):
        print(f"Persisting the model to {path}")
        model_dir = os.path.dirname(path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(path, "wb") as file:
            pickle.dump(model, file)
        
        return print(f"Done")

    def load_model(path):
        print(f"Loading the model from {path}")
        with open(path, "rb") as file:
            model = pickle.load(file)
        print(f"Done")
        return model
    
    
    def model_pipeline (cat_features, num_features):
        column_transformer = ColumnTransformer([
            ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
            ('scaling', StandardScaler(), num_features)]
        )

        pipeline = Pipeline(steps=[
            ('ohe_and_scaling', column_transformer),
            ('regression', Ridge())
        ])
        
        return pipeline

    
    def predict (model, data):
       y_pred = model.predict(data)
       return y_pred