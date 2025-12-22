import joblib
import pandas as pd
import os


class VisaPredictor:
    def __init__(self, model_name="Tuned_XGBoost"):
        self.model_path = f"data/{model_name}.joblib"
        self.columns_path = "data/model_columns.joblib"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Error: {model_name} not found.")

        self.model = joblib.load(self.model_path)
        self.model_columns = joblib.load(self.columns_path)

    def predict(self, input_data):
        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=self.model_columns, fill_value=0)

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]

        return "Certified" if prediction == 1 else "Denied", probability
