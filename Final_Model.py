import joblib
import pandas as pd
model_path = "saved_models/isolation_forest_model.joblib"
model = joblib.load(model_path)
new_data = pd.read_csv("new_transactions.csv")

predictions = model.predict(new_data)
anomalies = (predictions == -1).astype(int)

print("Anomaly predictions (1 = fraud/anomaly):")
print(anomalies)