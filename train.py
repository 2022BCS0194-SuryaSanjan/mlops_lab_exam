import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

df = pd.read_csv("Dataset/winequality-red.csv", sep=";")

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

metrics = {
    'mse': float(mse),
    'r2': float(r2)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved successfully!")
