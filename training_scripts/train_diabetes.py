import pandas as pd
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\rsude\Desktop\HAL\MEDAI\DS\DIA\diabetes.csv")

# Split features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X, y)

# Save model
pickle.dump(model, open("diabetes_model.pkl", "wb"))

print("✅ Diabetes model saved as diabetes_model.pkl")