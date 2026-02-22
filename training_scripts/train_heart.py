import pandas as pd
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\rsude\Desktop\HAL\MEDAI\DS\HRE\heart.csv")

# Split features & target
X = df.drop("target", axis=1)
y = df["target"]

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
pickle.dump(model, open("heart_model.pkl", "wb"))

print("✅ Heart model saved as heart_model.pkl")