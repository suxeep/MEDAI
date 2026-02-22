import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\\Users\\rsude\Desktop\\HAL\\MEDAI\DS\\KD\\kidney_disease.csv")

# Drop id column if exists
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Replace missing values
df.replace("?", np.nan, inplace=True)
df = df.fillna(df.mode().iloc[0])

# Encode categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split features & target
X = df.drop("classification", axis=1)
y = df["classification"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Accuracy check
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model + encoders
pickle.dump(model, open("kidney_model.pkl", "wb"))
pickle.dump(label_encoders, open("kidney_encoders.pkl", "wb"))

print("✅ Kidney model saved successfully!")