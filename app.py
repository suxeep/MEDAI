import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os

st.set_page_config(page_title="MedAI - Multi Disease System", layout="wide")

# ===============================
# Styling
# ===============================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stButton>button {
    background-color: #6C63FF;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #574fd6;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 MedAI - Multi Disease Early Screening System")
st.markdown("AI-powered detection for Brain Tumor, Pneumonia, Diabetes, Heart & Kidney Disease")

# ==========================================
# Load Models
# ==========================================
@st.cache_resource


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():

    pneumonia_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "pneumonia_model.h5")
    )

    brain_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "brain_tumor_model.h5")
    )

    diabetes_model = pickle.load(
        open(os.path.join(MODEL_DIR, "diabetes_model.pkl"), "rb")
    )

    heart_model = pickle.load(
        open(os.path.join(MODEL_DIR, "heart_model.pkl"), "rb")
    )

    kidney_model = pickle.load(
        open(os.path.join(MODEL_DIR, "kidney_model.pkl"), "rb")
    )

    kidney_encoders = pickle.load(
        open(os.path.join(MODEL_DIR, "kidney_encoders.pkl"), "rb")
    )

    return pneumonia_model, brain_model, diabetes_model, heart_model, kidney_model, kidney_encoders


pneumonia_model, brain_model, diabetes_model, heart_model, kidney_model, kidney_encoders = load_models()

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("Navigation")

disease = st.sidebar.radio(
    "Select Module",
    [
        "Brain Tumor Detection",
        "Pneumonia Detection",
        "Diabetes Prediction (CSV Upload)",
        "Heart Disease Prediction (CSV Upload)",
        "Kidney Disease Prediction (CSV Upload)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("MedAI 🚀")

# =========================================================
# 🧠 BRAIN TUMOR
# =========================================================
if disease == "Brain Tumor Detection":

    st.header("🧠 Brain Tumor Detection - MRI Dashboard")

    files = st.file_uploader(
        "Upload MRI Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files:
        if st.button("Submit for Brain Tumor Prediction"):

            results = []
            progress = st.progress(0)

            for i, file in enumerate(files):
                img = Image.open(file).convert("RGB").resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = brain_model.predict(img_array)[0][0]
                label = "Tumor" if prediction > 0.5 else "No Tumor"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                results.append({
                    "File Name": file.name,
                    "Prediction": label,
                    "Confidence": round(float(confidence), 3)
                })

                progress.progress((i + 1) / len(files))

            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            col1, col2 = st.columns(2)
            tumor_count = sum(r["Prediction"] == "Tumor" for r in results)

            with col1:
                st.metric("Total Images", len(results))
            with col2:
                st.metric("Tumor Detected", tumor_count)

            st.subheader("Prediction Distribution")
            st.bar_chart(df_results["Prediction"].value_counts())

            st.subheader("Confidence Levels")
            st.bar_chart(df_results.set_index("File Name")["Confidence"])

            st.download_button(
                "Download Results",
                df_results.to_csv(index=False),
                "brain_tumor_results.csv",
                "text/csv"
            )

# =========================================================
# 🫁 PNEUMONIA
# =========================================================
elif disease == "Pneumonia Detection":

    st.header("🫁 Pneumonia Detection Dashboard")

    files = st.file_uploader(
        "Upload X-ray Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files:
        if st.button("Submit for Pneumonia Prediction"):

            results = []
            progress = st.progress(0)

            for i, file in enumerate(files):
                img = Image.open(file).convert("RGB").resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = pneumonia_model.predict(img_array)[0][0]
                label = "Pneumonia" if prediction > 0.5 else "Normal"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                results.append({
                    "File Name": file.name,
                    "Prediction": label,
                    "Confidence": round(float(confidence), 3)
                })

                progress.progress((i + 1) / len(files))

            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            col1, col2 = st.columns(2)
            pneumonia_count = sum(r["Prediction"] == "Pneumonia" for r in results)

            with col1:
                st.metric("Total Images", len(results))
            with col2:
                st.metric("Pneumonia Detected", pneumonia_count)

            st.bar_chart(df_results["Prediction"].value_counts())
            st.bar_chart(df_results.set_index("File Name")["Confidence"])

            st.download_button(
                "Download Results",
                df_results.to_csv(index=False),
                "pneumonia_results.csv",
                "text/csv"
            )

# =========================================================
# 🍬 DIABETES
# =========================================================
elif disease == "Diabetes Prediction (CSV Upload)":

    st.header("🍬 Diabetes Risk Prediction")

    uploaded_csv = st.file_uploader("Upload Patient CSV", type=["csv"])

    if uploaded_csv:
        if st.button("Submit for Diabetes Prediction"):

            df = pd.read_csv(uploaded_csv)

            required_columns = [
                "Pregnancies","Glucose","BloodPressure",
                "SkinThickness","Insulin","BMI",
                "DiabetesPedigreeFunction","Age"
            ]

            if not all(col in df.columns for col in required_columns):
                st.error("CSV format incorrect.")
                st.stop()

            X = df[required_columns]
            prob = diabetes_model.predict_proba(X)[:, 1]

            df["Diabetes Risk"] = prob
            df["Risk Level"] = df["Diabetes Risk"].apply(
                lambda x: "High 🔴" if x > 0.7 else
                          "Medium 🟡" if x > 0.4 else
                          "Low 🟢"
            )

            st.dataframe(df)
            st.bar_chart(df["Risk Level"].value_counts())

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "diabetes_results.csv",
                "text/csv"
            )

# =========================================================
# 🫀 HEART
# =========================================================
elif disease == "Heart Disease Prediction (CSV Upload)":

    st.header("🫀 Heart Disease Risk Prediction")

    uploaded_csv = st.file_uploader("Upload Patient CSV", type=["csv"])

    if uploaded_csv:
        if st.button("Submit for Heart Prediction"):

            df = pd.read_csv(uploaded_csv)

            required_columns = [
                "age","sex","cp","trestbps","chol",
                "fbs","restecg","thalach","exang",
                "oldpeak","slope","ca","thal"
            ]

            if not all(col in df.columns for col in required_columns):
                st.error("CSV format incorrect.")
                st.stop()

            X = df[required_columns]
            prob = heart_model.predict_proba(X)[:, 1]

            df["Heart Risk"] = prob
            df["Risk Level"] = df["Heart Risk"].apply(
                lambda x: "High 🔴" if x > 0.7 else
                          "Medium 🟡" if x > 0.4 else
                          "Low 🟢"
            )

            st.dataframe(df)
            st.bar_chart(df["Risk Level"].value_counts())

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "heart_results.csv",
                "text/csv"
            )

# =========================================================
# 🧪 KIDNEY
# =========================================================
elif disease == "Kidney Disease Prediction (CSV Upload)":

    st.header("🧪 Chronic Kidney Disease Prediction")

    uploaded_csv = st.file_uploader("Upload Kidney CSV", type=["csv"])

    if uploaded_csv:
        if st.button("Submit for Kidney Prediction"):

            df = pd.read_csv(uploaded_csv)

            # Clean missing values
            df.replace("?", np.nan, inplace=True)
            df = df.fillna(df.mode().iloc[0])

            if "id" in df.columns:
                df = df.drop("id", axis=1)

            # Safe encoding
            for col in kidney_encoders:
                if col in df.columns:
                    le = kidney_encoders[col]

                    # Handle unseen labels
                    df[col] = df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )

                    df[col] = le.transform(df[col])

           
            # Drop target column if present
                if "classification" in df.columns:
                    df = df.drop("classification", axis=1)

            probabilities = kidney_model.predict_proba(df)[:, 1]
            
            
            
            df["Kidney Disease Risk"] = probabilities
            df["Risk Level"] = df["Kidney Disease Risk"].apply(
                lambda x: "High 🔴" if x > 0.7 else
                          "Medium 🟡" if x > 0.4 else
                          "Low 🟢"
            )

            st.dataframe(df)

            st.subheader("Risk Distribution")
            st.bar_chart(df["Risk Level"].value_counts())

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "kidney_results.csv",
                "text/csv"
            )


            st.success("Kidney Prediction Completed ✔")



