#  MedAI – Multi-Disease AI Screening Platform

##  Overview

MedAI is a multi-modal AI healthcare screening system that detects multiple diseases using both medical imaging and structured clinical data.

The platform integrates Deep Learning and Machine Learning models into a unified dashboard built using Streamlit.

---

##  Diseases Supported

| Disease | Input Type | Model Used |
|----------|------------|------------|
|  Brain Tumor | MRI Images | MobileNetV2 (CNN - Transfer Learning) |
|  Pneumonia | Chest X-ray Images | MobileNetV2 (CNN - Transfer Learning) |
|  Diabetes | Clinical CSV Data | XGBoost Classifier |
|  Heart Disease | Clinical CSV Data | XGBoost Classifier |
|  Kidney Disease | Mixed Clinical CSV Data | XGBoost + Label Encoding |

---

## Technologies Used

- **TensorFlow / Keras** – Deep Learning
- **MobileNetV2** – Transfer Learning for medical imaging
- **XGBoost** – Gradient Boosting for structured data
- **Scikit-learn** – Preprocessing & evaluation
- **Streamlit** – Web App framework
- **Pandas & NumPy** – Data processing
- **Matplotlib / Streamlit Charts** – Visualization

---

## System Architecture

- Image-based diseases → Convolutional Neural Networks
- Tabular diseases → Gradient Boosted Trees
- Integrated dashboard with:
  - Probability outputs
  - Risk level classification
  - Charts and metrics
  - CSV export functionality

---

## 📂 Project Structure
