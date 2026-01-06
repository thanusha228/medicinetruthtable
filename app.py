import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------ Load model artifacts ------------------
model = pickle.load(open("adherence_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Encoders
med_enc = pickle.load(open("Medicine_Name_encoder.pkl", "rb"))
cond_enc = pickle.load(open("Condition_encoder.pkl", "rb"))
gender_enc = pickle.load(open("Gender_encoder.pkl", "rb"))

# ------------------ UI ------------------
st.set_page_config(page_title="Medicine Truth Label AI", layout="centered")
st.title("🩺 Medicine Truth Label AI")

st.subheader("Enter your details")

age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
condition = st.text_input("Condition (e.g., Diabetes, Allergy)")
medicine = st.text_input("Medicine Name")

st.subheader("Medicine details (if known)")
side_effect = st.slider("Side Effect Severity (1 = low, 5 = high)", 1, 5, 3)
price = st.number_input("Approx Price (INR)", 10, 2000, 200)
dosage = st.number_input("Dosage (mg)", 50, 1000, 500)
effectiveness = st.slider("Effectiveness Score (1–10)", 1, 10, 7)
chronic = st.selectbox("Chronic Use?", ["No", "Yes"])

# ------------------ Analyze ------------------
if st.button("Analyze Medicine"):

    try:
        # Safe encoding (prevents crash on unseen labels)
        med_val = med_enc.transform([medicine])[0] if medicine in med_enc.classes_ else np.mean(med_enc.transform(med_enc.classes_))
        cond_val = cond_enc.transform([condition])[0] if condition in cond_enc.classes_ else np.mean(cond_enc.transform(cond_enc.classes_))
        gender_val = gender_enc.transform([gender])[0]

        input_row = {
            "Medicine_Name": med_val,
            "Condition": cond_val,
            "Age": age,
            "Gender": gender_val,
            "Side_Effect_Severity": side_effect,
            "Price_INR": price,
            "Dosage_mg": dosage,
            "Effectiveness_Score": effectiveness,
            "Chronic_Use": 1 if chronic == "Yes" else 0
        }

        df_input = pd.DataFrame([input_row])[features]
        scaled = scaler.transform(df_input)

        # Predict probability
        prob = model.predict_proba(scaled)[0][1]
        adherence_score = round(prob * 100, 1)

        # ------------------ Output ------------------
        st.subheader("🧪 Adherence Prediction")

        if adherence_score >= 70:
            st.success(f"Adherence Score: {adherence_score}/100 → Good adherence")
            st.markdown("### ✅ Recommendations")
            st.write("""
            • Continue taking medicine as prescribed  
            • Maintain a healthy diet and regular routine  
            • Stay hydrated and monitor mild side effects  
            • Do not skip doses even if symptoms improve  
            • Regular follow-ups are sufficient  
            """)
        elif adherence_score >= 40:
            st.warning(f"Adherence Score: {adherence_score}/100 → Moderate adherence (monitor)")
            st.markdown("### ⚠️ Recommendations")
            st.write("""
            • Monitor side effects closely  
            • Set reminders for doses  
            • Avoid self-adjusting dosage  
            • Consider lifestyle improvements (diet, exercise)  
            • Consult doctor if discomfort increases  
            """)
        else:
            st.error(f"Adherence Score: {adherence_score}/100 → Poor adherence (High risk)")
            st.markdown("### 🚨 Recommendations")
            st.write("""
            • High chance of stopping medicine early  
            • Consult doctor before continuing  
            • Discuss alternative medicines with lower side effects  
            • Do not ignore severe side effects  
            • Avoid alcohol and self-medication  
            """)

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")

# ------------------ Disclaimer ------------------
st.warning(
    "⚠ This tool is for educational purposes only. "
    "Always consult a certified medical professional."
)
