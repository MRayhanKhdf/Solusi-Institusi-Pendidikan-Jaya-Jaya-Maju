# streamlit_status_prediksi.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# -------------------------------
# 1. Load & Preprocessing Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/data.csv", sep=';')
    # Pastikan kolom 'Status' ada
    if "Status" not in df.columns:
        st.error("❌ Kolom 'Status' tidak ditemukan di data.csv. Harap pastikan kolom ini tersedia.")
        st.stop()
    # Hapus baris yang kosong di kolom Status
    df = df[df["Status"].notna()]
    return df

df = load_data()

# -------------------------------
# 2. Sidebar Input Data
# -------------------------------
st.sidebar.title("🎓 Masukkan Data Mahasiswa")

def user_input_features():
    age = st.sidebar.slider("Umur saat masuk", 17, 60, 21)
    admission_grade = st.sidebar.slider("Admission Grade", 90.0, 200.0, 150.0)
    prev_grade = st.sidebar.slider("Previous Qualification Grade", 80.0, 200.0, 140.0)
    scholarship = st.sidebar.selectbox("Penerima Beasiswa?", [0, 1])
    debtor = st.sidebar.selectbox("Memiliki Tanggungan Utang?", [0, 1])
    tuition_paid = st.sidebar.selectbox("Biaya Terbayar?", [0, 1])

    units1_enrolled = st.sidebar.slider("Unit Semester 1 Enrolled", 0, 20, 10)
    units1_approved = st.sidebar.slider("Unit Semester 1 Approved", 0, 20, 8)
    eval1 = st.sidebar.slider("Evaluasi Semester 1", 0, 50, 30)
    grade1 = st.sidebar.slider("Grade Semester 1", 0.0, 20.0, 14.0)

    units2_enrolled = st.sidebar.slider("Unit Semester 2 Enrolled", 0, 20, 10)
    units2_approved = st.sidebar.slider("Unit Semester 2 Approved", 0, 20, 8)
    eval2 = st.sidebar.slider("Evaluasi Semester 2", 0, 50, 30)
    grade2 = st.sidebar.slider("Grade Semester 2", 0.0, 20.0, 14.0)

    data = {
        "Age_at_enrollment": age,
        "Admission_grade": admission_grade,
        "Previous_qualification_grade": prev_grade,
        "Scholarship_holder": scholarship,
        "Debtor": debtor,
        "Tuition_fees_up_to_date": tuition_paid,
        "Curricular_units_1st_sem_enrolled": units1_enrolled,
        "Curricular_units_1st_sem_approved": units1_approved,
        "Curricular_units_1st_sem_evaluations": eval1,
        "Curricular_units_1st_sem_grade": grade1,
        "Curricular_units_2nd_sem_enrolled": units2_enrolled,
        "Curricular_units_2nd_sem_approved": units2_approved,
        "Curricular_units_2nd_sem_evaluations": eval2,
        "Curricular_units_2nd_sem_grade": grade2
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# -------------------------------
# 3. Model Training & Prediction
# -------------------------------
st.title("🎯 Prediksi Status Mahasiswa")
st.write("Prediksi apakah mahasiswa akan **Dropout**, **Enrolled**, atau **Graduate** berdasarkan fitur akademik dan sosial.")

features = [
    "Age_at_enrollment", "Admission_grade", "Previous_qualification_grade",
    "Scholarship_holder", "Debtor", "Tuition_fees_up_to_date",
    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_grade"
]

# Pisahkan fitur dan label
X = df[features]
y = df["Status"]

# Imputasi nilai kosong (jika ada)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Model Extra Trees
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Prediksi untuk input user
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# -------------------------------
# 4. Output Prediksi
# -------------------------------
st.subheader("📌 Hasil Prediksi")
status_map = {"Graduate": "🎓 Lulus", "Dropout": "⚠️ Dropout", "Enrolled": "📚 Masih Terdaftar"}
st.write(f"**Status Mahasiswa yang Diprediksi:** {status_map[prediction[0]]}")

# Tampilkan Probabilitas
st.subheader("🔍 Probabilitas Kelas")
proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
st.dataframe(proba_df.T.rename(columns={0: "Probabilitas"}))

# -------------------------------
# 5. Eksplorasi Dataset (Opsional)
# -------------------------------
with st.expander("📊 Lihat Cuplikan Dataset"):
    st.dataframe(df.sample(10))
