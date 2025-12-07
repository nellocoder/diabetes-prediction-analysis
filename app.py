import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']
    df = pd.read_csv(url, names=col_names)
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())
    return df

df = load_data()

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.title("🩺 Diabetes Risk Predictor")
st.write(f"This app uses Machine Learning to predict the likelihood of diabetes based on diagnostic measures. Model Accuracy: **{accuracy:.2%}**")

st.sidebar.header("Patient Data Input")

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=3, step=1)
    glucose = st.sidebar.number_input('Glucose Level', min_value=0, max_value=200, value=120, step=1)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=140, value=70, step=1)
    skin = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=20, step=1)
    insulin = st.sidebar.number_input('Insulin Level', min_value=0, max_value=900, value=79, step=1)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.000, max_value=3.000, value=0.372, step=0.001, format="%.3f")
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=29, step=1)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DPF': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Patient Details")
st.write(input_df)

if st.button("Predict Risk"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result")

    risk_score = probability[0][1]

    if risk_score > 0.5:
        st.error(f"⚠️ High Risk of Diabetes detected. (Probability: {risk_score:.2%})")
    else:
        st.success(f"✅ Low Risk. Healthy profile. (Probability: {risk_score:.2%})")

    st.write("---")
    st.write("### How this patient compares to the dataset:")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True, element="step", palette='seismic', ax=ax)
    plt.axvline(input_df['Glucose'][0], color='green', linestyle='--', linewidth=3, label='Patient Glucose')
    plt.legend()
    st.pyplot(fig)
