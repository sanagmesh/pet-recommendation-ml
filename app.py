import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Pet Recommendation System")

# Sample dataset
data = {
    "space": ["apartment","house","apartment","house"],
    "time": ["low","high","medium","high"],
    "budget": ["low","high","medium","high"],
    "experience": ["beginner","experienced","beginner","experienced"],
    "pet": ["fish","dog","cat","dog"]
}

df = pd.DataFrame(data)

# Encoding
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("pet", axis=1)
y = df["pet"]

model = DecisionTreeClassifier()
model.fit(X, y)

# Inputs
space = st.selectbox("Space", ["apartment","house"])
time = st.selectbox("Time", ["low","medium","high"])
budget = st.selectbox("Budget", ["low","medium","high"])
experience = st.selectbox("Experience", ["beginner","experienced"])

# Prediction
if st.button("Predict"):
    sample = [[space, time, budget, experience]]

    encoded = []
    for i, col in enumerate(["space","time","budget","experience"]):
        encoded.append(le_dict[col].transform([sample[0][i]])[0])

    pred = model.predict([encoded])
    result = le_dict["pet"].inverse_transform(pred)

    st.success(f"Recommended Pet: {result[0]}")