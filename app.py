import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Predictor")

@st.cache_resource
def train():
    df = pd.read_csv("car data.csv").drop_duplicates().dropna()
    df["Car_Age"] = 2026 - df["Year"]
    df = df.drop(["Year", "Car_Name"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]
    model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
    return model, X.columns

model, cols = train()

present = st.number_input("Showroom price (lakh)", 0.0, 100.0, 8.0)
kms = st.number_input("Kms driven", 0, 500000, 40000, step=1000)
age = st.slider("Car age (years)", 0, 20, 5)
fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller", ["Dealer", "Individual"])
trans = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Previous owners", [0, 1, 3])

if st.button("Predict Price"):
    row = pd.DataFrame(0.0, index=[0], columns=cols)
    row["Present_Price"] = present
    row["Kms_Driven"] = kms
    row["Car_Age"] = age
    row["Owner"] = owner
    for name, val in [("Fuel_Type", fuel), ("Seller_Type", seller), ("Transmission", trans)]:
        c = f"{name}_{val}"
        if c in row.columns:
            row[c] = 1
    price = model.predict(row)[0]
    st.success(f"Estimated selling price: ₹{price:.2f} lakh")
