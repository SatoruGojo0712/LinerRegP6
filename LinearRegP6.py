import streamlit as st
import pandas as pd    
import numpy as np  
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("2020neetrankscore.txt")
print (df.head(10))

x = df[['Marks']]
y = df['Rank']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

model=LinearRegression()
model.fit(x_train,y_train)

st.title("Rank Predictor")
print("Neet Mock Test Result Rankings by a Coaching centre, at All India Level of 2020 Neet upcoming Exam")
st.write("Enter your score to predict your Rank")

Score = st.number_input("Score calculated:", min_value=0.0, step=1.0)

if st.button("Predict Rank"):
     predicted_rank = model.predict([[Score]])[0]
     st.success(f"Predicted Rank: {predicted_rank:2f}")

st.write("Sample Data")
st.dataframe(df)     


