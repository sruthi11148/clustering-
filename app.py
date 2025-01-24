import numpy as np
import pandas as pd
import streamlit as st
import pickle

#Load all the Instances that are required

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("pca.pkl", 'rb') as file:
    pca = pickle.load(file)


def prediction(input_data):
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
    pred = model.predict(pca_data)[0]

    if pred == 0:
        return "Developed, The Country Doesn't need any Aid"
    elif pred == 1:
        return "Developing, The Country needs Less Aid"
    else:
        return "Under Developed, The Country needs More Aid"

def main():
    st.title("HELP International Foundation")
    st.subheader("A Machine Learning Model to predict the level of development of a Country")
    cld_mor = st.text_input("Enter Child Mortality Rate")
    lf_exp = st.text_input("Enter Average Life Expectancy")
    tot_fer = st.text_input("Enter Average Total Fertility")
    export = st.text_input("Enter the % of GDP Spent on Export")
    imports = st.text_input("Enter the % of GDP Spent on Imports")
    health = st.text_input("Enter the % of GDP Spent on Health")
    gdp = st.text_input("Enter the GDP per Capita")
    income = st.text_input("Enter the Net Income per Person")
    inflation = st.text_input("Enter the Inflation Rate")
    
    input_data = [[cld_mor,export,health,imports,income,inflation,lf_exp,tot_fer,gdp]]

    if st.button("Predict"):
        response = prediction(input_data)
        st.success(response)

if __name__ == "__main__":
    main()
 
