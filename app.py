import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Load all the instances that are required 
 with open('model.pkl','rb') as file:
     model=pickle.load(file)
 with open('scaler.pkl','rb') as file:
     model=pickle.load(file)
 with open('pca.pkl','rb') as file:
     model=pickle.load(file)
     
     
def prediction(input_data):
    scaled_data=scaler.transform(input_data)
    pca_data=pca.transform(scaled_data)
    pred=model.predict(pca_data)[0]
     if pred==0:
         return 'Developed'
     elif pred==1:
         return 'Developing'
     else:
         return 'Under Developed'
def main():
  st.title('HELP International Foundation')
  st.subheader(''' This application helps to classify the country on the basis of its scio-ecnomin and health factors''')
  chld_morh=st.text_input('Enter child mortality rate')
  lf_exp=st.text_input('Enter average life expectancy')
  tol_fer=st.text_input('Enter total fertility rate')
  health=st.text_input('Enter the % of GDP spent on health')
  export=st.text_input('Enter the % of GDP spent on exports')
  impor=st.text_input('Enter the % of GDP spent on imports')
  gdp=st.text_input('Enter GDP per population')
  income=st.text_input('Enter the income per person')
  infl=st.text_input('Enter inflation rate')

  inp_list=[[chld_morh,export,health,impor,income,infl,lf_exp,tol_fer,gdp]]
   

  if st.button('Predict'):
    response=prediction(inp_list)
    st.success(f'The country is {response}')
if __name__=='__main__':
    main()

 



