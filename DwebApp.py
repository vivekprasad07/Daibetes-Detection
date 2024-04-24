# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:08:16 2024

@author: thevv
"""
import numpy as np
import pickle
import streamlit as st

#loading the Model
loaded_model = pickle.load(open('D:/Projects/Machine Learning/Daibetes Detection/trained_model.sav', 'rb'))

#creating a function for prediction

def diabetic_prediction(input_data):
    
    
    #chaing the input data into a NUMPY ARRAY
    in_data = np.asarray(input_data)
    
    #reshaping the data to predicting one instance
    input_data_reshaped = in_data.reshape(1,-1)
    
    #standardising the input data
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==0):
        return 'The Person is Non Diabetic'
    else:
        return 'The Person is Diabetic'
    
def main():
    
    #Giving a title 
    
    st.title('Diabetic Prediction Web App')
    
    #getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Level of Glucose')
    BloodPressure = st.text_input('Level of Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Level of Insuline')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person ')

    #code for prediction
    daignosis =''
    
    #Creating a button for prediction
    
    if st.button('Daibetes Test Result'):
        daignosis = diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(daignosis)
    
    
if __name__ == '__main__':
    main()
    
    
    