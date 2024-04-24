# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


loaded_model = pickle.load(open('D:/Projects/Machine Learning/Daibetes Detection/trained_model.sav', 'rb'))


input_data = (4,173,70,14,168,29.7,0.361,33)
#chaing the input data into a NUMPY ARRAY
in_data = np.asarray(input_data)

#reshaping the data to predicting one instance
input_data_reshaped = in_data.reshape(1,-1)

#standardising the input data

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The Person is Non Diabetic')
else:
    print('The Person is Diabetic')