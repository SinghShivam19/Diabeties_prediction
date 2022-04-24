import numpy as np
import pickle
import streamlit as st
import os
loaded_model = pickle.load(open("C:/Users/SHIVAM SINGH/Dropbox/My PC (LAPTOP-IAV1MUTT)/Downloads/trained_model.pkl", 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # giving a title
    st.title('Diabetes Prediction Web App')

    # getting the input data from the user

    Pregnancies = st.slider('Number of Pregnancies', 0, 20, 5)
    Glucose = st.slider('Glucose Level', 0, 200, 99)
    BloodPressure = st.slider('Blood Pressure value', 0, 120, 23)
    SkinThickness = st.slider('Skin Thickness value', 0, 100, 33)
    Insulin = st.slider('Insulin Level', 0, 300, 23)
    BMI = st.slider('BMI value', 0.00, 60.00, 21.5)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function value', 0.00, 3.00, 1.5)
    Age = st.slider('Age of the Person', 0, 100, 23)

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()

