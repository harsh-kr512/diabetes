import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('diabetes_model.sav','rb'))

# Create a function for Prediction
def diabetes_prediction(input_data):

    # Change the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data).astype(float)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    # Give a title
    st.title('Diabetes Prediction Web App')
    
    # To get the input data from the user
    Pregnancies = st.number_input('Number of Pregnancies')
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('Blood Pressure value')
    SkinThickness = st.number_input('Skin Thickness value')
    Insulin = st.number_input('Insulin Level')
    BMI = st.number_input('BMI value')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    Age = st.number_input('Age of the Person')
    
    
    # Create a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.write(diagnosis)

if __name__ == '__main__':
    main()