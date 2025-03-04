import streamlit as st
from streamlit import set_option
import numpy as np
import pickle
from Diabetes import diab_data
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training model
with open('diabetes.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the scaler used during training
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('DIABETES DISEASES PREDICTION')

select = st.sidebar.selectbox('Select Your Preference', ['Prediction Page', 'Explore Page'])

if select == 'Prediction Page':
    # Apply custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #F3E251;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton button {
            background-color: #01080E;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #FFFFFF;
        }
        .diagnosis {
            font-size: 24px;
            font-weight: bold;
            color: #E74C3C;
            background-color: #FDEDEC;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('No of Pregnancies')
        
    with col2:    
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure')
        
    with col1:
        SkinThickness = st.text_input('SkinThickness')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
        
    with col3:    
        BMI = st.text_input('BMI')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

    with col2:
        Age = st.text_input('Age')

    diagnosis = ''

    if st.button('Predict'):
        try:
            input_data = [
                float(Pregnancies) if Pregnancies else 0,
                float(Glucose) if Glucose else 0,
                float(BloodPressure) if BloodPressure else 0,
                float(SkinThickness) if SkinThickness else 0,
                float(Insulin) if Insulin else 0,
                float(BMI) if BMI else 0,
                float(DiabetesPedigreeFunction) if DiabetesPedigreeFunction else 0,
                float(Age) if Age else 0
            ]
            
            input_data_to_array = np.asarray(input_data)
            input_data_reshaped = input_data_to_array.reshape(1, -1)
            standardized_data = scaler.transform(input_data_reshaped)
            prediction = loaded_model.predict(standardized_data)
            result = 'The person is non-diabetic' if prediction[0] == 0 else 'The person is diabetic'
            st.write("Predicted class for the given input data:", prediction[0])
                
            diagnosis = result
            
        except ValueError:
            diagnosis = 'Enter valid numerical values'
            
    st.success(diagnosis)

elif select == 'Explore Page':
    st.markdown(
        """<style>
        .main {
            background-color: #EFEB7F;
            padding: 10px;
            border-radius: 10px;
            cursor: pointer;
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader('Explore the data with graphs and visuals')

    df = diab_data

    st.subheader('Distribution of Age')
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Age'], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader('Glucose Levels by Outcome')
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Outcome', y='Glucose', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader('Correlation Heatmap')
    fig3, ax3 = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    st.subheader('BMI vs. Age Scatter Plot')
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x='Age', y='BMI', hue='Outcome', data=df, ax=ax4)
    st.pyplot(fig4)

    st.subheader('Distribution of Blood Pressure')
    fig5, ax5 = plt.subplots()
    sns.histplot(df['BloodPressure'], bins=30, kde=True, ax=ax5)
    st.pyplot(fig5)
