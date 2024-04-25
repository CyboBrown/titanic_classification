import streamlit as st  # used for streamlit api reference
# below all libraries were part of model building and hence add them again.
import pickle  # to load the saved pickle files
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler

# loading both the models from respective directory
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('titanic_classification.pkl', 'rb'))

# streamlit app title
st.title("Titanic Survival Classification")

# text input where user will enter the parameters
# input_p_id = float(st.number_input("Passenger ID: ", 0, 10000, step=1))
input_p_class = st.selectbox("Passenger Class: ", ["1st", "2nd", "3rd"])
input_sex = st.selectbox("Sex: ", ["Male", "Female"])
input_age = float(st.number_input("Age: ", 0, 200, step=1))
input_sib_sp = float(st.number_input("No. of Siblings/Spouse: ", 0, 100, step=1))
input_par_ch = float(st.number_input("No. of Parents/Children: ", 0, 100, step=1))
input_fare = float(st.number_input("Fare: "))
input_embarked = st.selectbox("Port of Embarkation: ", ["Cherbourg", "Queenstown", "Southampton"])
input_title = st.selectbox("Title: ", ['Mrs', 'Miss', 'Master', 'Other', 'Mr'])


# predict button, when clicked will execute the process
if st.button('Predict'):
    # st.header([input_p_id, input_p_class, input_sex, input_age, input_sib_sp, input_par_ch, input_fare, input_embarked, input_title])
    input_p_class = 1 if (input_p_class == '1st') else 2 if (input_p_class == '2nd') else 3 if (input_p_class == '3rd') else 0
    input_sex = 0 if (input_sex == 'Male') else 1
    input_embarked = 0 if (input_embarked == 'Southampton') else 1 if (input_embarked == 'Cherbourg') else 2 if (input_embarked == 'Queenstown') else -1
    titles = ['Mrs', 'Miss', 'Master', 'Other', 'Mr']
    input_is_mr = 1 if (input_title == 'Mr') else 0
    input_title = titles.index(input_title)
    input_age_group = 0 if (input_age <= 16) else 1 if (16 < input_age <= 32) else 2 if (32 < input_age <= 48) else 3 if (48 < input_age <= 64) else 4 if (input_age > 64) else -1
    input_is_elderly = 1 if (input_age >= 65) else 0
    input_family_size = input_sib_sp + input_par_ch + 1
    input_is_alone = 1 if (input_family_size == 1) else 0
    input_fare_group = 0 if (input_fare <= 7.91) else 1 if (7.91 < input_fare <= 14.454) else 2 if (14.454 < input_fare <= 31.0) else 3 if (input_fare > 31.0) else -1

    vector_input = [
        [input_p_class, input_sex, input_age, input_sib_sp, input_par_ch, input_fare, input_embarked, input_title,
         input_is_mr, input_age_group, input_is_elderly, input_family_size, input_is_alone, input_fare_group]
    ]
    # st.header(vector_input)
    prediction = model.predict(vector_input)[0]
    st.header(prediction)

    if prediction[0] <= 0.5:
        st.header("You would have most likely survive in Titanic.")
    else:
        st.header("You would have most likely die in Titanic.")
    st.text("The probability of you dying is about " + str(prediction[0]) + "%")
