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

# text input where user will enter the SMS text to predict (As shown above)
input_sms = st.text_area("Enter the message: ")

# predict button, when clicked will execute the process
if st.button('Predict'):

    # 1. preprocess - converting the input_sms received by user on app
    transform_sms = input_sms
    print(type(transform_sms))
    transform_sms = np.array(transform_sms)  # converting the list of string format to array of string format

    # 2. vectorize - converting the received text SMS into numeric for model understanding
    vector_input = tfidf.transform(transform_sms.astype('str')).toarray()
    print(type(vector_input))
    print(vector_input)
    vector_input = pd.DataFrame(vector_input, columns=tfidf.get_feature_names_out())

    # 3. predict - passing the converted text to model to predict if it is spam or ham
    prediction = model.predict(vector_input)[0]

    # 4. display - the result on app itself , if prediction result is 1 then ui(button) will display  Spam else Not Spam
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
