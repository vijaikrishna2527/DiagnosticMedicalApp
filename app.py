import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('  Medical Diagnostic App ⚕️')
st.subheader(' Does the patient have Diabetes ?')
df=pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View Data',False):
    st.write(df)
if st.sidebar.checkbox('View Distributions',False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
    
    
    
# step 1:  load the pickled model

model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()


# step 2: Get the front end user input

plasma=st.slider('Glucose',40,200,40)
pregs=st.number_input ('Pregnancies',0,20,0)
pres=st.slider ('BloodPressure',40,200,40)
skin=st.slider('SkinThickness',20,150,20)
insulin=st.slider ('Insulin',7,99,7)
bmi=st.slider ('BMI',14,850,14)
dpfa=st.slider ('DiabetesPedigreeFunction',0.05,2.50,0.05)
age=st.slider ('Age',21,90,21)


# Step 3: Get the moodel input 

input_data=[[pregs,pres,plasma,skin,insulin,bmi,dpfa,age]]


# Step 4: Get the prediction and  print result

prediction=clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader('Non Diabetic')
    else:
        st.subheader('Diabetic')
