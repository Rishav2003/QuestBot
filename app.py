import streamlit as st
import helper_org
import pickle

model1 = pickle.load(open('random_forest_model.pkl','rb'))
model = pickle.load(open('word2vec_model.pkl','rb'))
st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper_org.generate_feature(q1, q2, model)
    result = model1.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate') 


