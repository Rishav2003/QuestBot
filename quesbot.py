import streamlit as st
import helper_org
import pickle
from streamlit_option_menu import option_menu
import time 
import requests
from streamlit_lottie import st_lottie
model1 = pickle.load(open('random_forest_model.pkl','rb'))
model = pickle.load(open('word2vec_model.pkl','rb'))
     
with st.sidebar:
    
    selected = option_menu('QUESBOT: A Duplicacy detection platform',
                          
                          ['HOME',
                            'Detection'],
                          icons=['house','eye'],
                          default_index=0) 
if (selected == 'HOME'):
    st.title('WELCOME TO QuesBOT!!')
    def load_lottieurl(url: str):
         r = requests.get(url)
         if r.status_code != 200:
            return None
         return r.json()


    lottie_url_hello = "https://lottie.host/cad7940e-6aa9-40d4-ab08-c64e3de10b99/vRabVp46GO.json"
    lottie_hello = load_lottieurl(lottie_url_hello)
    st_lottie(lottie_hello, key="hello",height=550)
    st.header('What is QuesBot ? ')
    st.markdown(
    """
    <div style="text-align: justify">
        QuesBot is an advanced question pair detection platform designed to streamline the identification of duplicate or similar questions. 
        This intelligent tool is built to enhance the efficiency of question analysis, helping users identify and manage duplicate questions effortlessly.
    </div>
    """,
    unsafe_allow_html=True
)
    lottie_url_hello1 = "https://assets1.lottiefiles.com/packages/lf20_5LVVIi.json"
    lottie_hello1 = load_lottieurl(lottie_url_hello1)
    st_lottie(lottie_hello1, key="hello1",height=450)
    st.header('WHY QuesBot ? ')
    st.markdown('''
    <div style="text-align: justify">
    QuesBot stands as the quintessential solution for anyone seeking unparalleled accuracy in duplicate question pair detection. With a seamless transition from Bag of Words to Word2Vec, QuesBot captures nuanced semantic meanings, ensuring a level of precision unmatched in the field. 
    Its incorporation of diverse heuristic features, coupled with the elegance of Random Forest, enables effortless integration and achieves an impressive 83% accuracy without the need for intricate fine-tuning. 
    QuesBot's holistic approach to feature engineering and proven performance in Kaggle competitions make it the go-to choice for those valuing efficiency, accuracy, and simplicity in their quest for superior question pair detection.</div>''', unsafe_allow_html=True )
    lottie_url_hello2 = "https://lottie.host/5b6159b3-d0f8-4209-bd82-61f47a9a952f/eYhbNBDhjT.json"
    lottie_hello2 = load_lottieurl(lottie_url_hello2)
    st_lottie(lottie_hello2, key="hello2",height=450)
    st.markdown('You can toggle the Navbar using side button and check out dthe detection model ')
    st.subheader("Enjoy this site!")
    

if (selected == 'Detection'):
    
    # page title
    st.title('Question Pair Detection using Word2Vec')
    def load_lottieurl(url: str):
         r = requests.get(url)
         if r.status_code != 200:
            return None
         return r.json()


    lottie_url_hello = "https://lottie.host/94cd58e5-c976-4dd2-b028-cd6d8b620c9f/dceF4sKqtG.json"
    lottie_hello = load_lottieurl(lottie_url_hello)
    st_lottie(lottie_hello, key="hello",height=500)
    
    # getting the input data from the user
    q1 = st.text_input('Enter question 1')
    q2 = st.text_input('Enter question 2')

    
    
    # code for Prediction
    res = ''
    
    # creating a button for Prediction
    
    if st.button('Check Duplicacy!'):
        query = helper_org.generate_feature(q1, q2, model)
        result = model1.predict(query)[0]
        st.balloons()
        if result:
          res="The Question Pairs are Duplicate"
        else:
          res="The Question Pairs are not Duplicate"
        
        
    st.success(res)
    if res == "The Question Pairs are not Duplicate":
        lottie_url_hello1 = "https://lottie.host/8e275d82-0936-41dc-b03a-61153484757d/jW8eNwVZkW.json"
        lottie_hello1 = load_lottieurl(lottie_url_hello1)
        st_lottie(lottie_hello1, key="hello1",width=700,height=200)
        st.text('''Thank You for using this website.This app was made on a random forest model and
Word2Vec was used to capture semantic meaning.''')
        
    elif res == "The Question Pairs are Duplicate":
        lottie_url_hello2 = "https://assets3.lottiefiles.com/packages/lf20_j6fywzxe.json"
        lottie_hello2 = load_lottieurl(lottie_url_hello2)
        st_lottie(lottie_hello2, key="hello2",width=700,height=200)
        st.text('''Thank You for using this website.This app was made on a random forest model and
Word2Vec was used to capture semantic meaning.''')
    else:
        st.text("Happy to serve !!! Thank You For Using this web app.")



