import streamlit as st
from streamlit_option_menu import option_menu


from config import set_page_config

set_page_config()


st.title("Détection de fraude dans le secteur bancaire")
st.subheader("Auteur : Atef Amor")

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

st.write('La fraude bancaire représente une menace majeure pour les institutions financières et leurs clients. Les fraudes par carte bancaire, en particulier, ont des impacts économiques significatifs et mettent en péril la confiance des clients dans le système bancaire. Afin de contrer ce problème, notre projet vise à développer une solution de détection de fraude utilisant des techniques avancées de machine learning.')

st.write("")
st.write("")

st.image("C:\\Users\\21697\\OneDrive\\Bureau\\stage_biware\\Fraud_Detection_App\\fraud-det.jpeg", use_column_width=False, width=600 )
st.write("")
st.write("")
st.write("")


#css

page_bg = """"

    <style>
    [data-testid="stApp"]{
    background-color: #ffffff;

    }
    [data-testid="stHeader"]{
    background-color: rgba(0, 0, 0, 0)}
    [data-testid="stSidebar"]{
    background-color: #4a536b }
    [data-testid="stHeading"]{
    text-align: center; }
    .stSidebar li {
        color: white;
    }
    </style>
    """
