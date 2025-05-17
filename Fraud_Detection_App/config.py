import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="Fraud Detection",
        layout="wide",  
        initial_sidebar_state="expanded",
        page_icon='credit-card'
    )