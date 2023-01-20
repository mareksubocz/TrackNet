import streamlit as st
import pandas as pd
import numpy as np

#streamlit run demo.py
st.title('DEMO')

# raczej trzeba okreslij jaki format pliku powinien być wrzucany
uploaded_file = st.file_uploader("Prześlij swój plik", type=None, key=None, help="bla bla bla")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
