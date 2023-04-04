
import streamlit as st
import pandas as pd

st.title("Beebie App")

st.write(""" # My First app Hello *world* """)

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('goodbye')
