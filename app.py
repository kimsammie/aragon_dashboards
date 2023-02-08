import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

def write():
    st.title("Aragon Discord Channel Topics Discussed by the Community ")

    st.markdown(
        """
	<br><br/>
	
	Using topic modeling, we can extract the "hidden topics" from large volumes of messages in Aragon Discord channels. 
	
	Please see the most popular topics discussed for a given time period by first selecting the start date and end date for a channel of your interest.	
	
	After that, try different number of topics (e.g., a higher number for a longer time period) until you see coherent topics (i.e., words in the topic support each other). 
	
	At the bottom, please also check out the overall sentiment found in the messages for the chosen time period. 
	
	""",
        unsafe_allow_html=True,
    )
    df = pd.read_csv('Aragon financial simple.csv')

    st.sidebar.write(
        "[Source Code](https://github.com/kimsammie/Aragon_Discord_Metrics)"
    )
