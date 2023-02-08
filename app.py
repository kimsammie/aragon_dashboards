import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

st.title("A Simple Streamlit Web App")
name = st.text_input("Enter your name", '')
st.write(f"Hello {name}!")
# x = st.slider("Select an integer x", 0, 10, 1)
# y = st.slider("Select an integer y", 0, 10, 1)
df = pd.read_csv('Aragon_financial_simple.csv')

with sns.axes_style('white'):
    g = sns.catplot(
        data=df, x="Month", y="Amount", hue="Squad", kind="bar",
        sharex=False, margin_titles=True,
        aspect=4,ci=None
    )
    g.set(xlabel="Month", ylabel="Amount", title='MoM Expense by Squad')
    
st.write(df)
