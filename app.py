import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Operating Expense Summary by Squad")

# load static file for now
df = pd.read_csv('Aragon_financial_simple.csv')

col1, col2 = st.columns((1.5,1))

with col1:
  fig = px.histogram(df, x = 'Month', y='Amount', color="Squad", template = 'seaborn', barmode='group')
  fig.update_layout(title_text="MoM Expense by Squad",
                    yaxis_title="Amount", xaxis_title="Month")
  col1.plotly_chart(fig, use_container_width=True) 


with col2:
  # copy data for the pie chart    
  df_=df[['Squad', 'Amount']].copy()
  data=df_.groupby(['Squad']).sum()['Amount']
  labels = df_['Squad'].unique().tolist()

  fig = px.pie(df_, values=data, names=labels, template = 'seaborn')

  fig.update_layout(title_text="Expense Breakdown by Squad")
  col2.plotly_chart(fig, use_container_width=True)  

  



  
st.write(df)
