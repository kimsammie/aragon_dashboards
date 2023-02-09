import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.title("Operating Expense Summary by Squad")

# load static file for now
df = pd.read_csv('Aragon_financial_simple.csv')

col1, col2 = st.columns((1,1))

with col1:
  fig = px.histogram(df, x = 'Month', y='Amount', color="Squad", template = 'seaborn', barmode='group')
  # fig.show()
  fig.update_layout(title_text="MoM Expense by Squad",
                    title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title="Amount", xaxis_title="Month")
  g1.plotly_chart(fig, use_container_width=True) 


with col2:
  # copy data for the pie chart    
  df_=df[['Squad', 'Amount']].copy()
  data=df_.groupby(['Squad']).sum()['Amount']
  labels = df_['Squad'].unique().tolist()



  # fig = px.bar(data, x = 'Arrived Destination Resolved', y='y', template = 'seaborn')
  fig = px.pie(df_, values=data, names=labels, template = 'seaborn')

  # fig.update_traces(marker_color='#7A9E9F')

  # fig.update_layout(title_text="Predicted Number of Arrivals",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None)
  fig.update_layout(title_text="Expense Breakdown by Squad")
  g2.plotly_chart(fig, use_container_width=True)  

  
with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")

# # bar chart - MoM Expense by Squad
# fig = plt.figure(figsize=(10, 4))
# g1 = sns.catplot(
#     data=df, x="Month", y="Amount", hue="Squad", kind="bar",
#     sharex=False, margin_titles=True,
#     aspect=4,errorbar=None
# )
# g1.set(xlabel="Month", ylabel="Amount", title='MoM Expense by Squad')


# st.pyplot(g1)





    
# g2, ax1 = plt.subplots()
# palette_color = sns.color_palette('pastel')
# ax1.pie(data, labels=labels, autopct='%1.1f%%',startangle=90, colors=palette_color)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# st.pyplot(g2)
  
st.write(df)
