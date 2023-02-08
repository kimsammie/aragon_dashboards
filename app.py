import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

st.title("Operating Expense Summary by Squad")

# load static file for now
df = pd.read_csv('Aragon_financial_simple.csv')

# bar chart - MoM Expense by Squad
fig = plt.figure(figsize=(10, 4))
g = sns.catplot(
    data=df, x="Month", y="Amount", hue="Squad", kind="bar",
    sharex=False, margin_titles=True,
    aspect=4,ci=None
)
g.set(xlabel="Month", ylabel="Amount", title='MoM Expense by Squad')

st.pyplot(g)


# copy data for the pie chart    
df_=df[['Squad', 'Amount']].copy()
data=df_.groupby(['Squad']).sum()['Amount']
labels = df_['Squad'].unique().tolist()
fig1, ax1 = plt.subplots()
palette_color = sns.color_palette('pastel')
ax1.pie(data, labels=labels, autopct='%1.1f%%',startangle=90, colors=palette_color)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
  
# # define Seaborn color palette to use
# palette_color = sns.color_palette('pastel')
  
# # plotting data on chart
# pie = plt.pie(data, labels=labels, colors=palette_color, autopct='%.0f%%')

# pie = plt.title("Expense Breakdown by Squad")
         
# # displaying chart
# plt.show()

# st.pyplot(pie)
    

st.write(df)
