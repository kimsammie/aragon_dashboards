import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

st.title("Operating Expense Summary by Squad")

# load static file for now
df = pd.read_csv('Aragon_financial_simple.csv')

g1, g2 = st.columns((1,1))

# bar chart - MoM Expense by Squad
fig = plt.figure(figsize=(10, 4))
g1 = sns.catplot(
    data=df, x="Month", y="Amount", hue="Squad", kind="bar",
    sharex=False, margin_titles=True,
    aspect=4,errorbar=None
)
g1.set(xlabel="Month", ylabel="Amount", title='MoM Expense by Squad')

st.pyplot(g1)


# copy data for the pie chart    
df_=df[['Squad', 'Amount']].copy()
data=df_.groupby(['Squad']).sum()['Amount']
labels = df_['Squad'].unique().tolist()
g2, ax1 = plt.subplots()
palette_color = sns.color_palette('pastel')
ax1.pie(data, labels=labels, autopct='%1.1f%%',startangle=90, colors=palette_color)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(g2)
  
st.write(df)
