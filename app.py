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


fig = plt.figure(figsize=(10, 4))
g = sns.catplot(
    data=df, x="Month", y="Amount", hue="Squad", kind="bar",
    sharex=False, margin_titles=True,
    aspect=4,ci=None
)
g.set(xlabel="Month", ylabel="Amount", title='MoM Expense by Squad')

# titanic = sns.load_dataset("titanic")

# fig = plt.figure(figsize=(10, 4))
# sns.countplot(x="class", data=titanic)

st.pyplot(g)
    
df_=df[['Squad', 'Amount']].copy()
# data=df_['Amount']
data=df_.groupby(['Squad']).sum()['Amount']

labels = df_['Squad'].unique().tolist()

# labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
# sizes = [15, 30, 45, 10]
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(data, labels=labels, autopct='%1.1f%%',startangle=90)
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
