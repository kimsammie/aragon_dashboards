import streamlit as st
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# https://plotly.com/python/table/
import waterfall_chart
from plotly.subplots import make_subplots
from requests import request
import json
from collections import Counter
from datetime import datetime
import numpy as np
import requests
import datetime as dt
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk import *
from nltk.corpus import wordnet
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer
# from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from textblob import TextBlob

st.set_page_config(layout="wide")

st.title("Acme Financial Reporting")

st.write("")
st.write("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 OpEx", "Revenue", "Income Statement", "Balance Sheet", "Admin Input", 
                                                    "OpEx from Admin Input", "Workspace Expenses Detail", 
                                                    "Discord Channel Topics"])

with tab1:
  st.header("Operating Expense")
  st.write("A custom UI that displays components of expenses, trends and breakdowns all seamlessly off of files or data feeds.")
  st.write("")
  st.write("")
  st.write("")
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


  # display raw data
  fig = go.Figure(data=[go.Table(
      header=dict(values=list(df.columns),
                  fill_color='#264653',
                  font_color="white",
                  align='left'),
      cells=dict(values=[df.Squad, df.Category, df.Detail, df.Month, df.Amount],
                 fill_color='mintcream',
                 font_color="black",
                 align='left'))
  ])

  fig.update_layout(title_text="Raw Expense Data",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
  st.plotly_chart(fig, use_container_width=True)      

with tab2:
  st.header("Revenue")
  st.write("Track projected vs. actual over time so users can revisit their assumptions or reassess their operating models for more accurate predictions in the future.")
  st.write("")
  st.write("")
  st.write("")
  pod = ['Product Revenue', 'Services Revenue', 'Trading Revenue']
  month = ['Nov_22_Projected','Nov_22_Actual', 'Dec_22_Projected', 'Dec_22_Actual',
          'Jan_23_Projected','Jan_23_Actual','Feb_23_Projected','Feb_23_Actual']
  revenue_df = pd.DataFrame([[530000, 550000, 600000, 550000, 750000, 620000, 830000, 730000], 
                   [270000, 290000, 300000, 320000, 350000, 330000, 400000, 410000], 
                   [180000, 210000, 200000, 190000, 180000, 185000, 250000, 270000]], 
                  pod, month)
  revenue_df.index.name = "Source of Revenue"
  
  rev_proj_df = revenue_df[['Nov_22_Projected', 'Dec_22_Projected','Jan_23_Projected', 'Feb_23_Projected']]
  rev_proj_df.columns = ['Nov_22', 'Dec_22', 'Jan_23', 'Feb_23']
  rev_act_df = revenue_df[['Nov_22_Actual', 'Dec_22_Actual','Jan_23_Actual', 'Feb_23_Actual']]
  rev_act_df.columns = ['Nov_22', 'Dec_22', 'Jan_23', 'Feb_23']

      
  col1, col2 = st.columns((1,1))
  # display raw data
  with col1:
    headerlst = ["Source of Revenue"]+list(revenue_df.columns)
    fig = go.Figure(data=[go.Table(
        header=dict(values=headerlst,
                    fill_color='#264653',
                    font_color="white",
                    align='left'),
        cells=dict(values=[revenue_df.index, revenue_df.Nov_22_Projected,revenue_df.Nov_22_Actual, revenue_df.Dec_22_Projected, revenue_df.Dec_22_Actual,
            revenue_df.Jan_23_Projected,revenue_df.Jan_23_Actual,revenue_df.Feb_23_Projected,revenue_df.Feb_23_Actual],
                   fill_color='mintcream',
                   font_color="black",
                   align='left'))
    ])

    fig.update_layout(title_text="Revenue Summary by Pod",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
    st.plotly_chart(fig, use_container_width=True)  

  with col2:
    rev_df = pd.read_csv('aragon_revenue.csv')

    fig = px.histogram(rev_df, x = 'Month', y='Amount', color="Source", template = 'seaborn', barmode='group')
    fig.update_layout(title_text="Revenue Actual",
                      yaxis_title="Amount", xaxis_title="Month")
    col2.plotly_chart(fig, use_container_width=True) 

   
  graph1, graph2, graph3 = st.columns((1,1,1))
  # display raw data
  with graph1:
    prod_act = rev_act_df.iloc[0]
    prod_proj = rev_proj_df.iloc[0]
    prod_combined = pd.concat([prod_act, prod_proj], axis=1)
    prod_combined.columns = ['Actual', 'Projected']
    prod = px.line(prod_combined, template = 'seaborn')
#       fig.show()
#           fig = px.pie(df_, values=data, names=labels, template = 'seaborn')

    prod.update_layout(title_text="Product Revenue Trend")
    graph1.plotly_chart(prod, use_container_width=True) 
    
  with graph2:
    serv_act = rev_act_df.iloc[1]
    serv_proj = rev_proj_df.iloc[1]
    serv_combined = pd.concat([serv_act, serv_proj], axis=1)
    serv_combined.columns = ['Actual', 'Projected']
    serv = px.line(serv_combined, template = 'seaborn')
    serv.update_layout(title_text="Service Revenue Trend")
    graph2.plotly_chart(serv, use_container_width=True) 
      
  with graph3:
    trade_act = rev_act_df.iloc[2]
    trade_proj = rev_proj_df.iloc[2]
    trade_combined = pd.concat([trade_act, trade_proj], axis=1)
    trade_combined.columns = ['Actual', 'Projected']
    trade = px.line(trade_combined, template = 'seaborn')
    trade.update_layout(title_text="Trading Revenue Trend")
    graph3.plotly_chart(trade, use_container_width=True) 
    
with tab3:
    st.header("Income Statement")
  
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = 80366,
        # title = {"text": "Accounts<br><span style='font-size:0.8em;color:gray'>Subtitle</span><br><span style='font-size:0.8em;color:gray'>Subsubtitle</span>"},
        title = {"text": "Revenue<br><span style='font-size:0.8em;color:gray'>"},
        # delta = {'reference': 5034, 'relative': False},
        delta = {'reference': 5034, 'relative': True, "valueformat": ".2%"},
        domain = {'x': [0, 0.2], 'y': [0, 1]}))

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = 48765,
        title = {"text": "OpEx<br><span style='font-size:0.8em;color:gray'>"},
        delta = {'reference': 11654, 'relative': True, "valueformat": ".2%"},
        domain = {'x': [0.25, 0.45], 'y': [0, 1]}))

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = 4423,
        title = {"text": "Net Income<br><span style='font-size:0.8em;color:gray'>"},
        delta = {'reference': 6842, 'relative': True, "valueformat": ".2%"},
        domain = {'x': [0.5, 0.7], 'y': [0, 1]}))

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = 0.8,
        title = {"text": "Earnings per Token<br><span style='font-size:0.8em;color:gray'>"},
        delta = {'reference': 1, 'relative': True, "valueformat": ".2%"},
        domain = {'x': [0.75, 0.95], 'y': [0, 1]}))

    st.plotly_chart(fig, use_container_width=True)  
  
    income_df = pd.read_csv('income_stmt.csv')
    # display income statement
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(income_df.columns),
                fill_color='#264653',
                font_color="white",
                align='left'),
    cells=dict(values=[income_df.Components, income_df.FY_21, income_df.Q1_22, income_df.Q2_22, income_df.Q3_22, income_df.Q4_22, income_df.FY_22],
               fill_color='mintcream',
               font_color="black",
               align='left'))
    ])

    fig.update_layout(title_text="Summary Income Statement",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
    st.plotly_chart(fig, use_container_width=True)  
    
    col1, col2 = st.columns((1,1))
    
    with col1:
#       change colors for rows - https://stackoverflow.com/questions/66453291/python-plotly-table-change-color-of-a-specific-row
# bold fonts - https://stackoverflow.com/questions/51938245/display-dataframe-values-in-bold-font-in-one-row-only
      df = pd.read_csv('income_stmt_waterfall.csv')
      a = df.Components
      b = df.FY_22
      waterfall_chart.plot(a, b)
      
      fig = go.Figure(go.Waterfall(
      name = "Net Income Components",
      orientation = "v",
      measure = ["relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
      x = a,
      textposition = "outside",
#       text = ["+60", "+80", "", "-40", "-20", "Total"],
      y = b,
      connector = {"line":{"color":"rgb(63, 63, 63)"}},
      ))

      fig.update_layout(
              title = "Key Financial Drivers",
              showlegend = True
      )
#       fig.show()
      st.plotly_chart(fig, use_container_width=True) 
  
  
    with col2:
      df=pd.read_csv('income_trend.csv')
      fig = px.bar(df, x="period", y="amount", color="components", title="Net Income Trend")
      st.plotly_chart(fig, use_container_width=True)
      
        
# grid layout reference for later - https://github.com/streamlit/streamlit/issues/309
# two dataframes in a row - https://github.com/streamlit/streamlit/issues/4865

#       fig = make_subplots(rows=2, cols=1, subplot_titles=('Subplot title1',  'Subplot title2'))

# #       fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
# #                     row=1, col=1)
      
#       fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
#                     row=2, col=1)

# #       fig.show()
#       st.plotly_chart(fig, use_container_width=True)
  
with tab4:
  st.header("Balance Sheet")
    
  fig = go.Figure()

  # https://plotly.com/python/indicator/
  fig.add_trace(go.Indicator(
      mode = "number+delta",
      value = 44055,
      # title = {"text": "Accounts<br><span style='font-size:0.8em;color:gray'>Subtitle</span><br><span style='font-size:0.8em;color:gray'>Subsubtitle</span>"},
      title = {"text": "Current Assets<br><span style='font-size:0.8em;color:gray'>"},
      # delta = {'reference': 5034, 'relative': False},
      delta = {'reference': 828, 'relative': True, "valueformat": ".2%"},
      domain = {'x': [0, 0.2], 'y': [0, 1]}))

  fig.add_trace(go.Indicator(
      mode = "number+delta",
      value = 142477,
      title = {"text": "Long-Term Assets<br><span style='font-size:0.8em;color:gray'>"},
      delta = {'reference': 1208, 'relative': True, "valueformat": ".2%"},
      domain = {'x': [0.25, 0.45], 'y': [0, 1]}))

  fig.add_trace(go.Indicator(
      mode = "number+delta",
      value = 15323,
      title = {"text": "Liabilities<br><span style='font-size:0.8em;color:gray'>"},
      delta = {'reference': 3, 'relative': True, "valueformat": ".2%"},
      domain = {'x': [0.5, 0.7], 'y': [0, 1]}))

  fig.add_trace(go.Indicator(
      mode = "number+delta",
      value = 171209,
      title = {"text": "Equity<br><span style='font-size:0.8em;color:gray'>"},
      delta = {'reference': 2033, 'relative': True, "valueformat": ".2%"},
      domain = {'x': [0.75, 0.95], 'y': [0, 1]}))

  st.plotly_chart(fig, use_container_width=True)  

  col1, col2 = st.columns((1,1))
    
  with col1:
    # display balance sheet assets
    asset_df = pd.read_csv('bs_asset.csv')
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(asset_df.columns),
                fill_color='#264653',
                font_color="white",
                align='left'),
    cells=dict(values=[asset_df.Components, asset_df.Dec_21, asset_df.Mar_22, asset_df.Jun_22, asset_df.Sep_22, asset_df.Dec_22],
               fill_color='mintcream',
               font_color="black",
               align='left'))
    ])

    fig.update_layout(title_text="Summary Assets",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
    st.plotly_chart(fig, use_container_width=True)  

  with col2:
    # display balance sheet liabilities
    liab_df = pd.read_csv('bs_liab.csv')
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(liab_df.columns),
                fill_color='#264653',
                font_color="white",
                align='left'),
    cells=dict(values=[liab_df.Components, liab_df.Dec_21, liab_df.Mar_22, liab_df.Jun_22, liab_df.Sep_22, liab_df.Dec_22],
               fill_color='mintcream',
               font_color="black",
               align='left'))
    ])

    fig.update_layout(title_text="Summary Liabilities & Equity",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
    st.plotly_chart(fig, use_container_width=True)  
    

  df=pd.read_csv('current_asset_trend.csv')
  fig = px.bar(df, x="period", y="amount", color="components", title="Current Assets Trend")
  st.plotly_chart(fig, use_container_width=True)

with tab5:
  st.header("Admin Input")
  st.write('Multiple files can be uploaded, processed and standardized.')
  st.write('Select the sources of your financial data:')
  option_1 = st.checkbox('Gnosis')
  if option_1:
    text_input1 = st.text_input(
    "Enter the wallet address 👇",
#     "This is a placeholder",
#     key="placeholder",
    )
    uploaded_files = st.file_uploader("Upload the mapping", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)

  option_2 = st.checkbox('Parcel')
  if option_2:
    text_input2 = st.text_input(
    "Enter the wallet address 👇", key="parcel input"
    )
    
  option_3 = st.checkbox('Excel')
  if option_3:
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)

with tab6:
  st.header("OpEx from Admin Input")
  st.write("The below data was pulled directly from Gnosis Safe for Ops Guild. Additional Guilds and expense categories can be added.")
  st.write("")
  st.write("")
  st.write("")
  
  address = text_input1  
  gnosis_url = f"https://safe-transaction-mainnet.safe.global/api/v1/safes/{address}/all-transactions/"

  df = pd.read_csv('Aragon_financial_simple.csv')
  def merge_batches(url, txs=pd.DataFrame()):
    r = json.loads(request("GET", url).text)
    new_batch = pd.DataFrame.from_dict(r["results"])
    txs = pd.concat([txs, new_batch]).sort_values("executionDate")

    if r["next"]:
        txs = merge_batches(r["next"], txs)

    return txs

  txs = merge_batches(gnosis_url)

  transfers = pd.concat([pd.DataFrame(t) for t in txs["transfers"]]).reset_index()
  transfers["Amount"] = transfers["value"].astype(float) / 10 ** transfers["tokenInfo"].apply(lambda x: x["decimals"])
  
  squads = {
      "Data": [
          "0xbF0f24fD34bC76CD6d8Eb37cc99d4477ed3a98FB",
          "0x62ccd316e91EE0A0448E97251CdfA4dc660F34dc",
          "0xB5d08d891Ee61775BDDa71C7F5190573868309aE",
          "0x94Db1840c3C268023e9CdEd25049db37dA7f791d",
          "0xe5fF2C80759db9408dC9fD22b155b851Cd5aAA94",
          "0x1FCabBA469B151dF83a6A72Cdd1113c154F7A402",
          "0xf14B772F060F1315CAce35aF31955f54952e464D"
      ],
      "Ops": [
          "0xfE11aB456115186999724725fDf479A9569A641c",
          "0x51d93270eA1aD2ad0506c3BE61523823400E114C",
          "0x127277c6ED7Abaa30f72aa576d734ABe4C40D783",
          "0xd4ebc61981e5B9AB392b68f2638012E2346D534C",
          "0x2085E2838DE7f47128A94AC9d938ed4C5A28016B",
          "0x11f4c72e750407fa9572D0B3BE8AFcfD8f6FA16b",
          "0xa9A94e4718c045CCdf94266403aF4aDD53A2fD15",
          "0xd68512Dd51eeC0a1bbD8d3160FD62EF5bb740363",
          "0x29675e48606cF67603D501B85496AC7a842dABC3"

      ],
      "Finance": [
          "0xDacf1065b12849298dc5B47EcE9553094000074F",
          "0x580e5E54055d087D7F012dD43e54cceaE9ce4265",
          "0x9DF5Ce66CFA5655245AbbE337660a411d6EEBE37",
          "0x5496DB4eFC97A4B6592504638e70E925E6ea6e72",
          "0x22477DfBb4070100DB643Cb18fF6A06A186e9408",
      ],
      "Legal": ["0x7480Fa1a219F548E121F5a8F2bbF81eC61EfC318",
      "0xC9EeC90a0Ca7E61a24842eAfF2131DC90a7b4235", "0xe6B9b2a1eA3dfd9294f0e4a6abB41334b319c602"]
  }

  assert all([c == 1 for c in Counter([add for adds in squads.values() for add in adds]).values()])

  add_to_squad = {add: sq for sq, adds in squads.items() for add in adds}
  transfers.loc[:, "squad"] = transfers.loc[:, "to"].map(add_to_squad)  
  transfers.loc[transfers["squad"].isna(), ["executionDate", "transactionHash", "from", "to", "tokenId", "Amount"]].to_dict(orient="record")
  transfers_short = transfers[['executionDate', 'Amount', 'squad']]
  transfers_short['Amount'] = transfers_short['Amount'].round()
  transfers_short['Month'] = pd.to_datetime(transfers_short['executionDate']).apply(lambda x: x.strftime('%Y-%m')) 
  transfers_short.squad = transfers_short.squad.fillna('null')
#   st.write(transfers_short.head())
  
  col1, col2 = st.columns((1.5,1))

  with col1:
    fig = px.histogram(transfers_short, x = 'Month', y='Amount', color="squad", template = 'seaborn', barmode='group')
    fig.update_layout(title_text="MoM Expense by Squad",
                      yaxis_title="Amount", xaxis_title="Month")
    col1.plotly_chart(fig, use_container_width=True) 


  with col2:
    # copy data for the pie chart    
    transfers_short_=transfers_short[['squad', 'Amount']].copy()
    data=transfers_short_.groupby(['squad'], dropna=False).sum()
    data=data.sort_values(by=['Amount'], ascending=False)
    labels = data.index
    data=data['Amount']

    fig = px.pie(transfers_short_, values=data, names=labels, template = 'seaborn')

    fig.update_layout(title_text="Expense Breakdown by Squad")
    col2.plotly_chart(fig, use_container_width=True)  


  # display raw data
  fig = go.Figure(data=[go.Table(
      header=dict(values=['Month', 'Amount', 'Squad'],
                  fill_color='#264653',
                  font_color="white",
                  align='left'),
      cells=dict(values=[transfers_short.Month, transfers_short.Amount, transfers_short.squad], 
                 fill_color='mintcream',
                 font_color="black",
                 align='left'))
  ])

  fig.update_layout(title_text="Raw Expense Data",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
  st.plotly_chart(fig, use_container_width=True)   

with tab7:
  st.header("Workspace Expenses Detail")
  st.write("Drill-down view of the expenses. Guild > Squad > Workspace.")
  st.write("Contributor level or any other details can be available via permission-based access.")
  st.write("")
  st.write("")
  st.write("")
  # load static file for now
  df = pd.read_csv('DeWork.csv',encoding='windows-1252')
  df_=df[df['Amount'].notnull()]
  
  df_['Date']  = pd.to_datetime(df_['Date']).apply(lambda x: x.strftime('%Y-%m')) 
  df_ = df_.rename(columns={"Workspace Name": "Workspace_Name", "Task Name": "Task_Name"})

  col1, col2 = st.columns((1,1))

  with col1:
#     st.write("TBD")
    fig = px.histogram(df_, x = 'Date', y='Amount', color="Workspace_Name", template = 'seaborn', barmode='group')
    fig.update_layout(title_text="MoM Expense by Workspace",
                      yaxis_title="Amount", xaxis_title="Month")
    col1.plotly_chart(fig, use_container_width=True) 
    
  with col2:
    st.write("**Top 5 Contributors by Amount Paid**")
    df_agg = df_.groupby(['Date','Assignee', 'Workspace_Name']).agg({'Amount':sum})
    g = df_agg['Amount'].groupby('Date', group_keys=False)
    res = g.apply(lambda x: x.sort_values(ascending=False).head(5))
    st.write(res)
#     fig = go.Figure(data=[go.Table(
#     header=dict(values=['Date', 'Assignee', 'Workspace_Name', 'Amount'],
#                 fill_color='#264653',
#                 font_color="white",
#                 align='left'),
#     cells=dict(values=[res.Date, res.Assignee, res.Workspace_Name, res.Amount],
#                fill_color='mintcream',
#                font_color="black",
#                align='left'))
# ])

#     fig.update_layout(title_text="Raw Expense Data",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
#     st.plotly_chart(fig, use_container_width=True)     

  # display raw data
  fig = go.Figure(data=[go.Table(
      header=dict(values=['Date', 'Workspace_Name', 'Task_Name', 'Assignee', 'Amount'],
                  fill_color='#264653',
                  font_color="white",
                  align='left'),
      cells=dict(values=[df_.Date, df_.Workspace_Name, df_.Task_Name, df_.Assignee, df_.Amount],
                 fill_color='mintcream',
                 font_color="black",
                 align='left'))
  ])

  fig.update_layout(title_text="Raw Expense Data",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
  st.plotly_chart(fig, use_container_width=True)      

with tab8:
  st.header("Discord Channel Topics Discussed by the Community")
  st.write("""

	Using topic modeling, we can extract the "hidden topics" from large volumes of messages in Discord channels. 
	
	Please see the most popular topics discussed for a given time period by first selecting the start date and end date for a channel of your interest.	
	
	After that, try different number of topics (e.g., a higher number for a longer time period) until you see coherent topics (i.e., words in the topic support each other). 
	
	At the bottom, please also check out the overall sentiment found in the messages for the chosen time period. 
	
	""")
#   st.write('Select the sources of your financial data:')
#   option_1 = st.checkbox('Gnosis')
#   if option_1:
#     text_input1 = st.text_input(
#     "Enter the wallet address 👇"

#     )

# creating two functions as discord seems to take only one request i.e., either limit or before/after message id
# below is authorization from my discord login

# st.sidebar.write('Choose a week')
start_date_ofweek = st.date_input(
"Enter the start date (e.g., 2022/02/21)",
value=dt.datetime.now() - dt.timedelta(days=7),
)  # datetime.date format
end_date_ofweek = st.date_input(
"Enter the end date (e.g., 2022/02/28)", value=dt.datetime.now()
)

new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">**ERROR: Please choose the end date greater than the start date**</p>'
if start_date_ofweek > end_date_ofweek:
	st.markdown(new_title, unsafe_allow_html=True)
	
selection = st.selectbox(
"Choose the Discord channel",
["Option 1: General", "Option 2: Intro", "Option 3: Questions"],
)

if selection == "Option 1: General":
	channel_num = "672466989767458861"
elif selection == "Option 2: Intro":
	channel_num = "684539869502111755"
elif selection == "Option 3: Questions":
	channel_num = "694844628586856469"
	
numberof_topics = st.sidebar.number_input(
	"Enter the number of topics (2 to 10):",
	min_value=2,
	max_value=10,
	value=2,
	step=1,
	)

def retrieve_messages1(channelid):
	# payload={'page':2, 'count':100} # this with 'params=payload' doesn't work
	r = requests.get(
	    f"https://devops-server.aragon.org/discord/channel/messages?channelId={channelid}&limit=100"
	)
	jsonn = json.loads(r.text)
	return jsonn

def retrieve_messages2(channelid, messageid):
	r = requests.get(
	    f"https://devops-server.aragon.org/discord/channel/messages?channelId={channelid}&before={messageid}"
	)
	jsonn = json.loads(r.text)
	return jsonn

# NLTK Stop words
# from nltk.corpus import stopwords

stop_words = stopwords.words("english")
stop_words.extend(
        [
            "you",
            "me",
            "guy",
            "guys",
            "im",
            "us",
            "someone",
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank",
            "thx",
            "yes",
            "no",
            "ohh",
            "ha",
            "what",
            "would",
            "might",
            "could",
            "maybe",
            "may",
            "theres",
            "there",
            "here",
            "do",
            "does",
            "done",
            "be",
            "also",
            "still",
            "able",
            "since",
            "yet",
            "it",
            "many",
            "some",
            "rather",
            "make",
            "to",
            "and",
            "let",
            "please",
            "like",
            "not",
            "from",
            "ever",
            "try",
            "trying",
            "nice",
            "think",
            "thinking" "see",
            "seeing",
            "easy",
            "easily",
            "lot",
            "use",
            "using",
            "go",
            "going",
            "say",
            "said",
            "set",
            "want",
            "seem",
            "run",
            "need",
            "even",
            "right",
            "line",
            "take",
            "come",
            "look",
            "looking",
            "prob",
            "one",
            "feel",
            "way",
            "sure",
            "know",
            "get",
            "https",
            "http",
            "com",
            "etc",
            "daos",
            "subject",
        ]
    )
data = retrieve_messages1(channel_num)
df = pd.DataFrame(data)
df.sort_values("timestamp", ascending=False, inplace=True)
df.timestamp = pd.to_datetime(df.timestamp)

while len(df) < 1500:  # or use before/after timestamp
        latestid = df.tail(1)["id"].values[0]
        newdata = retrieve_messages2(channel_num, latestid)
        df1 = pd.DataFrame(newdata)
        df1.timestamp = pd.to_datetime(df1.timestamp)
        df = pd.concat([df, df1])  # expand the database
        df.sort_values("timestamp", ascending=False, inplace=True)
latestdate = df.tail(1)["timestamp"].values[0]

df = df.reset_index(drop=True)  # if not set to a variable it won't reset

latestdate = pd.to_datetime(latestdate).date()
earliestdate = latestdate + dt.timedelta(days=7)

df["timestamp"] = df["timestamp"].dt.date
start_date = pd.to_datetime(start_date_ofweek).date()
end_date = pd.to_datetime(end_date_ofweek).date()
one_week = (df["timestamp"] > start_date) & (df["timestamp"] <= end_date)
df_1wk = df.loc[one_week]
num_msgs = len(df_1wk)

st.write("**Note: the earliest date available:", earliestdate)

st.write("Start date:", start_date_ofweek)
st.write("End date:", end_date_ofweek)
st.write("Number of messages for the week:", len(df_1wk))

st.write("Number of Topics:", int(numberof_topics))

lemmatizer = WordNetLemmatizer()

    # Tokenize Sentences and Clean
def sent_to_words(sentences):
	for sent in sentences:
            sent = re.sub("\S*@\S*\s?", "", sent)  # remove emails
            sent = re.sub("\s+", " ", sent)  # remove newline chars
            sent = re.sub("'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(
                str(sent), deacc=True
            )  # split the sentence into a list of words. deacc=True option removes punctuations
            sent = [lemmatizer.lemmatize(w) for w in sent]
            yield (sent)

# Convert to list
data = df_1wk.content.values.tolist()
data_words = list(sent_to_words(data))

texts_out = [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in data_words
    ]
data_ready = texts_out

original_sentences = []
data_ready2 = []
for i in range(len(data_ready)):
	if len(data_ready[i]) > 3:
            data_ready2.append(data_ready[i])
            original_sentences.append(data_words[i])
data_ready = data_ready2
# build the topic model
# To build the LDA topic model using LdaModel(), need the corpus and the dictionary.
# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=int(numberof_topics),
        random_state=100,  # this serves as a seed (to repeat the training process)
        update_every=1,  # update the model every update_every chunksize chunks (essentially, this is for memory consumption optimization)
        chunksize=10,  # number of documents to consider at once (affects the memory consumption)
        passes=10,  # how many times the algorithm is supposed to pass over the whole corpus
        alpha="symmetric",  # `‘asymmetric’ and ‘auto’: the former uses a fixed normalized asymmetric 1.0/topicno prior, the latter learns an asymmetric prior directly from your data.
        iterations=100,
        per_word_topics=True,
    )  # setting this to True allows for extraction of the most likely topics given a word.
    # The training process is set in such a way that every word will be assigned to a topic. Otherwise, words that are not indicative are going to be omitted.
    # phi_value is another parameter that steers this process - it is a threshold for a word treated as indicative or not.

pprint(lda_model.print_topics())  # The trained topics (keywords and weights)

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
# Init output
	sent_topics_df = pd.DataFrame()

	# Get main topic in each document
	for i, row_list in enumerate(ldamodel[corpus]):
	    row = row_list[0] if ldamodel.per_word_topics else row_list
	    # print(row)
	    row = sorted(row, key=lambda x: (x[1]), reverse=True)
	    # Get the Dominant topic, Perc Contribution and Keywords for each document
	    for j, (topic_num, prop_topic) in enumerate(row):		
	       if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series(
                            [int(topic_num), round(prop_topic, 4), topic_keywords]
                        ),
                        ignore_index=True,
                    )
	       else:
                    break
	sent_topics_df.columns = [
	    "Dominant_Topic",
	    "Perc_Contribution",
	    "Topic_Keywords",
	]

	# Add original text to the end of the output
	contents = pd.Series(texts)
	sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
	return sent_topics_df

df_topic_sents_keywords = format_topics_sentences(
ldamodel=lda_model, corpus=corpus, texts=data_ready
)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = [
        "Document_No",
        "Dominant_Topic",
        "Topic_Perc_Contrib",
        "Keywords",
        "Text",
    ]
df_dominant_topic.head(10)

# the most representative sentence for each topic

# Get samples of sentences that most represent a given topic.
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby("Dominant_Topic")

for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat(
            [
                sent_topics_sorteddf_mallet,
                grp.sort_values(["Perc_Contribution"], ascending=False).head(1),
            ],
            axis=0,
        )

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = [
	"Topic_Num",
	"Topic_Perc_Contrib",
	"Keywords",
	"Representative Text",
	]

# Show
sent_topics_sorteddf_mallet.head(10)

# a word cloud with the size of the words proportional to the weight
# 1. Wordcloud of Top N words in each topic

# from wordcloud import WordCloud, STOPWORDS

cols = [
        color for name, color in mcolors.TABLEAU_COLORS.items()
    ]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(
        stopwords=stop_words,
        background_color="white",
        width=2500,
        height=1800,
        max_words=10,
        colormap="tab10",
        color_func=lambda *args, **kwargs: cols[i],
        prefer_horizontal=1.0,
    )

topics = lda_model.show_topics(formatted=False)	
    
fig, axes = plt.subplots(
        1, int(numberof_topics), figsize=(10, 10), sharex=True, sharey=True
    )  # nrows, ncols

for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title("Topic " + str(i + 1), fontdict=dict(size=16))
        plt.gca().axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis("off")
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
st.pyplot(fig)

polarity = []
sentiment_sentence = []
subjectivity = []
original_sentence = []
token_sentiments = []

for sentence in df_1wk.content:
        # st.write('sentence', sentence)
        # st.write('df_1wk.content', df_1wk.content)
        try:
            sentiment = TextBlob(sentence).sentiment
            # st.write('sentiment', sentiment)
            # token_sentiments = analyze_token_sentiment(sentence)
            # st.write('token_sentiments', token_sentiments)
            if sentiment.polarity > 0:
                polarity.append(sentiment.polarity)
                sentiment_sentence.append("Positive")
                subjectivity.append(sentiment.subjectivity)
                original_sentence.append(sentence)
                # token_sentiments.append(token_sentiments)

            elif sentiment.polarity < 0:
                polarity.append(sentiment.polarity)
                sentiment_sentence.append("Negative")
                subjectivity.append(sentiment.subjectivity)
                original_sentence.append(sentence)
                # token_sentiments.append(token_sentiments)

            else:
                polarity.append(sentiment.polarity)
                sentiment_sentence.append("Neutral")
                subjectivity.append(sentiment.subjectivity)
                original_sentence.append(sentence)
                # token_sentiments.append(token_sentiments)

        except:
            pass

sentiment_df_test = pd.DataFrame()
sentiment_df_test["polarity"] = polarity
sentiment_df_test["sentiment"] = sentiment_sentence
sentiment_df_test["original_sentence"] = original_sentence

# sentiment_df_test['subjectivity']=subjectivity
# sentiment_df_test['token_sentiments']=token_sentiments

sentiment_counts = sentiment_df_test.groupby(["sentiment"]).size()
st.write("Sentiment Counts:", sentiment_counts)

# visualize the sentiments
fig = plt.figure(figsize=(6, 6), dpi=100)
ax = plt.subplot(111)
sentiment_counts.plot.pie(
ax=ax, autopct="%1.1f%%", startangle=270, fontsize=12, label=""
)
st.pyplot(fig)

# st.write('Sentiment for the Actual Messages:', sentiment_df_test[['sentiment','original_sentence']])
st.write("Actual Messages:", df_1wk["content"])	

with st.sidebar:
#     st.write("Choose the time period")
    add_radio = st.radio(
        "Choose Year",
        ("2021", "2022", "2023")
    )

    add_radio = st.radio(
    "Choose Quarter",
    ("Q1", "Q2", "Q3", "Q4")
    )

    add_radio = st.radio(
    "Choose Squad",
    ("Data Squad", "Ops Squad", "Legal Squad", "Finance Squad")
    )

    # https://github.com/PablocFonseca/streamlit-aggrid
