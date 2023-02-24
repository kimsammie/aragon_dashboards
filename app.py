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

st.set_page_config(layout="wide")

st.title("Acme Financial Reporting")

st.write("")
st.write("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 OpEx", "Revenue", "Income Statement", "Balance Sheet", "Admin Input", 
                                                    "OpEx from Admin Input", "Workspace Expenses Detail"])

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
  st.write('Select the source of your financial data:')
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
  st.write("A custom UI that displays components of expenses, trends and breakdowns all seamlessly off of files or data feeds.")
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
#   st.write("until this works")
#   transfers_short['Month']  = transfers_short['executionDate'].apply(lambda x: x.strftime('%Y-%m')) 
#   def day_month_flip(date_to_flip):
#     return pd.to_datetime(date_to_flip.strftime('%Y-%m'))

#   transfers_short['Month']  = transfers_short['executionDate'].apply(lambda x: day_month_flip(x))
  
  col1, col2 = st.columns((1.5,1))

  with col1:
    fig = px.histogram(transfers_short, x = 'executionDate', y='Amount', color="squad", template = 'seaborn', barmode='group')
    fig.update_layout(title_text="MoM Expense by Squad",
                      yaxis_title="Amount", xaxis_title="Month")
    col1.plotly_chart(fig, use_container_width=True) 


  with col2:
    # copy data for the pie chart    
    transfers_short_=transfers_short[['squad', 'Amount']].copy()
    data=transfers_short_.groupby(['squad'], dropna=False).sum()
    data=data.sort_values(by=['Amount'], ascending=False)
    labels = transfers_short_['squad'].unique().tolist()
    st.write(labels)
    data=data['Amount']
    st.write(data)


    fig = px.pie(transfers_short_, values=data, names=labels, template = 'seaborn')

    fig.update_layout(title_text="Expense Breakdown by Squad")
    col2.plotly_chart(fig, use_container_width=True)  


  # display raw data
  fig = go.Figure(data=[go.Table(
      header=dict(values=list(transfers_short.columns),
                  fill_color='#264653',
                  font_color="white",
                  align='left'),
      cells=dict(values=[transfers_short.executionDate, transfers_short.Amount, transfers_short.squad], 
                 fill_color='mintcream',
                 font_color="black",
                 align='left'))
  ])

  fig.update_layout(title_text="Raw Expense Data",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
  st.plotly_chart(fig, use_container_width=True)   

with tab7:
  st.header("Workspace Expenses Detail")
  st.write("Drill-down view of the expenses. Guild > Squad > Workspace")
  st.write("")
  st.write("")
  st.write("")
  # load static file for now
  df = pd.read_csv('DeWork.csv',encoding='windows-1252')
  df_=df[df['Amount'].notnull()]
  
  df_['Date']  = pd.to_datetime(df_['Date']).apply(lambda x: x.strftime('%Y-%m')) 
  df_.rename(columns={"Workspace Name": "Workspace_Name", "Task Name": "Task_Name"})

  col1, col2 = st.columns((1,1))

  with col1:
#     st.write("TBD")
    fig = px.histogram(df_, x = 'Date', y='Amount', color="Workspace Name", template = 'seaborn', barmode='group')
    fig.update_layout(title_text="MoM Expense by Workspace",
                      yaxis_title="Amount", xaxis_title="Month")
    col1.plotly_chart(fig, use_container_width=True) 
    
  with col2:
    st.write("Top 5 Contributors by Amount Paid")
    df_agg = df_.groupby(['Date','Assignee', 'Workspace Name']).agg({'Amount':sum})
    g = df_agg['Amount'].groupby('Date', group_keys=False)
    res = g.apply(lambda x: x.sort_values(ascending=False).head(5))
    st.write(res)

  # display raw data
  fig = go.Figure(data=[go.Table(
      header=dict(values=list(df_.columns),
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
