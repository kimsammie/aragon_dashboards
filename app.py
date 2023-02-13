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

st.set_page_config(layout="wide")

st.title("Aragon Financial Reporting")

st.write("")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📈 OpEx", "Revenue", "Income Statement", "Balance Sheet"])

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
    
    col1, col2 = st.columns((1,1))

    with col1:
      income_df = pd.read_csv('income_stmt.csv')
        # display income statement
      fig = go.Figure(data=[go.Table(
          header=dict(values=list(income_df.columns),
                      fill_color='#264653',
                      font_color="white",
                      align='left'),
          cells=dict(values=[income_df.Components, income_df.Q1_22, income_df.Q2_22, income_df.Q3_22, income_df.Q4_22, income_df.FY_22],
                     fill_color='mintcream',
                     font_color="black",
                     align='left'))
      ])

      fig.update_layout(title_text="Summary Income Statement",title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=400)                                                               
      st.plotly_chart(fig, use_container_width=True)   
    
    with col2:
#       df = pd.read_csv('income_stmt_waterfall.csv')
#       a = df.Components
#       b = df.FY_22
#       waterfall_chart.plot(a, b)
      
#       fig = go.Figure(go.Waterfall(
#       name = "Net Income Components",
#       orientation = "v",
#       measure = ["relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
#       x = a,
#       textposition = "outside",
# #       text = ["+60", "+80", "", "-40", "-20", "Total"],
#       y = b,
#       connector = {"line":{"color":"rgb(63, 63, 63)"}},
#       ))

#       fig.update_layout(
#               title = "Key Financial Drivers",
#               showlegend = True
#       )

# #       fig.show()
#       st.plotly_chart(fig, use_container_width=True) 
  
  
  

      fig = make_subplots(rows=2, cols=1)

#       fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
#                     row=1, col=1)
      
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
      connector = {"line":{"color":"rgb(63, 63, 63)"}},row=1, col=1
      ))

      fig.update_layout(
              title = "Key Financial Drivers",
              showlegend = True
      )

      fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
                    row=2, col=1)

#       fig.show()
      st.plotly_chart(fig, use_container_width=True)
  
with tab4:
   st.header("Balance Sheet")
    
    
with st.sidebar:
    add_radio = st.radio(
        "Choose the time period",
        ("Q4 2022", "Q3 2022")
    )
