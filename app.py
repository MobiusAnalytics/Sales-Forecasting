#%%writefile app.py
 
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path
import streamlit_authenticator as stauth
import statsmodels.api as sm

#------ USER AUTHENTICATION-----------

names = ["Mobius DA"]
usernames = ["Mobius_Data_Analytics"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {"usernames":{}}
for un, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})

authenticator = stauth.Authenticate(credentials,"Campaign","abc123",cookie_expiry_days=0)

hide_streamlit_style = """<style> #MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

name,authetication_status,username = authenticator.login("LOGIN","main")
if authetication_status == False:
    st.error("Username/Password is incorrect")
if authetication_status == None:
    st.warning("Please enter your Username and Password")
    
#------ IF USER AUTHENTICATION STATUS IS TRUE  -----------   
if authetication_status:

    data_P1D1  = pd.read_csv('Sales_P1_D1.csv')
    data_P1D1['Date'] = pd.to_datetime(data_P1D1['Date'], infer_datetime_format=True)
    data_P1D1.set_index("Date", drop=True, inplace=True)
    data_P2D1  = pd.read_csv('Sales_P2_D1.csv')
    data_P2D1['Date'] = pd.to_datetime(data_P2D1['Date'], infer_datetime_format=True)
    data_P2D1.set_index("Date", drop=True, inplace=True)
    data_P1D2  = pd.read_csv('Sales_P1_D1.csv')
    data_P1D2['Date'] = pd.to_datetime(data_P1D2['Date'], infer_datetime_format=True)
    data_P1D2.set_index("Date", drop=True, inplace=True)
    data_P2D2  = pd.read_csv('Sales_P2_D1.csv')
    data_P2D2['Date'] = pd.to_datetime(data_P2D2['Date'], infer_datetime_format=True)
    data_P2D2.set_index("Date", drop=True, inplace=True)

    
    @st.cache
      
    # defining the function which will make the prediction using the data which the user inputs 
    def prediction(days,product,dealer):    
        days = int(days)
        if (product == "Product 1") & (dealer == "Dealer 1"):          
            forcast_model_full=sm.tsa.statespace.SARIMAX(data_P1D1['Sales'],order=([3,1,3]),seasonal_order=([3,1,3,12]),enforce_stationarity=False)
            SARIMAX_Model_P1D1=forcast_model_full.fit()
            pred = pd.DataFrame(SARIMAX_Model_P1D1.forecast(steps=int(days)))           
            pred.rename(columns={'predicted_mean':'Sales(Forecasted)'},inplace=True)
            predictions = pd.concat([data_P1D1,pred],axis=0).iloc[-days:]   
            print(data_P1D1.index)
            predictions['Product'] = product
            predictions['Dealer'] = dealer
            predictions['Sales'] = 0
            future_df = pd.concat([data_P1D1,pred],axis=0)
        elif (product == "Product 2") & (dealer == "Dealer 1"):  
            forcast_model_full=sm.tsa.statespace.SARIMAX(data_P2D1['Sales'],order=([3,1,3]),seasonal_order=([3,1,3,12]),enforce_stationarity=False)
            SARIMAX_Model_P2D1=forcast_model_full.fit()
            pred = pd.DataFrame(SARIMAX_Model_P2D1.forecast(steps=int(days)))
            pred.rename(columns={'predicted_mean':'Sales(Forecasted)'},inplace=True)
            predictions = pd.concat([data_P2D1,pred],axis=0).iloc[-days:]            
            predictions['Product'] = product
            predictions['Dealer'] = dealer
            predictions['Sales'] = 0
            future_df = pd.concat([data_P2D1,pred],axis=0)
        elif (product == "Product 1") & (dealer == "Dealer 2"): 
            forcast_model_full=sm.tsa.statespace.SARIMAX(data_P1D2['Sales'],order=([3,1,3]),seasonal_order=([3,1,3,12]),enforce_stationarity=False)
            SARIMAX_Model_P1D2=forcast_model_full.fit()
            pred = pd.DataFrame(SARIMAX_Model_P1D2.forecast(steps=int(days)))
            pred.rename(columns={'predicted_mean':'Sales(Forecasted)'},inplace=True)
            predictions = pd.concat([data_P1D2,pred],axis=0).iloc[-days:]            
            predictions['Product'] = product
            predictions['Dealer'] = dealer
            predictions['Sales'] = 0
            future_df = pd.concat([data_P1D2,pred],axis=0)
        elif (product == "Product 2") & (dealer == "Dealer 2"): 
            forcast_model_full=sm.tsa.statespace.SARIMAX(data_P2D2['Sales'],order=([3,1,3]),seasonal_order=([3,1,3,12]),enforce_stationarity=False)
            SARIMAX_Model_P2D2=forcast_model_full.fit()
            pred = pd.DataFrame(SARIMAX_Model_P2D2.forecast(steps=int(days)))
            pred.rename(columns={'predicted_mean':'Sales(Forecasted)'},inplace=True)
            predictions = pd.concat([data_P2D2,pred],axis=0).iloc[-days:]            
            predictions['Product'] = product
            predictions['Dealer'] = dealer
            predictions['Sales'] = 0
            future_df = pd.concat([data_P2D2,pred],axis=0)
                
        return predictions,future_df
        
            
          
        #return predictions #future_df
          
      
    # this is the main function in which we define our webpage  
    def main(): 
        authenticator.logout("Logout",'sidebar')
        st.sidebar.image("""https://website-assets-fw.freshworks.com/attachments/cjqyqslxm024h2wg059nrfstc-easy-forecasting-at-no-cost-2x.one-half.png""",width=300)
        
        global days, product, dealer 
        title = '<p style="font-family:sans-serif; color:black;text-align:center; font-size: 45px;"><b>Sales Forecasting for Dealers</b></p>'
        #subtitle = '<p style="font-family:sans-serif; color:grey;text-align:center; font-size: 20px;"><b>Time Series Forecasting</b></p>'
        st.markdown(title, unsafe_allow_html = True)        
        
        product = st.sidebar.selectbox("Product Name",("Product 1", "Product 2"))
        dealer = st.sidebar.selectbox("Dealer Name",("Dealer 1", "Dealer 2"))
        st.markdown('<p style="font-family:sans-serif; color:black;text-align:left; font-size: 14px;"><b>Forecast Sales (days)</b></p>',unsafe_allow_html = True)            
        
        days = st.slider("",min_value=1,max_value=90)
        ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
            background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)
        Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
            background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)           
        Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                        { color: rgb(14, 38, 74); } </style>''', unsafe_allow_html = True)         
        col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
            background: linear-gradient(to right, rgb(1, 183, 158) 0%, 
                                        rgb(1, 183, 158) {days}%, 
                                        rgba(151, 166, 195, 0.25) {days}%, 
                                        rgba(151, 166, 195, 0.25) 100%); }} </style>'''
        ColorSlider = st.markdown(col, unsafe_allow_html = True)                 
        result = ""
        
        if st.button("PREDICT"): 
            predictions,future_df = prediction(days,product,dealer)           
            predictions = round(predictions,3)
            with st.container():
                st.markdown('<p style="font-family:sans-serif; color:black;text-align:left; font-size: 14px;"><b>Table: Forecasted Sales for requested number of future days</b></p>',unsafe_allow_html = True)            
                st.write(round(predictions))
                predictions =predictions.to_csv(index=True).encode('utf-8')
                st.download_button(label='Download CSV',data=predictions,mime='text/csv',file_name='Download.csv')    
                chart_data = round(future_df[['Sales','Sales(Forecasted)']])                
                st.subheader("")
                st.markdown('<p style="font-family:sans-serif; color:black;text-align:left; font-size: 14px;"><b>Lineplot showing Actual vs Forecasted Sales</b></p>',unsafe_allow_html = True)
                #st.markdown("Lineplot showing Actual vs Forecasted Sales ")
                st.line_chart(chart_data)
            
    if __name__=='__main__': 
        main()
    
