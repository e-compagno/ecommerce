import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#import pandas_profiling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import joblib
import pickle

import sqlalchemy as db
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm import sessionmaker
import os

from datetime import datetime, timedelta

%matplotlib inline
# Customise plots
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 12

# Load MYSql connector 
SQL_USR, SQL_PSW= os.environ['SQL_USR'], os.environ['SQL_PSW']
mysql_str = 'mysql+mysqlconnector://'+SQL_USR+':'+SQL_PSW+'@localhost:3306/'
engine = db.create_engine(mysql_str+'Commerce')

# Load database ecommerce
con=engine.connect()
df_size=con.execute('SELECT COUNT(*) FROM ecommerce;').fetchall()[0][0]
print('Dataframe size: {}'.format(df_size))

# Load data
data = pd.read_sql('SELECT * FROM ecommerce;', engine).drop(columns=['index', 'id'], axis=1)

# date range in the dataset
print('Min_date: {0}\nMax_date: {1}\n\n'.format(data['invoicedate'].min(), data['invoicedate'].max()))

# RFM Analysis
# set snapshot date to 1 day after the max 
#snapshot_date = data['invoicedate'].max()+timedelta(days=1)
# Find total spent
#data['totalspent'] = data['quantity']*data['unitprice']

# Find recency, frequency, monetary value
#df=data.groupby('customerid').agg({
#    'invoicedate': lambda x: (snapshot_date-x.max()).days,
#    'invoiceno': 'count',
#    'totalspent': 'sum'
#})

#df.rename(columns={
#    'invoicedate': 'recency',
#    'invoiceno': 'frequency',
#    'totalspent': 'monetaryvalue'
#}, inplace=True)

query="""
SELECT 
    customerid,
    TIMESTAMPDIFF(DAY,
        MAX(invoicedate),
        (SELECT 
                DATE_ADD(MAX(invoicedate),
                        INTERVAL 1 DAY) AS snapshot_date
            FROM
                ecommerce)) AS recency,
    COUNT(quantity) AS frequency,
    ROUND(SUM(unitprice * quantity), 2) AS monetaryvalue
FROM
    ecommerce
GROUP BY customerid
ORDER BY customerid;
"""
df=pd.read_sql(query, engine)
df= df.set_index('customerid')

df.head()