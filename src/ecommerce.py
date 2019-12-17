import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#import pandas_profiling
from sklearn.pipeline import Pipeline 
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

df=pd.read_sql('SELECT * FROM rfm;', engine)
df= df.set_index('customerid')

# create labels for rfm segments
r_labels = range(4, 0, -1)
f_labels = range(1, 5)
m_labels = range(1, 5)
df['R'] = pd.qcut(df['recency'],
                      4,
                      labels = r_labels)
df['F'] = pd.qcut(df['frequency'],
                      4,
                      labels = f_labels)
df['M'] = pd.qcut(df['monetary_value'],
                      4,
                      labels = m_labels)

def join_rfm(x):
    """
    Create the rfm segment.
    """
    return str(x['R'])+str(x['F'])+str(x['M'])

df['RFM_segment'] = df.apply(join_rfm, axis=1)
df['RFM_score'] = df[['R', 'F', 'M']].sum(axis=1)
df.head()

# Segment analysis
segments = df.groupby('RFM_segment').size().sort_values(ascending=False)[:10]
plt.figure()
sns.barplot(segments.index,\
            segments)
plt.show()

# Summary metrics for RFM Score
df.groupby('RFM_score').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary_value': ['mean', 'count']
}).round(1)\
.sort_values(by='RFM_score', ascending=False)

# Create custom segments
def segment_me(df):
    """
    Create custom segments
    """
    if df['RFM_score'] >= 9:
        return 'Gold'
    elif (df['RFM_score']>=5) and (df['RFM_score']<9):
        return 'Silver'
    else:
        return 'Bronze'
df['general_segment'] = df.apply(segment_me, axis=1)        

# Summary metrics for general_segments
df.groupby('general_segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary_value': ['mean', 'count']
}).round(1)


# KMeans to build customer segments 

