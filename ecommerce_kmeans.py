import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#import pandas_profiling
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from sklearn.cluster import KMeans

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

df.head()

# Check skewness
#for col in df.columns:
#    sns.distplot(df[col])
#    plt.title(col)
#    plt.show()

# Remove skewness 
df_enc = df.copy()
df_enc.head()

for col in df_enc.columns:
    df_enc[col] = df_enc[col]+np.abs(df_enc[col])+1
    df_enc[col] = df_enc[col].apply(lambda x: np.log(x))

# Preprocessing pipeline
preprocessor=Pipeline([
    ('std', StandardScaler())
])

X = preprocessor.fit_transform(df_enc.values)