# https://www.kaggle.com/carrie1/ecommerce-data
# https://www.kaggle.com/miljan/customer-segmentation
import pandas as pd 
import sqlalchemy as db
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm import sessionmaker
import os
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

class Ecommerce:
    def __init__(self, datapath):
        self.data_types={
        #'InvoiceNo': int,
        #'StockCode': str,
        #'Description': str,
        'Quantity': int,
        'UnitPrice': float,
        #'CustomerID': int,
        'Country': str
        }

        try:
            self.df = pd.read_csv(datapath,\
                            encoding = "ISO-8859-1",\
                            header=0,\
                            sep=',',\
                            parse_dates=['InvoiceDate'],\
                            dtype=data_types,\
                            error_bad_lines=False,\
                            warn_bad_lines=True)
        except pd.io.common.CParserError:
            print("Your data contained rows that could not be parsed.")

        self.df.columns = self.df.columns.str.lower()

    def show_preprocessing(self):
        # Check Nan Values 
        nan_ratio = []
        for col in df.columns:
            nan_ratio=np.round(self.df[col].isnull().sum()/self.df.shape[0]*100, 2)
            print('{0}: {1:2f}'.format(col, nan_ratio))

        plt.figure(figsize=(5, 5))
        self.df.isnull().mean(axis=0).plot.barh()
        plt.title("Ratio of missing values per columns")
        plt.xlabel('Missing value (%)')
        plt.show()

    def preprocessing(self):
        """
        Find database information
        """

        # fillna description
        self.df['description'] = self.df['description'].fillna('UNKNOWN')

        # drop missing customerID
        self.df.dropna(subset=['customerid'], inplace=True)
        self.df['customerid'] = self.df['customerid'].astype(int)
        # generate index
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'index': 'id'})

    def send_to_mysql(self):
        # Load MYSql connector 
        SQL_USR, SQL_PSW= os.environ['SQL_USR'], os.environ['SQL_PSW']
        mysql_str = 'mysql+mysqlconnector://'+SQL_USR+':'+SQL_PSW+'@localhost:3306/'
        engine = db.create_engine(mysql_str)

        print('Create database Commerce')
        print('-'*30)
        # Create database diamonds
        con=engine.connect()
        con.execute('commit')
        con.execute('CREATE DATABASE if NOT EXISTS Commerce;')
        con.close()
        print('Done.\n')

        print('Create tables.')
        print('-'*30)
        # Select diamonds database
        engine = db.create_engine(mysql_str+'Commerce')
        con=engine.connect()

        Base=declarative_base()

        class Ecommerce(Base):
            """
            Class for creating the physical table.
            """
            __tablename__ = 'ecommerce'
            
            id=Column(Integer, primary_key=True)
            invoiceno=Column(String(16))
            stockcode=Column(String(16))
            description=Column(String(255))
            quantity=Column(Integer)
            invoicedate=Column(Date)
            unitprice=Column(Float)
            customerid=Column(Integer)
            country=Column(String(64))
            
            def __init__(self, id, invoiceno, stockcode, description, quantity, invoicedate, unitprice, customerid, country):
                self.id=id
                self.invoiceno=invoiceno
                self.stockcode=stockcode
                self.description=description
                self.quantity=quantity
                self.invoicedate=invoicedate
                self.unitprice=unitprice
                self.customerid=customerid
                self.country=country

        Base.metadata.create_all(engine)

        print('Done.\n')

        # Copy data
        print('Insert data from CSV to SQL.')
        print('-'*30)
        # Copy data into database
        # Insert data to database
        engine = db.create_engine(mysql_str+'Commerce')
        con=engine.connect()

        self.df.to_sql(name='ecommerce',\
                    con=engine,\
                    if_exists='replace')

        print('Done.\n')

data = Ecommerce('../data/data.csv')
data.preprocessing()
data.df.head()
data.send_to_mysql()