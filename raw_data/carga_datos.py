import os
import sys
import numpy as np
import datetime
import itertools
sys.path.append('C:\\Program Files\\Continuum\\Anaconda3\\Lib\\site-packages')
import matplotlib.pyplot as plt
#import pandas as pd
import pandas_datareader.data as web
import quandl
 
symbol_list = ["WFC", "JPM", "BAC", "C","GS", "SPY","MS", "BK","^IXIC", "^DJI", "^GDAXI", "^FTSE","^FCHI", "^N225","^HSI", "^AXJO","^GSPC"]
indices_list = ["^IXIC", "^DJI", "^GDAXI", "^FTSE","^FCHI", "^N225","^HSI", "^AXJO"]
index_list=["^GSPC"]
comodities=["LBMA/SILVER", "LBMA/GOLD", "JOHNMATT/PLAT","OPEC/ORB","CUR/EUR","CUR/AUD","CUR/GBP","CUR/JPY"]
monedas=["CUR/EUR","CUR/AUD","CUR/GBP","CUR/JPY"]


fechastart='2007-01-01'
fechaend='2017-01-01'
	
for symbol in symbol_list:
	#df=web.dataframe
	df = web.DataReader(symbol, start=fechastart, end=fechaend,data_source='yahoo')
	path = os.path.dirname(os.path.realpath(__file__))
	print ("Loading csv..." + str(symbol))
	file_path = path +  "\\raw_data\\" + symbol + ".csv"
	df.to_csv(file_path, sep=',')
	
	
for symbol in comodities:
	df =  quandl.get(symbol, trim_start = fechastart, trim_end =fechaend, authtoken="_N85bWLCNCWz14smKHSi")
	path = os.path.dirname(os.path.realpath(__file__))
	pos=symbol.find("/")
	name=symbol[pos+1:]
	print ("Loading csv..." + str(name))
	file_path = path +  "\\raw_data\\" + name + ".csv"
	df.to_csv(file_path, sep=',')
	
#NASDAQ Composite (^IXIC Yahoo Finance)
#Dow Jones Industrial Average (^DJI Quandl)
#Frankfurt DAX (^GDAXI Yahoo Finance)
#London FTSE-100 (^FTSE Yahoo Finance)
#Paris CAC 40 (^FCHI Yahoo Finance)
#Tokyo Nikkei-225 (^N225 Yahoo Finance)
#Hong Kong Hang Seng (^HSI Yahoo Finance)
#Australia ASX-200 (^AXJO Yahoo Finance)