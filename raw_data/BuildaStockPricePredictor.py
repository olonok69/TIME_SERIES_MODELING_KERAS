import os
import sys
import numpy as np
import datetime
import itertools
sys.path.append('C:\\Program Files\\Continuum\\Anaconda3\\Lib\\site-packages')
import matplotlib.pyplot as plt
import pandas as pd
#import library as mio



def get_data(symbols, dates):
    #"""Read stock data (adjusted close) for given symbols from CSV files."""
	df_final = pd.DataFrame(index=dates)
	i=0
	for symbol in symbols:
		path = os.path.dirname(os.path.realpath(__file__))
		print ("Loading csv..." + str(symbol))
		file_path = path +  "\\raw_data\\" + symbol + ".csv"
		df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "Adj Close"], na_values=["nan"])
		df_temp = df_temp.rename(columns={"Adj Close": symbol})
		df_final = df_final.join(df_temp)
		i+=1
		if i == 1:  # drop dates SPY did not trade
			df_final = df_final.dropna(subset=[symbol])

	return df_final
	
def write_Excel(__df_r1, filename):
	print ("Printing Report...")
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\raw_data\\" + filename
	writer = pd.ExcelWriter(total_file)

	__df_r1.to_excel(writer,'Adj_Close')
	writer.save()
                
def MOM(df, n, name_s):  
    M = pd.Series(df[name_s].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df         

def Volatity_n(df, symbol,n):
	#M = pd.Series(df[symbol].pct_change(n).std(), name = 'Volatility_' + str(n)) 
	df["Log_Ret"] = np.log(df[symbol] / df[symbol].shift(1))
	df["Volatility_"+ str(n)] = df["Log_Ret"].rolling(window=n, center=False).std() * np.sqrt(252)
	del df["Log_Ret"]
	return df 

def mergeDataframes(dataset1,dataset2, symbol1,symbol2):#, cut):
	data = pd.concat([dataset1, dataset2], axis=1).dropna()  # and remove NaNs, if any
	data.columns = [symbol1, symbol2]
	return data
	

def normalize(df, symbol):
	result = df.copy()
	for symbol in df.columns:
		max_value = df[symbol].max()
		min_value = df[symbol].min()
		mean_value= df[symbol].mean()
		result[symbol] = (df[symbol] - mean_value) / (max_value - min_value)
	return result
	
def adjusted_close(df, symbol,n):
	#df_final = pd.DataFrame(index=dates)
	M = pd.Series(df[symbol].pct_change(n), name = 'Adjusted_' + str(n))  
	df = df.join(M)
	return df

def main():

	symbol_list = ["WFC", "JPM", "BAC", "C","GS", "SPY","MS", "BK"]
	#indices_list = ["^IXIC", "^DJI", "^GDAXI", "^FTSE","^FCHI", "^N225","^HSI", "^AXJO"]
	indices_list = ["^AXJO"]
	index_list=["^GSPC"]
	numDaysArray = [21,42, 63, 126, 252] # 1 montn,2months, 3months, 6 months, year
	            
	start_date = "2007-01-01"
	end_date = "2017-01-01"
	dates = pd.date_range(start_date, end_date)  # date range as index
	df_data = get_data(symbol_list, dates)  # get data for each symbol
	df_index = get_data(index_list, dates)
	df_indices = get_data(indices_list, dates)  # get data for each symbol
	#df_comodities=mio.get_web_quandl("OPEC/ORB", "OIL")
	df_merged=mergeDataframes(df_index,df_indices,"^GSPC","^AXJO")
	
	for numDayIndex in numDaysArray:
		df_indices=normalize(df_indices,"^AXJO")
		df_indices=adjusted_close(df_indices,"^AXJO",numDayIndex) #calculate Adjusted Price in day t -n days from Array numDaysArray
		df_indices=MOM(df_indices,numDayIndex, "^AXJO") #calculate momentum in n days from Array numDaysArray
		df_index=normalize(df_index,"^GSPC")
		df_index=adjusted_close(df_index,"^GSPC",numDayIndex) #calculate Adjusted Price in day t -n days from Array numDaysArray
		df_index=MOM(df_index,numDayIndex, "^GSPC") #calculate momentum in n days from Array numDaysArray
		#df_index=Volatity_n(df_index,"^GSPC",numDayIndex)
	#df_merged=mergeDataframes(datasets,1)
	#df_merged.tail()
	#write_Excel(df_data, "prices_stock.xlsx")
	#write_Excel(df_index, "prices_index.xlsx")
	#write_Excel(df_indices, "prices_indices.xlsx")
	#write_Excel(df_comodities, "prices_comodities.xlsx")
	write_Excel(df_merged, "prices_merged.xlsx")
                

main()
